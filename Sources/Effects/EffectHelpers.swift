import Foundation
import Metal
import simd
import Accelerate

// MARK: - Blur weights

enum GaussianWeights {
    /// Normalized one-sided weights, centre first, `radius + 1` entries.
    static func make(radius: Int, sigma: Float? = nil) -> [Float] {
        let s = sigma ?? max(Float(radius) / 2, 0.5)
        var w = (0...radius).map { exp(-Float($0 * $0) / (2 * s * s)) }
        let total = w[0] + 2 * w.dropFirst().reduce(0, +)
        for i in w.indices { w[i] /= total }
        return w
    }
    static func box(radius: Int) -> [Float] {
        Array(repeating: 1 / Float(2 * radius + 1), count: radius + 1)
    }
}

// MARK: - Homographies (Corner Pin, Basic 3D)

enum Homography {
    /// Solves the 3x3 homography H that maps each `s[i]` to `d[i]` (4 point pairs). Returns nil if degenerate.
    static func solve(from s: [SIMD2<Double>], to d: [SIMD2<Double>]) -> simd_double3x3? {
        guard s.count == 4, d.count == 4 else { return nil }
        var a = [[Double]](repeating: [Double](repeating: 0, count: 9), count: 8)
        for i in 0..<4 {
            let x = s[i].x, y = s[i].y, u = d[i].x, v = d[i].y
            a[2 * i]     = [x, y, 1, 0, 0, 0, -u * x, -u * y, u]
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -v * x, -v * y, v]
        }
        for c in 0..<8 {                                   // Gauss-Jordan with partial pivoting
            var pivot = c
            for r in (c + 1)..<8 where abs(a[r][c]) > abs(a[pivot][c]) { pivot = r }
            if abs(a[pivot][c]) < 1e-12 { return nil }
            a.swapAt(c, pivot)
            for r in 0..<8 where r != c {
                let f = a[r][c] / a[c][c]
                if f == 0 { continue }
                for k in c...8 { a[r][k] -= f * a[c][k] }
            }
        }
        let h = (0..<8).map { a[$0][8] / a[$0][$0] }
        return simd_double3x3(columns: (SIMD3<Double>(h[0], h[3], h[6]),
                                        SIMD3<Double>(h[1], h[4], h[7]),
                                        SIMD3<Double>(h[2], h[5], 1)))
    }

    static func toFloat(_ m: simd_double3x3) -> simd_float3x3 {
        func col(_ c: SIMD3<Double>) -> SIMD3<Float> { SIMD3<Float>(Float(c.x), Float(c.y), Float(c.z)) }
        return simd_float3x3(columns: (col(m.columns.0), col(m.columns.1), col(m.columns.2)))
    }

    /// Corner Pin matrix for `fx_corner_pin` (maps output uv to source uv); load it with `EffectValues.setCornerPin`.
    /// `corners` are where the layer's top-left, top-right, bottom-right, bottom-left land, in output uv (0...1, y down).
    static func cornerPin(corners: [SIMD2<Float>]) -> simd_float3x3? {
        let unitSquare: [SIMD2<Double>] = [SIMD2(0, 0), SIMD2(1, 0), SIMD2(1, 1), SIMD2(0, 1)]
        let dst = corners.map { SIMD2<Double>(Double($0.x), Double($0.y)) }
        guard let h = solve(from: unitSquare, to: dst) else { return nil }
        return toFloat(h.inverse)
    }

    /// CPU-side Basic 3D homography (for hit-testing / on-canvas handles; `fx_basic3d` intersects the plane in the
    /// shader itself and does not use this). A plane (height 1, width `layerAspect`) rotated by yaw/pitch/roll
    /// (radians) at `distance` in front of a pinhole camera. Returns output-uv -> plane and the surface normal.
    static func basic3D(layerAspect: Float, outputAspect: Float, yaw: Float, pitch: Float, roll: Float,
                        distance: Float, fovY: Float) -> (outToPlane: simd_float3x3, normal: SIMD3<Float>)? {
        let q = simd_quatf(angle: roll, axis: SIMD3<Float>(0, 0, 1))
              * simd_quatf(angle: pitch, axis: SIMD3<Float>(1, 0, 0))
              * simd_quatf(angle: yaw, axis: SIMD3<Float>(0, 1, 0))
        let r = simd_float3x3(q)
        let f = 1 / tan(fovY / 2)
        // camera space -> output uv
        let k = simd_float3x3(columns: (SIMD3<Float>(0.5 * f / outputAspect, 0, 0),
                                        SIMD3<Float>(0, -0.5 * f, 0),
                                        SIMD3<Float>(0.5, 0.5, 1)))
        // plane (X, Y, 1) -> camera space
        let cam = simd_float3x3(columns: (r.columns.0, r.columns.1, SIMD3<Float>(0, 0, distance)))
        // plane uv -> plane (X, Y, 1)
        let w = layerAspect, h: Float = 1
        let muv = simd_float3x3(columns: (SIMD3<Float>(w, 0, 0), SIMD3<Float>(0, -h, 0), SIMD3<Float>(-w / 2, h / 2, 1)))
        let forward = k * cam * muv
        guard abs(forward.determinant) > 1e-9 else { return nil }
        let n = r * SIMD3<Float>(0, 0, -1)               // towards the camera
        return (forward.inverse, SIMD3<Float>(n.x, n.y, -n.z))
    }
}

// MARK: - Camera math (3D Camera effect)

enum CameraMath {
    /// Right-handed perspective projection with Metal's 0...1 clip depth (camera looks down -Z).
    static func perspective(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5), xs = ys / aspect, zs = far / (near - far)
        return simd_float4x4(columns: (SIMD4<Float>(xs, 0, 0, 0), SIMD4<Float>(0, ys, 0, 0),
                                       SIMD4<Float>(0, 0, zs, -1), SIMD4<Float>(0, 0, near * zs, 0)))
    }

    static func lookAt(eye: SIMD3<Float>, target: SIMD3<Float>, up: SIMD3<Float> = SIMD3<Float>(0, 1, 0)) -> simd_float4x4 {
        let z = simd_normalize(eye - target)
        let x = simd_normalize(simd_cross(up, z))
        let y = simd_cross(z, x)
        return simd_float4x4(columns: (SIMD4<Float>(x.x, y.x, z.x, 0), SIMD4<Float>(x.y, y.y, z.y, 0),
                                       SIMD4<Float>(x.z, y.z, z.z, 0),
                                       SIMD4<Float>(-simd_dot(x, eye), -simd_dot(y, eye), -simd_dot(z, eye), 1)))
    }

    static func translation(_ t: SIMD3<Float>) -> simd_float4x4 {
        var m = matrix_identity_float4x4
        m.columns.3 = SIMD4<Float>(t, 1)
        return m
    }

    /// World units are pixels, the comp centre is the origin, +Y is up. A layer at z = 0 with its anchor at the
    /// layer centre renders pixel-for-pixel when `fovY` matches the camera built here.
    static func comp2DCamera(compSize: SIMD2<Float>, fovY: Float = 0.6, near: Float = 1, far: Float = 20000)
        -> (viewProj: simd_float4x4, invViewProj: simd_float4x4, eyeZ: Float) {
        let eyeZ = (compSize.y / 2) / tan(fovY / 2)
        let view = lookAt(eye: SIMD3<Float>(0, 0, eyeZ), target: .zero)
        let proj = perspective(fovY: fovY, aspect: compSize.x / compSize.y, near: near, far: far)
        let vp = proj * view
        return (vp, vp.inverse, eyeZ)
    }

    /// `Camera3DParams.worldToLayer`. Layer space is pixels with the origin at the top-left corner (y down);
    /// world space is y-up, so the layer's y axis is flipped around the anchor.
    static func worldToLayer(position: SIMD3<Float>, rotation: simd_quatf, scale: Float, anchor: SIMD2<Float>) -> simd_float4x4 {
        let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, 1, 1))
        let s = simd_float4x4(diagonal: SIMD4<Float>(scale, scale, scale, 1))
        let model = translation(position) * simd_float4x4(rotation) * s * flip * translation(SIMD3<Float>(-anchor.x, -anchor.y, 0))
        return model.inverse
    }
}

// MARK: - Curves LUT

enum CurveLUT {
    /// Monotone cubic (Fritsch-Carlson) through control points in 0...1; no overshoot between points.
    static func samples(points: [SIMD2<Float>], count: Int = 256) -> [Float] {
        let p = points.sorted { $0.x < $1.x }
        let n = p.count
        guard n >= 2 else { return (0..<count).map { Float($0) / Float(count - 1) } }

        var delta = [Float](repeating: 0, count: n - 1)
        for i in 0..<(n - 1) { delta[i] = (p[i + 1].y - p[i].y) / max(p[i + 1].x - p[i].x, 1e-6) }
        var slope = [Float](repeating: 0, count: n)
        slope[0] = delta[0]; slope[n - 1] = delta[n - 2]
        if n > 2 {
            for i in 1..<(n - 1) { slope[i] = delta[i - 1] * delta[i] <= 0 ? 0 : (delta[i - 1] + delta[i]) / 2 }
        }
        for i in 0..<(n - 1) {
            if delta[i] == 0 { slope[i] = 0; slope[i + 1] = 0; continue }
            let a = slope[i] / delta[i], b = slope[i + 1] / delta[i]
            let s = a * a + b * b
            if s > 9 { let t = 3 / s.squareRoot(); slope[i] = t * a * delta[i]; slope[i + 1] = t * b * delta[i] }
        }
        return (0..<count).map { k in
            let x = Float(k) / Float(count - 1)
            if x <= p[0].x { return p[0].y }
            if x >= p[n - 1].x { return p[n - 1].y }
            var i = 0
            while i < n - 2 && x > p[i + 1].x { i += 1 }
            let h = p[i + 1].x - p[i].x
            let t = (x - p[i].x) / h, t2 = t * t, t3 = t2 * t
            let y = (2 * t3 - 3 * t2 + 1) * p[i].y + (t3 - 2 * t2 + t) * h * slope[i]
                  + (-2 * t3 + 3 * t2) * p[i + 1].y + (t3 - t2) * h * slope[i + 1]
            return min(max(y, 0), 1)
        }
    }

    private static func lookup(_ lut: [Float], _ x: Float) -> Float {
        let f = min(max(x, 0), 1) * Float(lut.count - 1)
        let i = Int(f), j = min(i + 1, lut.count - 1)
        return lut[i] + (lut[j] - lut[i]) * (f - Float(i))
    }

    /// 256x1 RGBA16F texture for the `fx_curves` kernel: per-channel curve followed by the master curve.
    static func makeTexture(device: MTLDevice, master: [SIMD2<Float>], red: [SIMD2<Float>], green: [SIMD2<Float>],
                            blue: [SIMD2<Float>]) -> MTLTexture? {
        let m = samples(points: master)
        let r = samples(points: red), g = samples(points: green), b = samples(points: blue)
        var data = [Float16](repeating: 1, count: 256 * 4)
        for i in 0..<256 {
            data[i * 4 + 0] = Float16(lookup(m, r[i]))
            data[i * 4 + 1] = Float16(lookup(m, g[i]))
            data[i * 4 + 2] = Float16(lookup(m, b[i]))
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 256, height: 1, mipmapped: false)
        d.usage = .shaderRead
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        data.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, 256, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 256 * 8) }
        return t
    }
}

// MARK: - .cube 3D LUT loader (Lumetri creative LUTs)

enum CubeLUT {
    static func load(_ url: URL, device: MTLDevice) -> MTLTexture? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var size = 0
        var rgba: [Float16] = []
        for line in text.split(whereSeparator: \.isNewline) {
            let l = line.trimmingCharacters(in: .whitespaces)
            if l.isEmpty || l.hasPrefix("#") { continue }
            if l.hasPrefix("LUT_3D_SIZE") { size = Int(l.split(separator: " ").last ?? "") ?? 0; continue }
            if l.first?.isLetter == true { continue }                       // TITLE, DOMAIN_MIN, ...
            let v = l.split(whereSeparator: { $0 == " " || $0 == "\t" }).compactMap { Float($0) }
            if v.count == 3 { rgba += [Float16(v[0]), Float16(v[1]), Float16(v[2]), 1] }
        }
        guard size > 1, rgba.count == size * size * size * 4 else { return nil }
        let d = MTLTextureDescriptor()
        d.textureType = .type3D; d.pixelFormat = .rgba16Float
        d.width = size; d.height = size; d.depth = size; d.usage = .shaderRead
        guard let t = device.makeTexture(descriptor: d) else { return nil }
        rgba.withUnsafeBytes {                                              // .cube order: red fastest = texture x fastest
            t.replace(region: MTLRegionMake3D(0, 0, 0, size, size, size), mipmapLevel: 0, slice: 0,
                      withBytes: $0.baseAddress!, bytesPerRow: size * 8, bytesPerImage: size * size * 8)
        }
        return t
    }
}

// MARK: - Audio Spectrum input

/// Turns PCM into the normalized 0...1 magnitude bins that `fx_audio_spectrum` reads from buffer(1).
final class SpectrumAnalyzer {
    let size: Int
    private let log2n: vDSP_Length
    private let setup: FFTSetup
    private var window: [Float]

    init?(size: Int = 2048) {
        guard size > 8, size & (size - 1) == 0 else { return nil }    // power of two
        self.size = size
        log2n = vDSP_Length(log2(Double(size)))
        guard let s = vDSP_create_fftsetup(log2n, FFTRadix(kFFTRadix2)) else { return nil }
        setup = s
        window = [Float](repeating: 0, count: size)
        vDSP_hann_window(&window, vDSP_Length(size), Int32(vDSP_HANN_NORM))
    }

    deinit { vDSP_destroy_fftsetup(setup) }

    /// `samples`: mono PCM (up to `size` frames used). Returns `size / 2` values in 0...1 (dB scaled from `floorDB`).
    func magnitudes(_ samples: [Float], floorDB: Float = -80) -> [Float] {
        let half = size / 2
        guard !samples.isEmpty else { return [Float](repeating: 0, count: half) }

        var windowed = [Float](repeating: 0, count: size)
        let count = min(size, samples.count)
        samples.withUnsafeBufferPointer { s in
            window.withUnsafeBufferPointer { w in
                vDSP_vmul(s.baseAddress!, 1, w.baseAddress!, 1, &windowed, 1, vDSP_Length(count))
            }
        }

        var real = [Float](repeating: 0, count: half)
        var imag = [Float](repeating: 0, count: half)
        var mags = [Float](repeating: 0, count: half)
        real.withUnsafeMutableBufferPointer { rp in
            imag.withUnsafeMutableBufferPointer { ip in
                var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                windowed.withUnsafeBufferPointer { wp in
                    wp.baseAddress!.withMemoryRebound(to: DSPComplex.self, capacity: half) { cp in
                        vDSP_ctoz(cp, 2, &split, 1, vDSP_Length(half))
                    }
                }
                vDSP_fft_zrip(setup, &split, 1, log2n, FFTDirection(FFT_FORWARD))
                vDSP_zvabs(&split, 1, &mags, 1, vDSP_Length(half))
            }
        }
        let norm = 1 / Float(size)
        return mags.map { m in
            let db = 20 * log10(max(m * norm, 1e-9))
            return min(max((db - floorDB) / -floorDB, 0), 1)
        }
    }

    /// Groups linear FFT bins into `bands` log-spaced bands (what the spectrum bars want).
    static func logBands(_ bins: [Float], bands: Int) -> [Float] {
        guard !bins.isEmpty, bands > 0 else { return [] }
        let top = Float(bins.count)
        return (0..<bands).map { b in
            let lo = Int(pow(top, Float(b) / Float(bands)) - 1)
            let hi = max(lo + 1, Int(pow(top, Float(b + 1) / Float(bands)) - 1))
            let slice = bins[min(lo, bins.count - 1)..<min(hi, bins.count)]
            return slice.max() ?? 0
        }
    }
}
