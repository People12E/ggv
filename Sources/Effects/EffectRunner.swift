import Foundation
import Metal
import simd

// =============================================================================
// EffectValues: the current parameter values of one effect instance.
// EffectRunner: executes an EffectDescriptor's pass graph on a command buffer.
//
// Frame data is premultiplied RGBA in .rgba16Float textures. Everything encodes onto the caller's command buffer
// and the caller's queue; the runner never commits. The texture pool assumes ONE command queue (Metal's hazard
// tracking then orders reuse of intermediates).
// =============================================================================

// MARK: - Values

struct EffectValues {
    let descriptor: EffectDescriptor
    private var storage: [String: [Float]]

    /// Starts at the descriptor's defaults.
    init(_ descriptor: EffectDescriptor) {
        self.descriptor = descriptor
        var s: [String: [Float]] = [:]
        for p in descriptor.params { s[p.name] = p.defaults }
        storage = s
    }

    /// Scalar access (first component). Unknown names read 0 and are ignored on write.
    subscript(name: String) -> Float {
        get { storage[name]?.first ?? 0 }
        set {
            guard let p = descriptor.params.first(where: { $0.name == name }), p.width == 1 else {
                assertionFailure("\(descriptor.id): '\(name)' is not a scalar parameter"); return
            }
            storage[name] = [newValue]
        }
    }

    func values(_ name: String) -> [Float] { storage[name] ?? [] }

    mutating func set(_ name: String, _ v: [Float]) {
        guard let p = descriptor.params.first(where: { $0.name == name }), p.width == v.count else {
            assertionFailure("\(descriptor.id): '\(name)' expects a different component count"); return
        }
        storage[name] = v
    }

    mutating func setPoint(_ name: String, _ x: Float, _ y: Float) { set(name, [x, y]) }
    mutating func setColor(_ name: String, _ c: SIMD4<Float>) { set(name, [c.x, c.y, c.z, c.w]) }

    /// Corner Pin: where the layer's top-left, top-right, bottom-right, bottom-left land, in comp uv (0...1, y down).
    /// Returns false (and leaves the matrix alone) when the quad is degenerate.
    @discardableResult
    mutating func setCornerPin(corners: [SIMD2<Float>]) -> Bool {
        guard let m = Homography.cornerPin(corners: corners) else { return false }
        var rowMajor: [Float] = []
        for r in 0..<3 { for c in 0..<3 { rowMajor.append(m[c][r]) } }   // simd_float3x3 is column-major: m[column][row]
        set("homography", rowMajor)
        return true
    }

    /// The kernel's `P` buffer: [passIndex, time, repeatIndex, tag, renderScale, params...].
    /// Pixel-unit parameters are multiplied by `renderScale` so a half-res proxy looks like the full-res render.
    func packed(passIndex: Int, repeatIndex: Int, tag: Float, time: Double, renderScale: Float) -> [Float] {
        var out: [Float] = [Float(passIndex), Float(time), Float(repeatIndex), tag, renderScale]
        out.reserveCapacity(FX_RESERVED_SLOTS + descriptor.slotCount)
        for p in descriptor.params {
            let v = storage[p.name] ?? p.defaults
            out.append(contentsOf: p.kind.scalesWithResolution ? v.map { $0 * renderScale } : v)
        }
        return out
    }

    /// A parameter after render-scale, for the runner's own use (repeat counts, particle counts).
    func scaledValue(_ name: String, renderScale: Float) -> Float {
        guard let p = descriptor.params.first(where: { $0.name == name }) else { return 0 }
        let v = storage[name]?.first ?? p.defaults.first ?? 0
        return p.kind.scalesWithResolution ? v * renderScale : v
    }
}

/// P[0...4] are reserved (see FXCommon.h).
let FX_RESERVED_SLOTS = 5

// MARK: - Errors

enum EffectRunnerError: LocalizedError {
    case missingFunction(String)
    case pipeline(String, Error)
    case noPasses(String)
    case missingBuffer(effect: String, needed: Int, got: Int)
    case badTexture(String)
    case unsupportedDevice

    var errorDescription: String? {
        switch self {
        case .missingFunction(let n): return "Metal function '\(n)' not found; are the FX_*.metal files in the app target?"
        case .pipeline(let n, let e): return "Could not build pipeline '\(n)': \(e.localizedDescription)"
        case .noPasses(let id): return "'\(id)' has no compute passes; use ParticleRenderer / Layer3DRenderer for it."
        case .missingBuffer(let id, let n, let g): return "'\(id)' needs \(n) extra buffer(s), got \(g)."
        case .badTexture(let m): return m
        case .unsupportedDevice: return "This device cannot run the effect pipeline."
        }
    }
}

// MARK: - Texture pool

/// Reuses intermediate .rgba16Float textures by size. Thread-safe.
final class TexturePool {
    private let device: MTLDevice
    private let lock = NSLock()
    private var free: [String: [MTLTexture]] = [:]

    init(device: MTLDevice) { self.device = device }

    private func key(_ w: Int, _ h: Int) -> String { "\(w)x\(h)" }

    func acquire(width: Int, height: Int) -> MTLTexture? {
        lock.lock()
        if var list = free[key(width, height)], let t = list.popLast() {
            free[key(width, height)] = list
            lock.unlock()
            return t
        }
        lock.unlock()
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        return device.makeTexture(descriptor: d)
    }

    func release(_ t: MTLTexture) {
        lock.lock(); defer { lock.unlock() }
        free[key(t.width, t.height), default: []].append(t)
    }

    /// Drop everything (call on memory warnings or when the comp size changes).
    func purge() { lock.lock(); free.removeAll(); lock.unlock() }
}

// MARK: - Runner

final class EffectRunner {
    let device: MTLDevice
    let library: MTLLibrary
    let pool: TexturePool

    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let pipelineLock = NSLock()
    private let dummy2D: MTLTexture
    private let dummy3D: MTLTexture

    /// `library` defaults to the app's default Metal library (the one built from FX_*.metal).
    init(device: MTLDevice, library: MTLLibrary? = nil) throws {
        guard let lib = library ?? device.makeDefaultLibrary() else { throw EffectRunnerError.unsupportedDevice }
        self.device = device
        self.library = lib
        self.pool = TexturePool(device: device)

        func zeroTexture(type: MTLTextureType) throws -> MTLTexture {
            let d = MTLTextureDescriptor()
            d.textureType = type; d.pixelFormat = .rgba16Float
            d.width = 1; d.height = 1; d.depth = 1
            d.usage = .shaderRead; d.storageMode = .shared
            guard let t = device.makeTexture(descriptor: d) else { throw EffectRunnerError.unsupportedDevice }
            let zeros = [UInt16](repeating: 0, count: 4)
            zeros.withUnsafeBytes {
                if type == .type3D {
                    t.replace(region: MTLRegionMake3D(0, 0, 0, 1, 1, 1), mipmapLevel: 0, slice: 0,
                              withBytes: $0.baseAddress!, bytesPerRow: 8, bytesPerImage: 8)
                } else {
                    t.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 8)
                }
            }
            return t
        }
        dummy2D = try zeroTexture(type: .type2D)
        dummy3D = try zeroTexture(type: .type3D)
    }

    func pipeline(_ kernel: String) throws -> MTLComputePipelineState {
        pipelineLock.lock(); defer { pipelineLock.unlock() }
        if let p = pipelines[kernel] { return p }
        guard let fn = library.makeFunction(name: kernel) else { throw EffectRunnerError.missingFunction(kernel) }
        do {
            let p = try device.makeComputePipelineState(function: fn)
            pipelines[kernel] = p
            return p
        } catch { throw EffectRunnerError.pipeline(kernel, error) }
    }

    /// Builds every pipeline the catalog uses up front (call off the main thread at launch to avoid first-use hitches).
    func warmUp() throws {
        for e in EffectCatalog.all { for p in e.passes { _ = try pipeline(p.kernel) } }
    }

    /// One scheduled dispatch of the expanded pass graph.
    private struct Step { let pass: PassDescriptor; let passIndex: Int; let repeatIndex: Int }

    private func expand(_ effect: EffectDescriptor, values: EffectValues, hasAux: Bool, renderScale: Float) -> [Step] {
        var steps: [Step] = []
        for (i, pass) in effect.passes.enumerated() {
            if pass.skipIfNoAux && !hasAux { continue }
            let n: Int
            switch pass.repeats {
            case .none: n = 1
            case .count(let name): n = min(max(Int(values[name].rounded()), 1), 16)
            case .kawase(let name):
                let radius = max(values.scaledValue(name, renderScale: renderScale), 0)
                n = min(max(Int(ceil(radius.squareRoot())), 1), 12)
            }
            for r in 0..<n { steps.append(Step(pass: pass, passIndex: i, repeatIndex: r)) }
        }
        return steps
    }

    /// Encodes `effect` reading `input` (and optionally `aux`, plus the descriptor's extra buffers) and writing `output`.
    ///
    /// - input / output: .rgba16Float; `output` needs `.shaderWrite`, `input` needs `.shaderRead`. Same size.
    /// - aux: the secondary layer / depth map / plate / LUT the descriptor's `aux` kind asks for (3D texture for Lumetri's look).
    /// - renderScale: proxy ratio (1 = full resolution, 0.5 = half) so pixel-unit parameters keep their on-screen size.
    func encode(_ effect: EffectDescriptor,
                values: EffectValues,
                input: MTLTexture,
                aux: MTLTexture? = nil,
                buffers: [MTLBuffer] = [],
                output: MTLTexture,
                time: Double,
                renderScale: Float = 1,
                commandBuffer: MTLCommandBuffer) throws {
        guard !effect.passes.isEmpty else { throw EffectRunnerError.noPasses(effect.id) }
        guard buffers.count >= effect.extraBuffers else {
            throw EffectRunnerError.missingBuffer(effect: effect.id, needed: effect.extraBuffers, got: buffers.count)
        }
        guard input.width == output.width, input.height == output.height else {
            throw EffectRunnerError.badTexture("\(effect.id): input and output must be the same size")
        }
        let steps = expand(effect, values: values, hasAux: aux != nil, renderScale: renderScale)
        guard !steps.isEmpty else { // every pass was optional and skipped: pass the image through
            if let blit = commandBuffer.makeBlitCommandEncoder() {
                blit.copy(from: input, to: output); blit.endEncoding()
            }
            return
        }
        guard let enc = commandBuffer.makeComputeCommandEncoder() else { throw EffectRunnerError.unsupportedDevice }
        enc.label = effect.name

        var current: MTLTexture = input
        var ownedIntermediate: MTLTexture? = nil          // pooled texture `current` points at, to release after use
        defer { enc.endEncoding() }

        for (n, step) in steps.enumerated() {
            let isLast = n == steps.count - 1
            let target: MTLTexture
            var pooled: MTLTexture? = nil
            if isLast {
                target = output
            } else {
                let w = max(1, Int((Float(output.width) * step.pass.scale).rounded()))
                let h = max(1, Int((Float(output.height) * step.pass.scale).rounded()))
                guard let t = pool.acquire(width: w, height: h) else { throw EffectRunnerError.unsupportedDevice }
                target = t; pooled = t
            }

            let pso = try pipeline(step.pass.kernel)
            enc.setComputePipelineState(pso)
            enc.setTexture(current, index: 0)
            enc.setTexture(target, index: 1)

            switch step.pass.aux {
            case .none:     enc.setTexture(dummy2D, index: 2)
            case .original: enc.setTexture(input, index: 2)
            case .userAux:  enc.setTexture(aux ?? (effect.aux == .lut3D ? dummy3D : dummy2D), index: 2)
            }

            var p = values.packed(passIndex: step.passIndex, repeatIndex: step.repeatIndex, tag: step.pass.tag,
                                  time: time, renderScale: renderScale)
            let pBytes = p.count * MemoryLayout<Float>.stride
            enc.setBytes(&p, length: pBytes, index: 0)
            for b in 0..<effect.extraBuffers { enc.setBuffer(buffers[b], offset: 0, index: 1 + b) }

            let w = pso.threadExecutionWidth
            let h = max(1, pso.maxTotalThreadsPerThreadgroup / w)
            enc.dispatchThreads(MTLSize(width: target.width, height: target.height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))

            if let done = ownedIntermediate { pool.release(done) }   // consumed by this step; hazard tracking orders reuse
            ownedIntermediate = pooled
            current = target
        }
    }

    /// Convenience: allocate a transient output texture the size of `input` (release it with `pool.release`).
    func makeOutput(like input: MTLTexture) -> MTLTexture? {
        pool.acquire(width: input.width, height: input.height)
    }
}
