import Foundation
import Metal
import simd

// All kernel parameter structs (GlowParams, SeparableParams, ...) come from Metal/BPShaderTypes.h via the
// bridging header, so Swift and MSL always agree on layout. Build them as `var p = GlowParams(); p.x = ...`.

// MARK: - Catalogue

enum EffectCategory: String, CaseIterable {
    case blurSharpen = "Blur & Sharpen"
    case colorCorrection = "Color Correction"
    case distort = "Distort"
    case generate = "Generate"
    case simulation = "Simulation"
    case stylize = "Stylize"
    case perspective = "Perspective"
    case keying = "Keying"
}

/// How an effect is driven.
enum EffectDriver {
    case single       // one dispatch through EffectRunner.encode
    case multiPass    // composed from several dispatches: see EffectGraphs.swift
    case simulation   // stateful: ParticleSystem / WaveWorld in ParticleSystem.swift
}

struct EffectSpec {
    let kernel: String          // primary kernel (multi-pass effects use others as well)
    let category: EffectCategory
    let displayName: String
    let inputTextures: Int      // textures bound before the output (the output is always bound last)
    let usesParams: Bool        // params struct at buffer(0)
    let extraBuffers: Int       // additional buffers starting at buffer(1)
    let driver: EffectDriver
}

private func makeSpec(_ kernel: String, _ category: EffectCategory, _ name: String, textures: Int,
                      params: Bool = true, buffers: Int = 0, driver: EffectDriver = .single) -> EffectSpec {
    EffectSpec(kernel: kernel, category: category, displayName: name, inputTextures: textures,
               usesParams: params, extraBuffers: buffers, driver: driver)
}

/// The 51 effects of the B.P library, in the order of the product spec.
enum EffectID: String, CaseIterable, Identifiable {
    // Blur & Sharpen
    case lensBlur, radialBlur, smartBlur, directionalBlur, fastBoxBlur, gaussianBlur, unsharpMask
    // Color Correction
    case curves, levels, hueSaturation, lumetri, photoFilter, colorBalance, tint
    // Distort
    case displacementMap, waveWarp, polarCoordinates, cornerPin, meshWarp, offset, spherize
    // Generate
    case grid, fourColorGradient, fractalNoise, stroke, audioSpectrum, cellPattern, checkerboard
    // Simulation
    case ccParticleWorld, shatter, waveWorld, foam, particlePlayground, ccParticleSystemsII
    // Stylize
    case glow, saberEdgeGlow, strobeLight, mosaic, findEdges, threshold, cartoon
    // Perspective
    case camera3D, bevelAlpha, dropShadow, basic3D, bevelEdges
    // Keying
    case keylight, lumaKey, colorRange, differenceMatte, spillSuppressor

    var id: String { rawValue }
    var category: EffectCategory { spec.category }
    var displayName: String { spec.displayName }

    var spec: EffectSpec {
        switch self {
        // Blur & Sharpen
        case .lensBlur:         return makeSpec("blurLens", .blurSharpen, "Camera Lens Blur", textures: 2)
        case .radialBlur:       return makeSpec("blurRadial", .blurSharpen, "Radial Blur", textures: 1)
        case .smartBlur:        return makeSpec("blurSmart", .blurSharpen, "Smart Blur", textures: 1)
        case .directionalBlur:  return makeSpec("blurDirectional", .blurSharpen, "Directional Blur", textures: 1)
        case .fastBoxBlur:      return makeSpec("blurSeparable", .blurSharpen, "Fast Box Blur", textures: 1, buffers: 1, driver: .multiPass)
        case .gaussianBlur:     return makeSpec("blurSeparable", .blurSharpen, "Gaussian Blur", textures: 1, buffers: 1, driver: .multiPass)
        case .unsharpMask:      return makeSpec("unsharpMask", .blurSharpen, "Unsharp Mask", textures: 2, driver: .multiPass)
        // Color Correction
        case .curves:           return makeSpec("colorCurves", .colorCorrection, "Curves", textures: 2, params: false)
        case .levels:           return makeSpec("colorLevels", .colorCorrection, "Levels", textures: 1)
        case .hueSaturation:    return makeSpec("colorHueSaturation", .colorCorrection, "Hue/Saturation", textures: 1)
        case .lumetri:          return makeSpec("colorLumetri", .colorCorrection, "Lumetri Color", textures: 2)
        case .photoFilter:      return makeSpec("colorPhotoFilter", .colorCorrection, "Photo Filter", textures: 1)
        case .colorBalance:     return makeSpec("colorBalance", .colorCorrection, "Color Balance", textures: 1)
        case .tint:             return makeSpec("colorTint", .colorCorrection, "Tint", textures: 1)
        // Distort
        case .displacementMap:  return makeSpec("displaceChromatic", .distort, "Displacement Map", textures: 2)
        case .waveWarp:         return makeSpec("distortWaveWarp", .distort, "Wave Warp", textures: 1)
        case .polarCoordinates: return makeSpec("distortPolar", .distort, "Polar Coordinates", textures: 1)
        case .cornerPin:        return makeSpec("distortCornerPin", .distort, "Corner Pin", textures: 1)
        case .meshWarp:         return makeSpec("distortMeshWarp", .distort, "Mesh Warp", textures: 1, buffers: 1)
        case .offset:           return makeSpec("distortOffset", .distort, "Offset", textures: 1)
        case .spherize:         return makeSpec("distortSpherize", .distort, "Spherize", textures: 1)
        // Generate (generators write a fresh image; composite it over the layer with the blend-mode pass)
        case .grid:             return makeSpec("generateGrid", .generate, "Grid", textures: 0)
        case .fourColorGradient:return makeSpec("generateFourColorGradient", .generate, "4-Color Gradient", textures: 0)
        case .fractalNoise:     return makeSpec("generateFractalNoise", .generate, "Fractal Noise", textures: 0)
        case .stroke:           return makeSpec("generateStroke", .generate, "Stroke", textures: 1)
        case .audioSpectrum:    return makeSpec("generateAudioSpectrum", .generate, "Audio Spectrum", textures: 0, buffers: 1)
        case .cellPattern:      return makeSpec("generateCellPattern", .generate, "Cell Pattern", textures: 0)
        case .checkerboard:     return makeSpec("generateCheckerboard", .generate, "Checkerboard", textures: 0)
        // Simulation. The three particle effects share one engine and differ only by emitter preset.
        case .ccParticleWorld:      return makeSpec("particleUpdate", .simulation, "CC Particle World", textures: 0, params: false, driver: .simulation)
        case .particlePlayground:   return makeSpec("particleUpdate", .simulation, "Particle Playground", textures: 0, params: false, driver: .simulation)
        case .ccParticleSystemsII:  return makeSpec("particleUpdate", .simulation, "CC Particle Systems II", textures: 0, params: false, driver: .simulation)
        case .waveWorld:        return makeSpec("waveStep", .simulation, "Wave World", textures: 1, driver: .simulation)
        case .shatter:          return makeSpec("simShatter", .simulation, "Shatter", textures: 1)
        case .foam:             return makeSpec("simFoam", .simulation, "Foam", textures: 1)
        // Stylize
        case .glow:             return makeSpec("glowThreshold", .stylize, "Glow", textures: 1, driver: .multiPass)
        case .saberEdgeGlow:    return makeSpec("stylizeEdgeGlowSource", .stylize, "Saber Edge Glow", textures: 1, driver: .multiPass)
        case .strobeLight:      return makeSpec("stylizeStrobe", .stylize, "Strobe Light", textures: 1)
        case .mosaic:           return makeSpec("stylizeMosaic", .stylize, "Mosaic", textures: 1)
        case .findEdges:        return makeSpec("stylizeFindEdges", .stylize, "Find Edges", textures: 1)
        case .threshold:        return makeSpec("stylizeThreshold", .stylize, "Threshold", textures: 1)
        case .cartoon:          return makeSpec("stylizeCartoon", .stylize, "Cartoon", textures: 1, driver: .multiPass)
        // Perspective
        case .camera3D:         return makeSpec("perspectiveCamera3D", .perspective, "3D Camera", textures: 2)   // layer + colour out; depth out is bound last
        case .bevelAlpha:       return makeSpec("perspectiveBevel", .perspective, "Bevel Alpha", textures: 2, driver: .multiPass)
        case .dropShadow:       return makeSpec("perspectiveDropShadow", .perspective, "Drop Shadow", textures: 2, driver: .multiPass)
        case .basic3D:          return makeSpec("perspectiveBasic3D", .perspective, "Basic 3D", textures: 1)
        case .bevelEdges:       return makeSpec("perspectiveBevel", .perspective, "Bevel Edges", textures: 2)
        // Keying (all output premultiplied RGBA)
        case .keylight:         return makeSpec("keyChroma", .keying, "Keylight", textures: 1)
        case .lumaKey:          return makeSpec("keyLuma", .keying, "Luma Key", textures: 1)
        case .colorRange:       return makeSpec("keyColorRange", .keying, "Color Range", textures: 1)
        case .differenceMatte:  return makeSpec("keyDifferenceMatte", .keying, "Difference Matte", textures: 2)
        case .spillSuppressor:  return makeSpec("keySpillSuppress", .keying, "Spill Suppressor", textures: 1)
        }
    }
}

// MARK: - Scratch texture pool

/// Recycles intermediate textures so multi-pass effects do not allocate GPU memory every frame.
/// Releasing a texture right after encoding is safe: later passes on the same queue are ordered after earlier ones.
final class TexturePool {
    private let device: MTLDevice
    private var free: [String: [MTLTexture]] = [:]
    private let lock = NSLock()

    init(device: MTLDevice) { self.device = device }

    func acquire(width: Int, height: Int, format: MTLPixelFormat) -> MTLTexture {
        let key = "\(width)x\(height)/\(format.rawValue)"
        lock.lock(); defer { lock.unlock() }
        if let t = free[key]?.popLast() { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite, .renderTarget]
        d.storageMode = .private
        guard let t = device.makeTexture(descriptor: d) else {
            fatalError("Could not allocate a \(width)x\(height) scratch texture (out of GPU memory?)")
        }
        return t
    }

    func acquire(like t: MTLTexture) -> MTLTexture { acquire(width: t.width, height: t.height, format: t.pixelFormat) }

    func release(_ t: MTLTexture) {
        lock.lock(); defer { lock.unlock() }
        free["\(t.width)x\(t.height)/\(t.pixelFormat.rawValue)", default: []].append(t)
    }
}

// MARK: - Runner

enum EffectError: LocalizedError {
    case noMetalLibrary
    case missingKernel(String)
    case encoderUnavailable

    var errorDescription: String? {
        switch self {
        case .noMetalLibrary: return "No default Metal library in the app bundle"
        case .missingKernel(let n): return "Missing Metal function \(n)"
        case .encoderUnavailable: return "Could not create a command encoder"
        }
    }
}

/// Binding convention for every compute effect:
///   textures: inputs in order, then the output; params struct at buffer(0); extra buffers from buffer(1).
final class EffectRunner {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let pool: TexturePool
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private var weightBuffers: [String: MTLBuffer] = [:]
    private let lock = NSLock()

    init(device: MTLDevice, library: MTLLibrary? = nil) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw EffectError.encoderUnavailable }
        guard let lib = library ?? device.makeDefaultLibrary() else { throw EffectError.noMetalLibrary }
        queue = q; self.library = lib
        pool = TexturePool(device: device)
    }

    func pipeline(_ name: String) throws -> MTLComputePipelineState {
        lock.lock(); defer { lock.unlock() }
        if let p = pipelines[name] { return p }
        guard let fn = library.makeFunction(name: name) else { throw EffectError.missingKernel(name) }
        let p = try device.makeComputePipelineState(function: fn)
        pipelines[name] = p
        return p
    }

    /// Encode one kernel. `params` is copied with setBytes (keep it under 4 KB).
    func encode<P>(kernel: String, in cb: MTLCommandBuffer, inputs: [MTLTexture], output: MTLTexture,
                   params: P, buffers: [MTLBuffer] = []) throws {
        var p = params
        try withUnsafeBytes(of: &p) { raw in
            try encodeRaw(kernel: kernel, in: cb, inputs: inputs, output: output, bytes: raw, buffers: buffers)
        }
    }

    /// Variant for kernels without a params struct (Curves).
    func encode(kernel: String, in cb: MTLCommandBuffer, inputs: [MTLTexture], output: MTLTexture,
                buffers: [MTLBuffer] = []) throws {
        try encodeRaw(kernel: kernel, in: cb, inputs: inputs, output: output, bytes: nil, buffers: buffers)
    }

    private func encodeRaw(kernel: String, in cb: MTLCommandBuffer, inputs: [MTLTexture], output: MTLTexture,
                           bytes: UnsafeRawBufferPointer?, buffers: [MTLBuffer]) throws {
        let pso = try pipeline(kernel)
        guard let enc = cb.makeComputeCommandEncoder() else { throw EffectError.encoderUnavailable }
        enc.setComputePipelineState(pso)
        for (i, t) in inputs.enumerated() { enc.setTexture(t, index: i) }
        enc.setTexture(output, index: inputs.count)
        if let bytes, let base = bytes.baseAddress { enc.setBytes(base, length: bytes.count, index: 0) }
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i + 1) }

        let w = pso.threadExecutionWidth
        let h = max(pso.maxTotalThreadsPerThreadgroup / w, 1)
        enc.dispatchThreads(MTLSize(width: output.width, height: output.height, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: w, height: h, depth: 1))
        enc.endEncoding()
    }

    /// Cached weight table for blurSeparable (center first, radius + 1 entries).
    func blurWeights(radius: Int, sigma: Float?, box: Bool) -> MTLBuffer {
        let key = "\(box ? "box" : "gauss")/\(radius)/\(sigma ?? -1)"
        lock.lock(); defer { lock.unlock() }
        if let b = weightBuffers[key] { return b }
        let w = box ? GaussianWeights.box(radius: radius) : GaussianWeights.make(radius: radius, sigma: sigma)
        let b = device.makeBuffer(bytes: w, length: w.count * MemoryLayout<Float>.stride, options: .storageModeShared)!
        weightBuffers[key] = b
        return b
    }
}
