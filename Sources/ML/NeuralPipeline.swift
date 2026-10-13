import CoreML
import CoreVideo
import Metal
import IOSurface
import Vision

/// Zero-copy bridge: Metal textures and Core ML inputs/outputs share the same IOSurface-backed
/// CVPixelBuffers, so frames never cross a CPU copy between the compositor and the Neural Engine.
final class SharedFramePool {
    let device: MTLDevice
    private var textureCache: CVMetalTextureCache?

    init(device: MTLDevice) {
        self.device = device
        CVMetalTextureCacheCreate(nil, nil, device, nil, &textureCache)
    }

    func makeBuffer(width: Int, height: Int, format: OSType = kCVPixelFormatType_32BGRA) -> CVPixelBuffer? {
        let attrs: [CFString: Any] = [
            kCVPixelBufferIOSurfacePropertiesKey: [:],          // IOSurface backing => shareable with ANE/GPU
            kCVPixelBufferMetalCompatibilityKey: true,
            kCVPixelBufferWidthKey: width,
            kCVPixelBufferHeightKey: height,
            kCVPixelBufferPixelFormatTypeKey: format
        ]
        var pb: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, format, attrs as CFDictionary, &pb)
        return pb
    }

    /// Metal view of the same memory (no copy).
    func texture(for buffer: CVPixelBuffer, pixelFormat: MTLPixelFormat = .bgra8Unorm) -> MTLTexture? {
        guard let cache = textureCache else { return nil }
        var cvTex: CVMetalTexture?
        let w = CVPixelBufferGetWidth(buffer), h = CVPixelBufferGetHeight(buffer)
        CVMetalTextureCacheCreateTextureFromImage(nil, cache, buffer, nil, pixelFormat, w, h, 0, &cvTex)
        return cvTex.flatMap { CVMetalTextureGetTexture($0) }
    }
}

enum NeuralModule: String {
    case rotobrush  = "BP_Rotobrush"      // semantic video matting (frame + prev matte -> alpha)
    case inpaint    = "BP_Inpaint"        // frame + mask -> filled frame
    case iris       = "BP_IrisFace"       // face restoration
    case chronos    = "BP_ChronosRIFE"    // two frames + t -> interpolated frame
    case proteus    = "BP_ProteusDetail"  // detail recovery / upscaling
}

/// Loads compiled .mlmodelc files bundled with the app (convert your own models with coremltools;
/// this project does not ship trained weights).
final class NeuralEngine {
    private var models: [NeuralModule: MLModel] = [:]
    private let config: MLModelConfiguration

    init() {
        config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine   // prefer ANE; use .all to allow GPU fallback
    }

    func model(_ module: NeuralModule) throws -> MLModel {
        if let m = models[module] { return m }
        guard let url = Bundle.main.url(forResource: module.rawValue, withExtension: "mlmodelc") else {
            throw NSError(domain: "BP.Neural", code: 404,
                          userInfo: [NSLocalizedDescriptionKey: "Missing model \(module.rawValue)"])
        }
        let m = try MLModel(contentsOf: url, configuration: config)
        models[module] = m
        return m
    }

    /// Generic single/multi-image inference. Feature names must match the converted model's I/O.
    func run(_ module: NeuralModule,
             inputs: [String: CVPixelBuffer],
             scalars: [String: Double] = [:],
             outputName: String) async throws -> CVPixelBuffer {
        let m = try model(module)
        var dict: [String: MLFeatureValue] = inputs.mapValues { MLFeatureValue(pixelBuffer: $0) }
        for (k, v) in scalars { dict[k] = MLFeatureValue(double: v) }
        let provider = try MLDictionaryFeatureProvider(dictionary: dict)

        // Ask Core ML to write straight into a pre-allocated IOSurface buffer where the model allows.
        let opts = MLPredictionOptions()
        if let out = inputs.values.first,
           let target = CVPixelBufferCreateLike(out) {
            opts.outputBackings = [outputName: target]
        }
        let result = try await m.prediction(from: provider, options: opts)
        guard let buf = result.featureValue(for: outputName)?.imageBufferValue else {
            throw NSError(domain: "BP.Neural", code: 500,
                          userInfo: [NSLocalizedDescriptionKey: "No image output '\(outputName)'"])
        }
        return buf
    }

    // Convenience wrappers --------------------------------------------------

    func rotobrush(frame: CVPixelBuffer, previousMatte: CVPixelBuffer) async throws -> CVPixelBuffer {
        try await run(.rotobrush, inputs: ["frame": frame, "prev_matte": previousMatte], outputName: "matte")
    }

    func inpaint(frame: CVPixelBuffer, mask: CVPixelBuffer) async throws -> CVPixelBuffer {
        try await run(.inpaint, inputs: ["frame": frame, "mask": mask], outputName: "filled")
    }

    /// t in (0,1): position between frameA and frameB. For 100x slow-mo call with t = k/100.
    func interpolate(frameA: CVPixelBuffer, frameB: CVPixelBuffer, t: Double) async throws -> CVPixelBuffer {
        try await run(.chronos, inputs: ["frame0": frameA, "frame1": frameB], scalars: ["t": t], outputName: "mid")
    }

    func restoreFace(frame: CVPixelBuffer) async throws -> CVPixelBuffer {
        try await run(.iris, inputs: ["image": frame], outputName: "restored")
    }

    func recoverDetail(frame: CVPixelBuffer) async throws -> CVPixelBuffer {
        try await run(.proteus, inputs: ["image": frame], outputName: "enhanced")
    }
}

private func CVPixelBufferCreateLike(_ src: CVPixelBuffer) -> CVPixelBuffer? {
    let attrs: [CFString: Any] = [
        kCVPixelBufferIOSurfacePropertiesKey: [:],
        kCVPixelBufferMetalCompatibilityKey: true
    ]
    var pb: CVPixelBuffer?
    CVPixelBufferCreate(nil, CVPixelBufferGetWidth(src), CVPixelBufferGetHeight(src),
                        CVPixelBufferGetPixelFormatType(src), attrs as CFDictionary, &pb)
    return pb
}

// MARK: - Example: Rotobrush temporal propagation feeding the Metal compositor

final class RotoPropagator {
    let pool: SharedFramePool
    let engine: NeuralEngine
    private var lastMatte: CVPixelBuffer?

    init(pool: SharedFramePool, engine: NeuralEngine) { self.pool = pool; self.engine = engine }

    /// Seed with a user-painted stroke mask on the first frame, then propagate forward.
    func seed(_ matte: CVPixelBuffer) { lastMatte = matte }

    func next(frame: CVPixelBuffer) async throws -> MTLTexture? {
        guard let prev = lastMatte else { return nil }
        let matte = try await engine.rotobrush(frame: frame, previousMatte: prev)
        lastMatte = matte
        return pool.texture(for: matte, pixelFormat: .r8Unorm)   // used as track matte in the compositor
    }
}
