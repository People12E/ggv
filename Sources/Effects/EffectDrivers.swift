import Foundation
import Metal
import simd

// =============================================================================
// Render-pipeline drivers for the effects that are not compute passes:
//   ParticleRenderer  - CC Particle World / Particle Playground / CC Particle Systems II (one instanced draw)
//   Layer3DRenderer   - real 3D layers: shadow map pass + colour/depth pass, feeding the 3D Camera effect's blur
// Both encode onto the caller's command buffer and never commit.
// =============================================================================

// MARK: - Particles

final class ParticleRenderer {
    private let pso: MTLRenderPipelineState
    static let maxParticles = 200_000

    init(device: MTLDevice, library: MTLLibrary? = nil, pixelFormat: MTLPixelFormat = .rgba16Float) throws {
        guard let lib = library ?? device.makeDefaultLibrary() else { throw EffectRunnerError.unsupportedDevice }
        guard let vf = lib.makeFunction(name: "fx_particle_vertex") else { throw EffectRunnerError.missingFunction("fx_particle_vertex") }
        guard let ff = lib.makeFunction(name: "fx_particle_fragment") else { throw EffectRunnerError.missingFunction("fx_particle_fragment") }
        let d = MTLRenderPipelineDescriptor()
        d.label = "Particles"
        d.vertexFunction = vf
        d.fragmentFunction = ff
        d.colorAttachments[0].pixelFormat = pixelFormat
        // The fragment shader returns premultiplied colour, and alpha * (1 - additive), so one blend state serves both
        // "over" (additive = 0) and additive light (additive = 1: alpha 0 leaves the destination alpha untouched).
        d.colorAttachments[0].isBlendingEnabled = true
        d.colorAttachments[0].rgbBlendOperation = .add
        d.colorAttachments[0].alphaBlendOperation = .add
        d.colorAttachments[0].sourceRGBBlendFactor = .one
        d.colorAttachments[0].sourceAlphaBlendFactor = .one
        d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        do { pso = try device.makeRenderPipelineState(descriptor: d) }
        catch { throw EffectRunnerError.pipeline("fx_particle", error) }
    }

    /// Draws the particles into `output` (needs `.renderTarget` usage). `source` is the layer the effect sits on:
    /// copied underneath when Show Source is on, and sampled for Colour From Layer.
    func encode(_ effect: EffectDescriptor,
                values: EffectValues,
                source: MTLTexture,
                output: MTLTexture,
                time: Double,
                renderScale: Float = 1,
                commandBuffer cb: MTLCommandBuffer) throws {
        guard effect.particleKernels != nil else { throw EffectRunnerError.badTexture("\(effect.id) is not a particle effect") }
        guard output.usage.contains(.renderTarget) else {
            throw EffectRunnerError.badTexture("ParticleRenderer output needs .renderTarget usage")
        }

        var keepSource = false
        if values["showSource"] > 0.5, source.width == output.width, source.height == output.height,
           source.pixelFormat == output.pixelFormat, let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: source, to: output)
            blit.endEncoding()
            keepSource = true
        }

        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = output
        rp.colorAttachments[0].loadAction = keepSource ? .load : .clear
        rp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        rp.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { throw EffectRunnerError.unsupportedDevice }
        enc.label = effect.name
        enc.setRenderPipelineState(pso)

        var p = values.packed(passIndex: 0, repeatIndex: 0, tag: 0, time: time, renderScale: renderScale)
        let pBytes = p.count * MemoryLayout<Float>.stride
        enc.setVertexBytes(&p, length: pBytes, index: 0)
        enc.setFragmentBytes(&p, length: pBytes, index: 0)
        var viewSize = SIMD2<Float>(Float(output.width), Float(output.height))
        enc.setVertexBytes(&viewSize, length: MemoryLayout<SIMD2<Float>>.size, index: 1)
        enc.setVertexTexture(source, index: 0)

        // Slot i is reborn every `longevity` seconds, so live particles = birth rate x longevity.
        var count = Int((values["birthRate"] * values["longevity"]).rounded())
        if values["emitterMode"] == 2 { count = max(count, Int(values["gridColumns"] * values["gridRows"])) }   // one per grid cell
        count = min(max(count, 1), ParticleRenderer.maxParticles)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4, instanceCount: count)
        enc.endEncoding()
    }
}

// MARK: - 3D layers

/// Mirrors `Layer3DUniforms` in FX_Perspective.metal field for field (272 bytes).
struct Layer3DUniforms {
    var mvp = matrix_identity_float4x4
    var model = matrix_identity_float4x4
    var lightViewProj = matrix_identity_float4x4
    var lightPosDir = SIMD4<Float>(0, 0, 1, 0)      // xyz position (point) or direction towards the light; w 0 dir / 1 point
    var lightColor = SIMD4<Float>(0, 0, 0, 0)       // rgb * intensity
    var ambient = SIMD4<Float>(1, 1, 1, 0)
    var misc = SIMD4<Float>(1, 0.002, 0, 0)         // opacity, shadow bias, receivesLight, receivesShadows
    var sizeFar = SIMD4<Float>(0, 0, 20000, 0)      // layer size px, far plane
}

struct Layer3DItem {
    var texture: MTLTexture                         // the layer's rendered image (premultiplied rgba16Float)
    var model: simd_float4x4                        // see Layer3DRenderer.modelMatrix
    var opacity: Float = 1
    var receivesLight = false
    var receivesShadows = false
    var castsShadows = false
}

struct Light3D {
    enum Kind { case directional, point }
    var kind: Kind = .directional
    /// Point light: world position. Directional light: the direction pointing TOWARDS the light.
    var position = SIMD3<Float>(0.3, 0.6, 1)
    var color = SIMD3<Float>(1, 1, 1)
    var intensity: Float = 1
    var ambient = SIMD3<Float>(repeating: 0.25)
}

extension CameraMath {
    /// Right-handed orthographic projection, camera looks down -Z, Metal 0...1 depth.
    static func orthographic(left l: Float, right r: Float, bottom b: Float, top t: Float, near n: Float, far f: Float) -> simd_float4x4 {
        simd_float4x4(columns: (SIMD4<Float>(2 / (r - l), 0, 0, 0),
                                SIMD4<Float>(0, 2 / (t - b), 0, 0),
                                SIMD4<Float>(0, 0, 1 / (n - f), 0),
                                SIMD4<Float>(-(r + l) / (r - l), -(t + b) / (t - b), n / (n - f), 1)))
    }
}

/// Draws a stack of textured quads in 3D with optional lighting and shadows.
///
/// Output: `color` (premultiplied rgba16Float) and `depth` (r16Float, view-space distance / far, 1 = nothing there).
/// Feed both to the 3D Camera effect for depth of field:
///     runner.encode(EffectCatalog.camera3D, values: v, input: color, aux: depth, output: out, time: t, commandBuffer: cb)
///
/// Limits: layers are depth-tested against each other but transparent texels still write depth (a nearly transparent
/// layer in front can mask depth of what is behind it); lighting is a single light with PCF shadows; two-sided diffuse.
final class Layer3DRenderer {
    private let device: MTLDevice
    private let shadowPSO: MTLRenderPipelineState
    private let mainPSO: MTLRenderPipelineState
    private let depthLE: MTLDepthStencilState
    private var shadowMap: MTLTexture?
    private var depthBuffer: MTLTexture?

    init(device: MTLDevice, library: MTLLibrary? = nil,
         colorFormat: MTLPixelFormat = .rgba16Float, depthFormat: MTLPixelFormat = .r16Float) throws {
        guard let lib = library ?? device.makeDefaultLibrary() else { throw EffectRunnerError.unsupportedDevice }
        func fn(_ n: String) throws -> MTLFunction {
            guard let f = lib.makeFunction(name: n) else { throw EffectRunnerError.missingFunction(n) }
            return f
        }
        self.device = device

        let sd = MTLRenderPipelineDescriptor()
        sd.label = "3D shadow"
        sd.vertexFunction = try fn("fx_shadow_vertex")
        sd.fragmentFunction = try fn("fx_shadow_fragment")
        sd.depthAttachmentPixelFormat = .depth32Float
        let md = MTLRenderPipelineDescriptor()
        md.label = "3D layers"
        md.vertexFunction = try fn("fx_layer3d_vertex")
        md.fragmentFunction = try fn("fx_layer3d_fragment")
        md.colorAttachments[0].pixelFormat = colorFormat
        md.colorAttachments[0].isBlendingEnabled = true
        md.colorAttachments[0].rgbBlendOperation = .add
        md.colorAttachments[0].alphaBlendOperation = .add
        md.colorAttachments[0].sourceRGBBlendFactor = .one
        md.colorAttachments[0].sourceAlphaBlendFactor = .one
        md.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        md.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        md.colorAttachments[1].pixelFormat = depthFormat            // linear depth / far; overwritten, not blended
        md.depthAttachmentPixelFormat = .depth32Float
        do {
            shadowPSO = try device.makeRenderPipelineState(descriptor: sd)
            mainPSO = try device.makeRenderPipelineState(descriptor: md)
        } catch { throw EffectRunnerError.pipeline("fx_layer3d", error) }

        let ds = MTLDepthStencilDescriptor()
        ds.depthCompareFunction = .lessEqual
        ds.isDepthWriteEnabled = true
        guard let state = device.makeDepthStencilState(descriptor: ds) else { throw EffectRunnerError.unsupportedDevice }
        depthLE = state
    }

    /// Layer-space -> world. Layer space is pixels, y down, centred on the layer; world is pixels, y up, comp centre
    /// at the origin. `anchor` is in layer pixels from the top-left corner (After Effects convention).
    static func modelMatrix(position: SIMD3<Float>, rotation: simd_quatf, scale: Float,
                            anchor: SIMD2<Float>, size: SIMD2<Float>) -> simd_float4x4 {
        let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, 1, 1))
        let s = simd_float4x4(diagonal: SIMD4<Float>(scale, scale, scale, 1))
        let toAnchor = CameraMath.translation(SIMD3<Float>(-(anchor.x - size.x / 2), -(anchor.y - size.y / 2), 0))
        return CameraMath.translation(position) * simd_float4x4(rotation) * s * flip * toAnchor
    }

    private func makeTarget(_ existing: inout MTLTexture?, format: MTLPixelFormat, width: Int, height: Int) -> MTLTexture? {
        if let t = existing, t.width == width, t.height == height { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
        d.usage = [.renderTarget, .shaderRead]
        d.storageMode = .private
        existing = device.makeTexture(descriptor: d)
        return existing
    }

    /// Light view-projection covering every layer (directional: orthographic fit; point: 100-degree perspective).
    private func lightViewProj(_ light: Light3D, items: [Layer3DItem]) -> simd_float4x4 {
        guard !items.isEmpty else { return matrix_identity_float4x4 }
        let centres = items.map { SIMD3<Float>($0.model.columns.3.x, $0.model.columns.3.y, $0.model.columns.3.z) }
        let c = centres.reduce(SIMD3<Float>(repeating: 0), +) / Float(items.count)
        var radius: Float = 1
        for (item, p) in zip(items, centres) {
            let sx = simd_length(SIMD3<Float>(item.model.columns.0.x, item.model.columns.0.y, item.model.columns.0.z))
            let half = 0.5 * hypot(Float(item.texture.width), Float(item.texture.height)) * sx
            radius = max(radius, simd_distance(p, c) + half)
        }
        func up(for dir: SIMD3<Float>) -> SIMD3<Float> { abs(simd_dot(simd_normalize(dir), SIMD3<Float>(0, 1, 0))) > 0.99 ? SIMD3<Float>(1, 0, 0) : SIMD3<Float>(0, 1, 0) }
        switch light.kind {
        case .directional:
            let dir = simd_normalize(light.position)
            let eye = c + dir * (radius * 2)
            let view = CameraMath.lookAt(eye: eye, target: c, up: up(for: dir))
            let proj = CameraMath.orthographic(left: -radius, right: radius, bottom: -radius, top: radius, near: 0.1, far: radius * 4)
            return proj * view
        case .point:
            let toC = c - light.position
            let view = CameraMath.lookAt(eye: light.position, target: c, up: up(for: toC))
            let far = simd_length(toC) + radius * 2
            return CameraMath.perspective(fovY: 1.75, aspect: 1, near: 1, far: max(far, 10)) * view
        }
    }

    /// - viewProj: from CameraMath (e.g. `comp2DCamera(...).viewProj`, or the Camera layer's own matrices).
    /// - color / depth: render targets (`.renderTarget` + `.shaderRead`), same size; `depth` is r16Float by default.
    /// - shadowMapSize: 1024 / 2048 / 4096.
    func render(items: [Layer3DItem],
                viewProj: simd_float4x4,
                light: Light3D?,
                color: MTLTexture,
                depth: MTLTexture,
                far: Float = 20000,
                shadowMapSize: Int = 2048,
                shadowBias: Float = 0.002,
                commandBuffer cb: MTLCommandBuffer) throws {
        guard color.width == depth.width, color.height == depth.height else {
            throw EffectRunnerError.badTexture("3D colour and depth targets must be the same size")
        }
        guard let shadow = makeTarget(&shadowMap, format: .depth32Float, width: shadowMapSize, height: shadowMapSize),
              let zbuf = makeTarget(&depthBuffer, format: .depth32Float, width: color.width, height: color.height) else {
            throw EffectRunnerError.unsupportedDevice
        }

        // Back to front by clip-space w of each layer's centre.
        let sorted = items.sorted { (viewProj * $0.model.columns.3).w > (viewProj * $1.model.columns.3).w }
        let lightVP = light.map { lightViewProj($0, items: sorted) } ?? matrix_identity_float4x4

        func baseUniforms(for item: Layer3DItem) -> Layer3DUniforms {
            var u = Layer3DUniforms()
            u.model = item.model
            u.mvp = viewProj * item.model
            u.lightViewProj = lightVP
            if let l = light {
                u.lightPosDir = l.kind == .point ? SIMD4<Float>(l.position, 1) : SIMD4<Float>(simd_normalize(l.position), 0)
                u.lightColor = SIMD4<Float>(l.color * l.intensity, 0)
                u.ambient = SIMD4<Float>(l.ambient, 0)
            }
            let lit: Float = (light != nil && item.receivesLight) ? 1 : 0
            u.misc = SIMD4<Float>(item.opacity, shadowBias, lit, (lit > 0 && item.receivesShadows) ? 1 : 0)
            u.sizeFar = SIMD4<Float>(Float(item.texture.width), Float(item.texture.height), far, 0)
            return u
        }

        // 1. Shadow pass (always runs so the depth texture is valid; only casters draw).
        let sp = MTLRenderPassDescriptor()
        sp.depthAttachment.texture = shadow
        sp.depthAttachment.loadAction = .clear
        sp.depthAttachment.clearDepth = 1
        sp.depthAttachment.storeAction = .store
        guard let senc = cb.makeRenderCommandEncoder(descriptor: sp) else { throw EffectRunnerError.unsupportedDevice }
        senc.label = "3D shadow map"
        if light != nil {
            senc.setRenderPipelineState(shadowPSO)
            senc.setDepthStencilState(depthLE)
            senc.setCullMode(.none)
            for item in sorted where item.castsShadows {
                var u = baseUniforms(for: item)
                senc.setVertexBytes(&u, length: MemoryLayout<Layer3DUniforms>.stride, index: 0)
                senc.setFragmentTexture(item.texture, index: 0)
                senc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
            }
        }
        senc.endEncoding()

        // 2. Colour + depth pass.
        let mp = MTLRenderPassDescriptor()
        mp.colorAttachments[0].texture = color
        mp.colorAttachments[0].loadAction = .clear
        mp.colorAttachments[0].clearColor = MTLClearColorMake(0, 0, 0, 0)
        mp.colorAttachments[0].storeAction = .store
        mp.colorAttachments[1].texture = depth
        mp.colorAttachments[1].loadAction = .clear
        mp.colorAttachments[1].clearColor = MTLClearColorMake(1, 0, 0, 0)       // 1 = far / nothing
        mp.colorAttachments[1].storeAction = .store
        mp.depthAttachment.texture = zbuf
        mp.depthAttachment.loadAction = .clear
        mp.depthAttachment.clearDepth = 1
        mp.depthAttachment.storeAction = .dontCare
        guard let menc = cb.makeRenderCommandEncoder(descriptor: mp) else { throw EffectRunnerError.unsupportedDevice }
        menc.label = "3D layers"
        menc.setRenderPipelineState(mainPSO)
        menc.setDepthStencilState(depthLE)
        menc.setCullMode(.none)
        menc.setFragmentTexture(shadow, index: 1)
        for item in sorted {
            var u = baseUniforms(for: item)
            menc.setVertexBytes(&u, length: MemoryLayout<Layer3DUniforms>.stride, index: 0)
            menc.setFragmentBytes(&u, length: MemoryLayout<Layer3DUniforms>.stride, index: 0)
            menc.setFragmentTexture(item.texture, index: 0)
            menc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        }
        menc.endEncoding()
    }
}
