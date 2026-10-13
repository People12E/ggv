import Metal
import simd

// Multi-pass effects, composed from the single-dispatch kernels. Every function only *encodes* work into the
// supplied command buffer; scratch textures come from the runner's pool.

extension GlowParams {
    static var standard: GlowParams {
        var p = GlowParams()
        p.threshold = 0.8; p.knee = 0.2; p.intensity = 1.0; p.tint = SIMD3<Float>(1, 1, 1)
        return p
    }
}

extension EffectRunner {

    // MARK: Blur & sharpen

    /// One horizontal + one vertical pass of blurSeparable.
    private func separable(_ cb: MTLCommandBuffer, from: MTLTexture, to: MTLTexture,
                           radius: Int, weights: MTLBuffer) throws {
        let tmp = pool.acquire(like: to); defer { pool.release(tmp) }
        var p = SeparableParams(direction: SIMD2<Float>(1, 0), radius: Int32(radius))
        try encode(kernel: "blurSeparable", in: cb, inputs: [from], output: tmp, params: p, buffers: [weights])
        p.direction = SIMD2<Float>(0, 1)
        try encode(kernel: "blurSeparable", in: cb, inputs: [tmp], output: to, params: p, buffers: [weights])
    }

    /// Gaussian blur. `radius` in pixels (1...128); sigma defaults to radius / 2.
    func gaussianBlur(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, radius: Int, sigma: Float? = nil) throws {
        let r = max(1, min(radius, 128))
        try separable(cb, from: src, to: dst, radius: r, weights: blurWeights(radius: r, sigma: sigma, box: false))
    }

    /// Fast Box Blur: `iterations` box passes (3 closely approximates a Gaussian).
    func fastBoxBlur(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, radius: Int, iterations: Int = 3) throws {
        let r = max(1, min(radius, 128))
        let n = max(1, iterations)
        let weights = blurWeights(radius: r, sigma: nil, box: true)
        var input = src
        var temps: [MTLTexture] = []
        defer { temps.forEach(pool.release) }
        for i in 0..<n {
            let out: MTLTexture
            if i == n - 1 { out = dst } else { out = pool.acquire(like: dst); temps.append(out) }
            try separable(cb, from: input, to: out, radius: r, weights: weights)
            input = out
        }
    }

    func unsharpMask(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, radius: Int, amount: Float, threshold: Float) throws {
        let blurred = pool.acquire(like: src); defer { pool.release(blurred) }
        try gaussianBlur(cb, src: src, dst: blurred, radius: radius)
        try encode(kernel: "unsharpMask", in: cb, inputs: [src, blurred], output: dst,
                   params: UnsharpParams(amount: amount, threshold: threshold))
    }

    // MARK: Stylize

    /// Kawase chain used by Glow and Saber: ping-pongs `a` <-> `b` and returns whichever holds the result.
    private func kawaseChain(_ cb: MTLCommandBuffer, a: MTLTexture, b: MTLTexture, iterations: Int) throws -> MTLTexture {
        var read = a, write = b
        for i in 0..<max(0, iterations) {
            try encode(kernel: "kawaseBlur", in: cb, inputs: [read], output: write, params: KawaseParams(iteration: Float(i)))
            swap(&read, &write)
        }
        return read
    }

    /// Glow: threshold, N Kawase passes, additive composite. At 4K, run this on a half-resolution copy for a wider glow.
    func glow(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, params: GlowParams, iterations: Int = 4) throws {
        let a = pool.acquire(like: src), b = pool.acquire(like: src)
        defer { pool.release(a); pool.release(b) }
        try encode(kernel: "glowThreshold", in: cb, inputs: [src], output: a, params: params)
        let blurred = try kawaseChain(cb, a: a, b: b, iterations: iterations)
        try encode(kernel: "glowComposite", in: cb, inputs: [src, blurred], output: dst, params: params)
    }

    /// Saber-style edge glow: Sobel edge source tinted by `saber.color`, blurred, then added over the layer.
    /// Only `glow.intensity` and `glow.tint` matter here (threshold/knee are used by plain Glow).
    func saberEdgeGlow(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture,
                       saber: SaberParams, glow: GlowParams, iterations: Int = 5) throws {
        let a = pool.acquire(like: src), b = pool.acquire(like: src)
        defer { pool.release(a); pool.release(b) }
        try encode(kernel: "stylizeEdgeGlowSource", in: cb, inputs: [src], output: a, params: saber)
        let blurred = try kawaseChain(cb, a: a, b: b, iterations: iterations)
        try encode(kernel: "glowComposite", in: cb, inputs: [src, blurred], output: dst, params: glow)
    }

    /// Cartoon: edge-preserving smoothing first (flat colour regions), then posterize + ink lines.
    func cartoon(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, params: CartoonParams, smoothing: Int = 3) throws {
        let smooth = pool.acquire(like: src); defer { pool.release(smooth) }
        var sp = SmartParams()
        sp.radius = Int32(max(1, min(smoothing, 8)))
        sp.spatialSigma = max(1, Float(smoothing) / 2)
        sp.threshold = 0.15
        try encode(kernel: "blurSmart", in: cb, inputs: [src], output: smooth, params: sp)
        try encode(kernel: "stylizeCartoon", in: cb, inputs: [smooth], output: dst, params: params)
    }

    // MARK: Perspective

    /// Drop shadow. `offsetPixels` is converted to uv here; the shadow is cast by the layer's alpha.
    func dropShadow(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, blurRadius: Int,
                    offsetPixels: SIMD2<Float>, color: SIMD3<Float>, opacity: Float, shadowOnly: Bool = false) throws {
        let blurred = pool.acquire(like: src); defer { pool.release(blurred) }
        if blurRadius > 0 { try gaussianBlur(cb, src: src, dst: blurred, radius: blurRadius) }
        var p = DropShadowParams()
        p.offset = offsetPixels / SIMD2<Float>(Float(src.width), Float(src.height))
        p.color = color
        p.opacity = opacity
        p.shadowOnly = shadowOnly ? 1 : 0
        try encode(kernel: "perspectiveDropShadow", in: cb, inputs: [src, blurRadius > 0 ? blurred : src], output: dst, params: p)
    }

    /// Bevel Alpha: lighting from the (softened) alpha channel. `softness` is the blur radius in pixels.
    func bevelAlpha(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, params: BevelParams, softness: Int = 4) throws {
        var p = params
        p.mode = 0
        let height = pool.acquire(like: src); defer { pool.release(height) }
        if softness > 0 { try gaussianBlur(cb, src: src, dst: height, radius: softness) }
        try encode(kernel: "perspectiveBevel", in: cb, inputs: [src, softness > 0 ? height : src], output: dst, params: p)
    }

    /// Bevel Edges: lighting from the distance to the layer rectangle (`params.edgeWidth` pixels wide).
    func bevelEdges(_ cb: MTLCommandBuffer, src: MTLTexture, dst: MTLTexture, params: BevelParams) throws {
        var p = params
        p.mode = 2
        try encode(kernel: "perspectiveBevel", in: cb, inputs: [src, src], output: dst, params: p)
    }

    /// 3D Camera pass for one layer. Writes premultiplied colour and normalized depth (0 near ... 1 far/no hit, r32Float).
    func camera3D(_ cb: MTLCommandBuffer, layer: MTLTexture, color: MTLTexture, depth: MTLTexture, params: Camera3DParams) throws {
        // Two outputs: colour is bound at index 1 (as an "input" slot) and depth last, matching the kernel.
        try encode(kernel: "perspectiveCamera3D", in: cb, inputs: [layer, color], output: depth, params: params)
    }
}
