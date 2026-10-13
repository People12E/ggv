import Foundation

// =============================================================================
// The 51 effects of the B.P library.
//
// Each entry names the passes (kernels in FX_*.metal), the aux input it needs, and every parameter in slot order.
// A parameter's slot is the sum of the widths of the parameters declared before it, so ORDER IS THE ABI:
// tools/check_catalog.py verifies this file against the `// @fx` annotation above every kernel and against the
// PARAM() reads inside it. Run it after touching either side.
//
// Unit conventions (the kernels do the conversion): .percent values are 0...100, .angle values are degrees,
// .pixels values are comp pixels at full resolution (the runner scales them for proxy renders).
// =============================================================================

private let dimensionsOptions = ["Horizontal and Vertical", "Horizontal", "Vertical"]
private let channelOptions = ["Red", "Green", "Blue", "Alpha", "Luminance"]
private let blendOptions = ["None (generator only)", "Normal", "Multiply", "Screen", "Overlay", "Add", "Difference"]
private let paintOptions = ["On Original Image", "On Transparent", "Reveal Original Image"]

enum EffectCatalog {

    static let all: [EffectDescriptor] = blurSharpen + colorCorrection + distort + generate + simulation + stylize + perspective + keying

    static let byID: [String: EffectDescriptor] = Dictionary(uniqueKeysWithValues: all.map { ($0.id, $0) })

    static func descriptor(_ id: String) -> EffectDescriptor? { byID[id] }

    static func effects(in category: EffectCategory) -> [EffectDescriptor] { all.filter { $0.category == category } }

    static let blurSharpen: [EffectDescriptor] = [lensBlur, radialBlur, smartBlur, directionalBlur, fastBoxBlur, gaussianBlur, unsharpMask]
    static let colorCorrection: [EffectDescriptor] = [curves, levels, hueSaturation, lumetriColor, photoFilter, colorBalance, tint]
    static let distort: [EffectDescriptor] = [displacementMap, waveWarp, polarCoordinates, cornerPin, meshWarp, offset, spherize]
    static let generate: [EffectDescriptor] = [grid, fourColorGradient, fractalNoise, stroke, audioSpectrum, cellPattern, checkerboard]
    static let simulation: [EffectDescriptor] = [ccParticleWorld, shatter, waveWorld, foam, particlePlayground, ccParticleSystemsII]
    static let stylize: [EffectDescriptor] = [glow, saberEdgeGlow, strobeLight, mosaic, findEdges, threshold, cartoon]
    static let perspective: [EffectDescriptor] = [camera3D, bevelAlpha, dropShadow, basic3D, bevelEdges]
    static let keying: [EffectDescriptor] = [keylight, lumaKey, colorRange, differenceMatte, spillSuppressor]

    // MARK: - Blur & Sharpen

    static let lensBlur: EffectDescriptor = FX("lensBlur", "Camera Lens Blur", .blurSharpen,
        passes: [.pass("fx_lens_blur")],
        aux: .depth,
        notes: "Iris-shaped bokeh. With Use Depth on, the aux depth map (from the 3D layer pass or a luma map) drives per-pixel blur.",
        params: [
            P("radius", 16, 0, 250, .pixels),
            P("blades", 6, 0, 16),
            P("rotation", 0, -360, 360, .angle),
            P("curvature", 0, 0, 1),
            P("depthSource", 0, 0, 1, .toggle),
            P("focalDistance", 0.5, 0, 1),
            P("invertDepth", 0, 0, 1, .toggle),
            P("highlightGain", 1, 0, 10),
            P("highlightThreshold", 0.8, 0, 1),
        ])

    static let radialBlur: EffectDescriptor = FX("radialBlur", "Radial Blur", .blurSharpen,
        passes: [.pass("fx_radial", aux: .none)],
        params: [
            P("type", 0, 0, 1, .choice, ["Spin", "Zoom"]),
            P("amount", 10, -360, 360),
            P2("center", 0.5, 0.5),
        ])

    static let smartBlur: EffectDescriptor = FX("smartBlur", "Smart Blur", .blurSharpen,
        passes: [.pass("fx_smart_blur", aux: .none)],
        params: [
            P("radius", 3, 0, 100, .pixels),
            P("threshold", 25, 0, 100, .percent),
            P("mode", 0, 0, 2, .choice, ["Normal", "Edge Only", "Overlay Edge"]),
        ])

    static let directionalBlur: EffectDescriptor = FX("directionalBlur", "Directional Blur", .blurSharpen,
        passes: [.pass("fx_directional", aux: .none)],
        params: [
            P("direction", 0, -360, 360, .angle),
            P("length", 10, 0, 1000, .pixels),
        ])

    static let fastBoxBlur: EffectDescriptor = FX("fastBoxBlur", "Fast Box Blur", .blurSharpen,
        passes: [.pass("fx_box", tag: 0, aux: .none, repeats: .count("iterations")),
                 .pass("fx_box", tag: 1, aux: .none, repeats: .count("iterations"))],
        runner: ["iterations"],
        params: [
            P("blurriness", 10, 0, 500, .pixels),
            P("dimensions", 0, 0, 2, .choice, dimensionsOptions),
            P("iterations", 3, 1, 16),
            P("repeatEdge", 0, 0, 1, .toggle),
        ])

    static let gaussianBlur: EffectDescriptor = FX("gaussianBlur", "Gaussian Blur", .blurSharpen,
        passes: [.pass("fx_gaussian", tag: 0, aux: .none), .pass("fx_gaussian", tag: 1, aux: .none)],
        params: [
            P("blurriness", 10, 0, 500, .pixels),
            P("dimensions", 0, 0, 2, .choice, dimensionsOptions),
            P("repeatEdge", 0, 0, 1, .toggle),
        ])

    static let unsharpMask: EffectDescriptor = FX("unsharpMask", "Unsharp Mask", .blurSharpen,
        passes: [.pass("fx_unsharp_blur", tag: 0, aux: .none),
                 .pass("fx_unsharp_blur", tag: 1, aux: .none),
                 .pass("fx_unsharp_apply", aux: .original)],
        params: [
            P("radius", 3, 0, 250, .pixels),
            P("amount", 100, 0, 500, .percent),
            P("threshold", 0, 0, 255),
        ])

    // MARK: - Color Correction

    static let curves: EffectDescriptor = FX("curves", "Curves", .colorCorrection,
        passes: [.pass("fx_curves")],
        aux: .lut1D,
        notes: "Aux is the 256x1 LUT from CurveLUT.makeTexture (master + R/G/B curves).",
        params: [
            P("amount", 100, 0, 100, .percent),
        ])

    static let levels: EffectDescriptor = FX("levels", "Levels", .colorCorrection,
        passes: [.pass("fx_levels", aux: .none)],
        params: [
            P("inputBlack", 0, 0, 1),
            P("inputWhite", 1, 0, 1),
            P("gamma", 1, 0.1, 10),
            P("outputBlack", 0, 0, 1),
            P("outputWhite", 1, 0, 1),
        ])

    static let hueSaturation: EffectDescriptor = FX("hueSaturation", "Hue/Saturation", .colorCorrection,
        passes: [.pass("fx_hue_sat", aux: .none)],
        params: [
            P("hue", 0, -180, 180, .angle),
            P("saturation", 0, -100, 100, .percent),
            P("lightness", 0, -100, 100, .percent),
            P("colorize", 0, 0, 1, .toggle),
            P("colorizeHue", 0, 0, 360, .angle),
            P("colorizeSaturation", 25, 0, 100, .percent),
            P("colorizeLightness", 0, -100, 100, .percent),
        ])

    static let lumetriColor: EffectDescriptor = FX("lumetriColor", "Lumetri Color", .colorCorrection,
        passes: [.pass("fx_lumetri", aux: .none),
                 .pass("fx_lut3d", skipIfNoAux: true)],
        aux: .lut3D,
        notes: "Optional creative look: pass a .cube texture (CubeLUT.load) as aux; without one the LUT pass is skipped. Wheel alpha is the strength.",
        params: [
            P("exposure", 0, -5, 5),
            P("contrast", 0, -100, 100, .percent),
            P("highlights", 0, -100, 100, .percent),
            P("shadows", 0, -100, 100, .percent),
            P("whites", 0, -100, 100, .percent),
            P("blacks", 0, -100, 100, .percent),
            P("temperature", 0, -100, 100, .percent),
            P("tint", 0, -100, 100, .percent),
            P("saturation", 100, 0, 200, .percent),
            P("vibrance", 0, -100, 100, .percent),
            P4("shadowWheel", 0.5, 0.5, 0.5, 0),
            P4("midWheel", 0.5, 0.5, 0.5, 0),
            P4("highlightWheel", 0.5, 0.5, 0.5, 0),
            P("lookIntensity", 100, 0, 100, .percent),
        ])

    static let photoFilter: EffectDescriptor = FX("photoFilter", "Photo Filter", .colorCorrection,
        passes: [.pass("fx_photo_filter", aux: .none)],
        params: [
            P4("filterColor", 0.92, 0.55, 0.15),
            P("density", 25, 0, 100, .percent),
            P("preserveLuminosity", 1, 0, 1, .toggle),
        ])

    static let colorBalance: EffectDescriptor = FX("colorBalance", "Color Balance", .colorCorrection,
        passes: [.pass("fx_color_balance", aux: .none)],
        params: [
            P("shadowR", 0, -100, 100),
            P("shadowG", 0, -100, 100),
            P("shadowB", 0, -100, 100),
            P("midR", 0, -100, 100),
            P("midG", 0, -100, 100),
            P("midB", 0, -100, 100),
            P("highR", 0, -100, 100),
            P("highG", 0, -100, 100),
            P("highB", 0, -100, 100),
            P("preserveLuminosity", 1, 0, 1, .toggle),
        ])

    static let tint: EffectDescriptor = FX("tint", "Tint", .colorCorrection,
        passes: [.pass("fx_tint", aux: .none)],
        params: [
            P4("mapBlackTo", 0, 0, 0),
            P4("mapWhiteTo", 1, 1, 1),
            P("amount", 100, 0, 100, .percent),
        ])

    // MARK: - Distort

    static let displacementMap: EffectDescriptor = FX("displacementMap", "Displacement Map", .distort,
        passes: [.pass("fx_displace")],
        aux: .layer,
        notes: "Aux is the map layer. 50% grey = no displacement.",
        params: [
            P("mapChannelX", 0, 0, 4, .choice, channelOptions),
            P("mapChannelY", 1, 0, 4, .choice, channelOptions),
            P("maxHorizontal", 5, -1000, 1000, .pixels),
            P("maxVertical", 5, -1000, 1000, .pixels),
            P("mapMode", 0, 0, 2, .choice, ["Stretch to Fit", "Center", "Tile"]),
            P("edgeBehavior", 0, 0, 1, .choice, ["Clamp", "Wrap"]),
        ])

    static let waveWarp: EffectDescriptor = FX("waveWarp", "Wave Warp", .distort,
        passes: [.pass("fx_wave_warp", aux: .none)],
        params: [
            P("waveType", 0, 0, 4, .choice, ["Sine", "Square", "Triangle", "Sawtooth", "Noise"]),
            P("waveHeight", 20, 0, 1000, .pixels),
            P("waveWidth", 100, 1, 2000, .pixels),
            P("direction", 90, -360, 360, .angle),
            P("speed", 1, -10, 10),
            P("phase", 0, -360, 360, .angle),
        ])

    static let polarCoordinates: EffectDescriptor = FX("polarCoordinates", "Polar Coordinates", .distort,
        passes: [.pass("fx_polar", aux: .none)],
        params: [
            P("conversion", 0, 0, 1, .choice, ["Rect to Polar", "Polar to Rect"]),
            P("interpolation", 100, 0, 100, .percent),
        ])

    static let cornerPin: EffectDescriptor = FX("cornerPin", "Corner Pin", .distort,
        passes: [.pass("fx_corner_pin", aux: .none)],
        notes: "Set the matrix with EffectValues.setCornerPin(corners:) (solves the homography on the CPU).",
        params: [
            PM3("homography"),
        ])

    static let meshWarp: EffectDescriptor = FX("meshWarp", "Mesh Warp", .distort,
        passes: [.pass("fx_mesh_warp", aux: .none)],
        extraBuffers: 1,
        notes: "Buffer 1 = columns*rows float2 offsets (normalized, row-major). Offsets are an approximation of AE's bezier mesh.",
        params: [
            P("columns", 4, 2, 32),
            P("rows", 4, 2, 32),
        ])

    static let offset: EffectDescriptor = FX("offset", "Offset", .distort,
        passes: [.pass("fx_offset", aux: .none)],
        params: [
            P("shiftX", 0, -10000, 10000, .pixels),
            P("shiftY", 0, -10000, 10000, .pixels),
            P("wrapAround", 1, 0, 1, .toggle),
            P("blendWithOriginal", 0, 0, 100, .percent),
        ])

    static let spherize: EffectDescriptor = FX("spherize", "Spherize", .distort,
        passes: [.pass("fx_spherize", aux: .none)],
        params: [
            P("radius", 200, 1, 5000, .pixels),
            P("amount", 100, -100, 100, .percent),
            P2("center", 0.5, 0.5),
        ])

    // MARK: - Generate (blendMode 0 = generator only, otherwise composited over the layer)

    static let grid: EffectDescriptor = FX("grid", "Grid", .generate,
        passes: [.pass("fx_grid", aux: .none)],
        params: [
            P2("anchor", 0, 0, .pixelPoint),
            P("cellWidth", 100, 2, 4000, .pixels),
            P("cellHeight", 100, 2, 4000, .pixels),
            P("border", 3, 0, 500, .pixels),
            P("feather", 0, 0, 500, .pixels),
            P4("color", 1, 1, 1),
            P("blendMode", 1, 0, 6, .choice, blendOptions),
        ])

    static let fourColorGradient: EffectDescriptor = FX("fourColorGradient", "4-Color Gradient", .generate,
        passes: [.pass("fx_gradient4", aux: .none)],
        params: [
            P2("point1", 0.15, 0.15),
            P2("point2", 0.85, 0.15),
            P2("point3", 0.15, 0.85),
            P2("point4", 0.85, 0.85),
            P4("color1", 1.0, 0.2, 0.3),
            P4("color2", 1.0, 0.8, 0.2),
            P4("color3", 0.2, 0.4, 1.0),
            P4("color4", 0.2, 0.9, 0.6),
            P("blend", 50, 0, 100, .percent),
            P("jitter", 0, 0, 100, .percent),
            P("opacity", 100, 0, 100, .percent),
            P("blendMode", 0, 0, 6, .choice, blendOptions),
        ])

    static let fractalNoise: EffectDescriptor = FX("fractalNoise", "Fractal Noise", .generate,
        passes: [.pass("fx_fractal_noise", aux: .none)],
        params: [
            P("noiseType", 0, 0, 3, .choice, ["Basic", "Turbulent", "Billow", "Ridged"]),
            P("contrast", 100, 0, 400, .percent),
            P("brightness", 0, -100, 100, .percent),
            P("scale", 100, 1, 5000, .pixels),
            P("complexity", 6, 1, 8),
            P("subInfluence", 70, 0, 100, .percent),
            P("subScale", 56, 10, 400, .percent),
            P("evolution", 0, -3600, 3600, .angle),
            P("evolutionSpeed", 30, -720, 720),
            P2("offset", 0, 0, .pixelPoint),
            P("invert", 0, 0, 1, .toggle),
            P("opacity", 100, 0, 100, .percent),
            P("blendMode", 0, 0, 6, .choice, blendOptions),
        ])

    static let stroke: EffectDescriptor = FX("stroke", "Stroke", .generate,
        passes: [.pass("fx_stroke", aux: .none)],
        extraBuffers: 1,
        notes: "Buffer 1 = polyline as float3 (x, y normalized, cumulative arc length 0...1), up to 256 points. Set pointCount to match.",
        params: [
            P("pointCount", 0, 0, 256),
            P("brushSize", 10, 0.5, 500, .pixels),
            P("brushHardness", 75, 0, 100, .percent),
            P("opacity", 100, 0, 100, .percent),
            P("start", 0, 0, 100, .percent),
            P("end", 100, 0, 100, .percent),
            P4("color", 1, 1, 1),
            P("paintStyle", 0, 0, 2, .choice, paintOptions),
        ])

    static let audioSpectrum: EffectDescriptor = FX("audioSpectrum", "Audio Spectrum", .generate,
        passes: [.pass("fx_audio_spectrum", aux: .none)],
        extraBuffers: 1,
        notes: "Buffer 1 = band magnitudes 0...1 (SpectrumAnalyzer.magnitudes + logBands). Start/End define the baseline (Radial: centre and radius).",
        params: [
            P("bandCount", 64, 1, 256),
            P("style", 0, 0, 2, .choice, ["Digital Bars", "Smooth Line", "Radial Bars"]),
            P2("start", 0.1, 0.7),
            P2("end", 0.9, 0.7),
            P("thickness", 4, 1, 100, .pixels),
            P("maxHeight", 200, 0, 4000, .pixels),
            P4("color", 1, 1, 1),
            P("paintStyle", 0, 0, 1, .choice, ["On Original Image", "On Transparent"]),
        ])

    static let cellPattern: EffectDescriptor = FX("cellPattern", "Cell Pattern", .generate,
        passes: [.pass("fx_cell_pattern", aux: .none)],
        params: [
            P("pattern", 0, 0, 3, .choice, ["Bubbles", "Crystals", "Plates", "Pillow"]),
            P("cellSize", 60, 4, 2000, .pixels),
            P("contrast", 100, 0, 400, .percent),
            P2("offset", 0, 0, .pixelPoint),
            P("evolution", 0, -3600, 3600, .angle),
            P("evolutionSpeed", 30, -720, 720),
            P("invert", 0, 0, 1, .toggle),
            P("opacity", 100, 0, 100, .percent),
            P("blendMode", 0, 0, 6, .choice, blendOptions),
        ])

    static let checkerboard: EffectDescriptor = FX("checkerboard", "Checkerboard", .generate,
        passes: [.pass("fx_checkerboard", aux: .none)],
        params: [
            P2("anchor", 0, 0, .pixelPoint),
            P("cellWidth", 100, 2, 4000, .pixels),
            P("cellHeight", 100, 2, 4000, .pixels),
            P("feather", 0, 0, 500, .pixels),
            P4("color", 1, 1, 1),
            P("blendMode", 1, 0, 6, .choice, blendOptions),
        ])

    // MARK: - Simulation (stateless: every frame is a pure function of comp time, so scrubbing is exact)

    static let ccParticleWorld: EffectDescriptor = FX("ccParticleWorld", "CC Particle World", .simulation,
        particles: ("fx_particle_vertex", "fx_particle_fragment"),
        runner: ["birthRate", "showSource"],
        notes: "Perspective emitter with depth. Drawn by ParticleRenderer; Birth Rate x Longevity = particle count.",
        params: particleParams(.world))

    static let shatter: EffectDescriptor = FX("shatter", "Shatter", .simulation,
        passes: [.pass("fx_shatter", aux: .none)],
        notes: "Keyframe Progress 0 -> 100. Pieces travel at most ~2 piece-sizes from home.",
        params: [
            P("pieceSize", 80, 4, 1000, .pixels),
            P("progress", 0, 0, 100, .percent),
            P("strength", 300, 0, 5000, .pixels),
            P("rotation", 90, 0, 720, .angle),
            P("gravity", 400, -5000, 5000, .pixels),
            P2("origin", 0.5, 0.5),
            P("seed", 1, 0, 9999, .seed),
            P("irregularity", 0.8, 0, 1),
        ])

    static let waveWorld: EffectDescriptor = FX("waveWorld", "Wave World", .simulation,
        passes: [.pass("fx_wave_world", aux: .none)],
        notes: "Analytic ripples / plane waves (not a fluid solve), so it scrubs exactly.",
        params: [
            P("waveType", 0, 0, 1, .choice, ["Ripples", "Plane Waves"]),
            P("amplitude", 12, 0, 500, .pixels),
            P("wavelength", 80, 2, 2000, .pixels),
            P("speed", 200, 0, 5000, .pixels),
            P("damping", 0.5, 0, 10),
            P("sourceCount", 3, 1, 8),
            P("seed", 1, 0, 9999, .seed),
            P2("origin", 0.5, 0.5),
            P("output", 1, 0, 2, .choice, ["Height Map", "Refracted Layer", "Normal Map"]),
            P("refraction", 30, 0, 500, .pixels),
            P("lightAngle", 315, 0, 360, .angle),
        ])

    static let foam: EffectDescriptor = FX("foam", "Foam", .simulation,
        passes: [.pass("fx_foam", aux: .none)],
        params: [
            P("bubbleSize", 40, 4, 500, .pixels),
            P("sizeVariation", 50, 0, 100, .percent),
            P("density", 40, 0, 100, .percent),
            P("riseSpeed", 60, -2000, 2000, .pixels),
            P("wobble", 0.3, 0, 3),
            P("refraction", 0.8, 0, 3),
            P("rimStrength", 100, 0, 300, .percent),
            P("specular", 100, 0, 300, .percent),
            P("seed", 1, 0, 9999, .seed),
            P("viewMode", 0, 0, 1, .choice, ["Rendered", "Bubbles Only"]),
        ])

    static let particlePlayground: EffectDescriptor = FX("particlePlayground", "Particle Playground", .simulation,
        particles: ("fx_particle_vertex", "fx_particle_fragment"),
        runner: ["birthRate", "showSource"],
        notes: "Preset: grid emitter that samples the layer colour (particles start on the image and burst out).",
        params: particleParams(.playground))

    static let ccParticleSystemsII: EffectDescriptor = FX("ccParticleSystemsII", "CC Particle Systems II", .simulation,
        particles: ("fx_particle_vertex", "fx_particle_fragment"),
        runner: ["birthRate", "showSource"],
        notes: "Preset: 2-D fountain with gravity and drag.",
        params: particleParams(.systemsII))

    // MARK: - Stylize

    static let glow: EffectDescriptor = FX("glow", "Glow", .stylize,
        passes: [.pass("fx_glow_extract", scale: 0.5, aux: .none),
                 .pass("fx_kawase", scale: 0.5, aux: .none, repeats: .kawase("radius")),
                 .pass("fx_glow_composite", aux: .original)],
        runner: ["radius"],
        params: [
            P("threshold", 0.6, 0, 4),
            P("softness", 0.1, 0, 2),
            P("radius", 40, 0, 500, .pixels),
            P("intensity", 1, 0, 10),
            P("colorMode", 0, 0, 1, .choice, ["Original Colors", "A & B Colors"]),
            P4("colorA", 1.0, 0.85, 0.4),
            P4("colorB", 1.0, 0.3, 0.1),
            P("operation", 0, 0, 1, .choice, ["Add", "Screen"]),
        ])

    static let saberEdgeGlow: EffectDescriptor = FX("saberEdgeGlow", "Saber Edge Glow", .stylize,
        passes: [.pass("fx_saber_core", scale: 0.5, aux: .none),
                 .pass("fx_kawase", scale: 0.5, aux: .none, repeats: .kawase("glowRadius")),
                 .pass("fx_saber_composite", aux: .original)],
        runner: ["glowRadius"],
        notes: "Energy-beam glow driven by the layer's bright shapes (draw the beam in a shape layer, then apply). Not the Video Copilot plug-in.",
        params: [
            P("threshold", 0.5, 0, 1),
            P("softness", 0.15, 0, 1),
            P("coreIntensity", 1.5, 0, 10),
            P("innerRadius", 6, 0, 200, .pixels),
            P("glowRadius", 60, 0, 500, .pixels),
            P("glowIntensity", 1, 0, 10),
            P4("coreColor", 1, 1, 1),
            P4("glowColor", 0.2, 0.5, 1.0),
            P("turbulence", 0, 0, 100, .pixels),
            P("flicker", 0, 0, 1),
        ])

    static let strobeLight: EffectDescriptor = FX("strobeLight", "Strobe Light", .stylize,
        passes: [.pass("fx_strobe", aux: .none)],
        params: [
            P4("strobeColor", 1, 1, 1),
            P("blendWithOriginal", 0, 0, 100, .percent),
            P("strobeDuration", 0.05, 0, 10, .seconds),
            P("strobePeriod", 1, 0.01, 60, .seconds),
            P("probability", 100, 0, 100, .percent),
            P("operator", 0, 0, 1, .choice, ["Operates On Color", "Makes Layer Transparent"]),
            P("seed", 1, 0, 9999, .seed),
        ])

    static let mosaic: EffectDescriptor = FX("mosaic", "Mosaic", .stylize,
        passes: [.pass("fx_mosaic", aux: .none)],
        params: [
            P("horizontalBlocks", 10, 1, 2000),
            P("verticalBlocks", 10, 1, 2000),
            P("sharpColors", 0, 0, 1, .toggle),
        ])

    static let findEdges: EffectDescriptor = FX("findEdges", "Find Edges", .stylize,
        passes: [.pass("fx_find_edges", aux: .none)],
        params: [
            P("invert", 0, 0, 1, .toggle),
            P("blendWithOriginal", 0, 0, 100, .percent),
        ])

    static let threshold: EffectDescriptor = FX("threshold", "Threshold", .stylize,
        passes: [.pass("fx_threshold", aux: .none)],
        params: [
            P("level", 0.5, 0, 1),
            P("softness", 0.01, 0, 0.5),
        ])

    static let cartoon: EffectDescriptor = FX("cartoon", "Cartoon", .stylize,
        passes: [.pass("fx_cartoon", aux: .none)],
        params: [
            P("shadingSteps", 6, 2, 32),
            P("shadingSmoothness", 50, 0, 100, .percent),
            P("detailRadius", 2, 0.5, 20, .pixels),
            P("detailThreshold", 20, 0, 100, .percent),
            P("edgeThreshold", 0.2, 0, 2),
            P("edgeWidth", 1.5, 0.5, 20, .pixels),
            P("edgeSoftness", 0.1, 0, 1),
            P("edgeOpacity", 100, 0, 100, .percent),
            P4("edgeColor", 0, 0, 0),
        ])

    // MARK: - Perspective

    /// The 3D Camera effect has two halves: Layer3DRenderer draws the 3D layers (colour + depth), then this
    /// descriptor runs the depth-of-field blur over that colour image with the depth image as aux.
    static let camera3D: EffectDescriptor = FX("camera3D", "3D Camera", .perspective,
        passes: [.pass("fx_lens_blur")],
        aux: .depth,
        runner: ["fieldOfView", "farPlane", "shadowMapSize", "shadowBias"],
        notes: "Depth of field = Camera Lens Blur with Use Depth on. Field of View, Far Plane and shadow settings are read by Layer3DRenderer.",
        params: [
            P("radius", 12, 0, 250, .pixels),
            P("blades", 6, 0, 16),
            P("rotation", 0, -360, 360, .angle),
            P("curvature", 0, 0, 1),
            P("depthSource", 1, 0, 1, .toggle),
            P("focalDistance", 0.5, 0, 1),
            P("invertDepth", 0, 0, 1, .toggle),
            P("highlightGain", 1, 0, 10),
            P("highlightThreshold", 0.8, 0, 1),
            P("fieldOfView", 39.6, 5, 150, .angle),
            P("farPlane", 20000, 100, 200000, .pixels),
            P("shadowMapSize", 1, 0, 2, .choice, ["1024", "2048", "4096"]),
            P("shadowBias", 0.002, 0, 0.05),
        ])

    static let bevelAlpha: EffectDescriptor = FX("bevelAlpha", "Bevel Alpha", .perspective,
        passes: [.pass("fx_bevel_alpha", aux: .none)],
        params: [
            P("edgeThickness", 5, 0.5, 200, .pixels),
            P("lightAngle", 315, 0, 360, .angle),
            P("lightIntensity", 50, 0, 200, .percent),
            P4("lightColor", 1, 1, 1),
        ])

    static let dropShadow: EffectDescriptor = FX("dropShadow", "Drop Shadow", .perspective,
        passes: [.pass("fx_drop_shadow", aux: .none)],
        params: [
            P4("shadowColor", 0, 0, 0),
            P("opacity", 50, 0, 100, .percent),
            P("direction", 135, 0, 360, .angle),
            P("distance", 10, 0, 1000, .pixels),
            P("softness", 10, 0, 500, .pixels),
            P("shadowOnly", 0, 0, 1, .toggle),
        ])

    static let basic3D: EffectDescriptor = FX("basic3D", "Basic 3D", .perspective,
        passes: [.pass("fx_basic3d", aux: .none)],
        params: [
            P("swivel", 0, -360, 360, .angle),
            P("tilt", 0, -360, 360, .angle),
            P("distance", 0, -2000, 5000, .pixels),
            P("fieldOfView", 60, 5, 150, .angle),
            P("specular", 0, 0, 100, .percent),
        ])

    static let bevelEdges: EffectDescriptor = FX("bevelEdges", "Bevel Edges", .perspective,
        passes: [.pass("fx_bevel_edges", aux: .none)],
        params: [
            P("edgeThickness", 5, 0.5, 200, .pixels),
            P("lightAngle", 315, 0, 360, .angle),
            P("lightIntensity", 50, 0, 200, .percent),
            P4("lightColor", 1, 1, 1),
        ])

    // MARK: - Keying

    static let keylight: EffectDescriptor = FX("keylight", "Keylight", .keying,
        passes: [.pass("fx_keylight", aux: .none)],
        notes: "Screen-difference keyer in the style of Keylight; not Foundry's algorithm.",
        params: [
            P4("screenColor", 0.1, 0.75, 0.2),
            P("screenGain", 100, 0, 400, .percent),
            P("screenBalance", 50, 0, 100, .percent),
            P("clipBlack", 0, 0, 100, .percent),
            P("clipWhite", 100, 0, 100, .percent),
            P("despill", 100, 0, 100, .percent),
            P("viewMode", 0, 0, 2, .choice, ["Final Result", "Screen Matte", "Despilled Source"]),
        ])

    static let lumaKey: EffectDescriptor = FX("lumaKey", "Luma Key", .keying,
        passes: [.pass("fx_luma_key", aux: .none)],
        params: [
            P("keyType", 0, 0, 3, .choice, ["Key Out Brighter", "Key Out Darker", "Key Out Similar", "Key Out Dissimilar"]),
            P("threshold", 50, 0, 100, .percent),
            P("tolerance", 10, 0, 100, .percent),
            P("edgeThin", 0, 0, 20, .pixels),
            P("edgeFeather", 0, 0, 50, .pixels),
        ])

    static let colorRange: EffectDescriptor = FX("colorRange", "Color Range", .keying,
        passes: [.pass("fx_color_range", aux: .none)],
        params: [
            P("colorSpace", 0, 0, 2, .choice, ["Lab", "YUV", "RGB"]),
            P4("minColor", 0, 0.4, 0),
            P4("maxColor", 0.4, 1, 0.4),
            P("fuzziness", 20, 0, 100, .percent),
            P("invert", 0, 0, 1, .toggle),
        ])

    static let differenceMatte: EffectDescriptor = FX("differenceMatte", "Difference Matte", .keying,
        passes: [.pass("fx_difference_matte")],
        aux: .plate,
        notes: "Aux is the clean plate (same framing, subject absent).",
        params: [
            P("differenceMode", 0, 0, 1, .choice, ["Colors", "Luminance"]),
            P("tolerance", 10, 0, 100, .percent),
            P("softness", 5, 0, 100, .percent),
            P("invert", 0, 0, 1, .toggle),
        ])

    static let spillSuppressor: EffectDescriptor = FX("spillSuppressor", "Spill Suppressor", .keying,
        passes: [.pass("fx_spill_suppress", aux: .none)],
        params: [
            P4("suppressColor", 0.1, 0.75, 0.2),
            P("amount", 100, 0, 100, .percent),
            P("mode", 0, 0, 1, .choice, ["Average", "Maximum"]),
            P("preserveLuminance", 1, 0, 1, .toggle),
        ])
}

// MARK: - Particle presets

/// All three particle effects share one engine (fx_particle_vertex) and differ only in these starting values.
struct ParticleDefaults {
    var emitterMode: Float = 0
    var birthRate: Float = 200
    var longevity: Float = 2
    var emitterPosition = SIMD2<Float>(0.5, 0.5)
    var emitterZ: Float = 0
    var emitterRadius: Float = 0
    var gridColumns: Float = 40
    var gridRows: Float = 24
    var velocity: Float = 200
    var velocityVariation: Float = 30
    var direction: Float = 0
    var spread: Float = 60
    var gravity: Float = 0
    var resistance: Float = 0
    var turbulence: Float = 0
    var depthSpread: Float = 0
    var birthSize: Float = 12
    var deathSize: Float = 4
    var sizeVariation: Float = 30
    var birthColor = SIMD4<Float>(1, 1, 1, 1)
    var deathColor = SIMD4<Float>(1, 1, 1, 0)
    var fadeOut: Float = 30
    var cameraDistance: Float = 1000
    var perspective: Float = 0
    var shape: Float = 1
    var additive: Float = 1
    var colorFromLayer: Float = 0
    var showSource: Float = 1
    var seed: Float = 1
    var preroll: Float = 1
}

extension ParticleDefaults {
    static let world = ParticleDefaults(
        birthRate: 300, longevity: 2.5, emitterPosition: SIMD2(0.5, 0.6),
        velocity: 250, velocityVariation: 40, direction: 0, spread: 70, gravity: 300, resistance: 0.3,
        turbulence: 10, depthSpread: 150, birthSize: 14, deathSize: 4,
        birthColor: SIMD4(1.0, 0.7, 0.3, 1), deathColor: SIMD4(1.0, 0.1, 0.0, 0),
        perspective: 1, shape: 1, additive: 1)

    static let playground = ParticleDefaults(
        emitterMode: 2, birthRate: 400, longevity: 2.4, gridColumns: 40, gridRows: 24,
        velocity: 60, velocityVariation: 100, direction: 0, spread: 360, resistance: 1,
        birthSize: 18, deathSize: 6, sizeVariation: 20,
        birthColor: SIMD4(1, 1, 1, 1), deathColor: SIMD4(1, 1, 1, 0),
        shape: 0, additive: 0, colorFromLayer: 1, showSource: 0)

    static let systemsII = ParticleDefaults(
        birthRate: 250, longevity: 2, emitterPosition: SIMD2(0.5, 0.8),
        velocity: 300, velocityVariation: 30, direction: 0, spread: 40, gravity: 400, resistance: 0.1,
        birthSize: 8, deathSize: 2,
        birthColor: SIMD4(1.0, 0.9, 0.5, 1), deathColor: SIMD4(1.0, 0.2, 0.0, 0),
        shape: 1, additive: 1)
}

private let emitterOptions = ["Point", "Disc", "Grid From Layer", "Line"]
private let particleShapeOptions = ["Disc", "Glow", "Square"]

/// 37 slots; keep in step with the `fx_particle_vertex` / `fx_particle_fragment` annotations.
private func particleParams(_ d: ParticleDefaults) -> [ParamDescriptor] {
    return [
        P("emitterMode", d.emitterMode, 0, 3, .choice, emitterOptions),
        P("birthRate", d.birthRate, 1, 5000),
        P("longevity", d.longevity, 0.05, 30, .seconds),
        P2("emitterPosition", d.emitterPosition.x, d.emitterPosition.y),
        P("emitterZ", d.emitterZ, -5000, 5000, .pixels),
        P("emitterRadius", d.emitterRadius, 0, 5000, .pixels),
        P("gridColumns", d.gridColumns, 1, 200),
        P("gridRows", d.gridRows, 1, 200),
        P("velocity", d.velocity, 0, 5000, .pixels),
        P("velocityVariation", d.velocityVariation, 0, 100, .percent),
        P("direction", d.direction, -360, 360, .angle),
        P("spread", d.spread, 0, 360, .angle),
        P("gravity", d.gravity, -5000, 5000, .pixels),
        P("resistance", d.resistance, 0, 10),
        P("turbulence", d.turbulence, 0, 500, .pixels),
        P("depthSpread", d.depthSpread, 0, 5000, .pixels),
        P("birthSize", d.birthSize, 0, 500, .pixels),
        P("deathSize", d.deathSize, 0, 500, .pixels),
        P("sizeVariation", d.sizeVariation, 0, 100, .percent),
        P4("birthColor", d.birthColor.x, d.birthColor.y, d.birthColor.z, d.birthColor.w),
        P4("deathColor", d.deathColor.x, d.deathColor.y, d.deathColor.z, d.deathColor.w),
        P("fadeOut", d.fadeOut, 0, 100, .percent),
        P("cameraDistance", d.cameraDistance, 1, 20000, .pixels),
        P("perspective", d.perspective, 0, 1, .toggle),
        P("shape", d.shape, 0, 2, .choice, particleShapeOptions),
        P("additive", d.additive, 0, 1, .toggle),
        P("colorFromLayer", d.colorFromLayer, 0, 1, .toggle),
        P("showSource", d.showSource, 0, 1, .toggle),
        P("seed", d.seed, 0, 9999, .seed),
        P("preroll", d.preroll, 0, 1, .toggle),
    ]
}
