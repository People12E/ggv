import Foundation

// MARK: - Categories

enum EffectCategory: String, CaseIterable, Identifiable {
    case blurSharpen    = "Blur & Sharpen"
    case colorCorrection = "Color Correction"
    case distort        = "Distort"
    case generate       = "Generate"
    case simulation     = "Simulation"
    case stylize        = "Stylize"
    case perspective    = "Perspective"
    case keying         = "Keying"
    var id: String { rawValue }
}

// MARK: - Parameters

/// How a parameter is stored in the kernel's `P` buffer and shown by the inspector.
/// Every kind occupies `width` consecutive float slots, in declaration order, starting at slot 0
/// (the kernel sees them at P[5 + slot]; slots 0...4 of P are reserved, see FXCommon.h).
enum ParamKind: Equatable {
    case scalar, percent, pixels, angle, seconds, toggle, choice, seed
    case point        // normalised 0...1 (x, y), 2 slots
    case pixelPoint   // pixels (x, y) from the top-left, 2 slots, scaled by render scale
    case color        // straight RGBA 0...1, 4 slots
    case matrix3      // 3x3 row-major, 9 slots (corner pin homography)

    var width: Int {
        switch self {
        case .point, .pixelPoint: return 2
        case .color: return 4
        case .matrix3: return 9
        default: return 1
        }
    }
    /// Pixel-unit values are multiplied by the render scale so proxy renders look like full-res ones.
    var scalesWithResolution: Bool { self == .pixels || self == .pixelPoint }
}

struct ParamDescriptor: Identifiable {
    var id: String { name }
    let name: String
    let kind: ParamKind
    let defaults: [Float]
    let minValue: Float
    let maxValue: Float
    let options: [String]          // labels for .choice

    var width: Int { kind.width }

    /// "highlightThreshold" -> "Highlight Threshold"
    var title: String {
        var out = ""
        for ch in name {
            if ch.isUppercase && !out.isEmpty { out.append(" ") }
            out.append(out.isEmpty ? Character(ch.uppercased()) : ch)
        }
        return out
    }
}

// Catalog DSL. tools/check_catalog.py parses these call shapes, keep them one-per-parameter.
func P(_ name: String, _ def: Float, _ lo: Float, _ hi: Float,
       _ kind: ParamKind = .scalar, _ options: [String] = []) -> ParamDescriptor {
    ParamDescriptor(name: name, kind: kind, defaults: [def], minValue: lo, maxValue: hi, options: options)
}
func P2(_ name: String, _ x: Float, _ y: Float, _ kind: ParamKind = .point) -> ParamDescriptor {
    ParamDescriptor(name: name, kind: kind, defaults: [x, y], minValue: -10_000, maxValue: 10_000, options: [])
}
func P4(_ name: String, _ r: Float, _ g: Float, _ b: Float, _ a: Float = 1) -> ParamDescriptor {
    ParamDescriptor(name: name, kind: .color, defaults: [r, g, b, a], minValue: 0, maxValue: 1, options: [])
}
func PM3(_ name: String) -> ParamDescriptor {
    ParamDescriptor(name: name, kind: .matrix3, defaults: [1, 0, 0, 0, 1, 0, 0, 0, 1], minValue: -100, maxValue: 100, options: [])
}

// MARK: - Passes

/// What a pass reads from texture(2).
enum AuxSource {
    case none
    case userAux      // the optional secondary layer / map / plate / depth / LUT supplied by the caller
    case original     // the effect's own input (composite passes)
}

/// Repeat a pass N times. `repeatIndex` is passed to the kernel in P[2].
enum RepeatRule {
    case none
    case count(String)    // N = the named parameter, clamped to 1...16
    case kawase(String)   // N derived from a pixel radius parameter: ceil(sqrt(radius)), clamped to 1...12
}

struct PassDescriptor {
    var kernel: String
    var tag: Float = 0            // P[3], e.g. 0 = horizontal, 1 = vertical
    var scale: Float = 1          // render-target scale relative to the output (blur chains run at 0.5)
    var aux: AuxSource = .userAux
    var repeats: RepeatRule = .none
    var skipIfNoAux = false       // optional pass: dropped when the caller supplies no aux texture (e.g. Lumetri's look LUT)

    static func pass(_ kernel: String, tag: Float = 0, scale: Float = 1,
                     aux: AuxSource = .userAux, repeats: RepeatRule = .none,
                     skipIfNoAux: Bool = false) -> PassDescriptor {
        PassDescriptor(kernel: kernel, tag: tag, scale: scale, aux: aux, repeats: repeats, skipIfNoAux: skipIfNoAux)
    }
}

/// What kind of secondary input an effect expects in texture(2).
enum AuxKind {
    case none
    case layer      // a map / matte layer (Displacement Map)
    case depth      // linear depth / far, as written by the 3D layer pass (colour attachment 1)
    case plate      // clean plate (Difference Matte)
    case lut1D      // 256x1 RGBA16F, built by CurveLUT
    case lut3D      // 3D texture (.cube)
}

// MARK: - Effect

struct EffectDescriptor: Identifiable {
    let id: String
    let name: String
    let category: EffectCategory
    var passes: [PassDescriptor]
    var particleKernels: (vertex: String, fragment: String)?
    var params: [ParamDescriptor]
    var aux: AuxKind
    var extraBuffers: Int                 // number of MTLBuffers the caller binds at buffer(1)...
    var runnerParams: [String]            // parameters consumed by the runner, not read by the kernels
    var notes: String

    /// First float slot of each parameter, in declaration order.
    var slotOffsets: [String: Int] {
        var map: [String: Int] = [:]
        var next = 0
        for p in params { map[p.name] = next; next += p.width }
        return map
    }
    var slotCount: Int { params.reduce(0) { $0 + $1.width } }
}

func FX(_ id: String, _ name: String, _ category: EffectCategory,
        passes: [PassDescriptor] = [],
        particles: (String, String)? = nil,
        aux: AuxKind = .none,
        extraBuffers: Int = 0,
        runner: [String] = [],
        notes: String = "",
        params: [ParamDescriptor]) -> EffectDescriptor {
    EffectDescriptor(id: id, name: name, category: category, passes: passes,
                     particleKernels: particles.map { (vertex: $0.0, fragment: $0.1) },
                     params: params, aux: aux, extraBuffers: extraBuffers,
                     runnerParams: runner, notes: notes)
}
