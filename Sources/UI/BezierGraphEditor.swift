import SwiftUI

/// Cubic Bezier easing from (0,0) to (1,1) with two draggable handles, CSS cubic-bezier style.
struct BezierEasing: Equatable {
    var p1 = CGPoint(x: 0.42, y: 0)
    var p2 = CGPoint(x: 0.58, y: 1)

    private func coord(_ t: Double, _ a: Double, _ b: Double) -> Double {
        let u = 1 - t
        return 3 * u * u * t * a + 3 * u * t * t * b + t * t * t
    }

    /// Eased value for normalized time x in [0,1] (solves x(t)=x by Newton, bisection fallback).
    func value(at x: Double) -> Double {
        let x = min(max(x, 0), 1)
        var t = x
        for _ in 0..<6 {
            let err = coord(t, p1.x, p2.x) - x
            let u = 1 - t
            let d = 3 * u * u * p1.x + 6 * u * t * (p2.x - p1.x) + 3 * t * t * (1 - p2.x)
            if abs(err) < 1e-6 { return coord(t, p1.y, p2.y) }
            if abs(d) < 1e-6 { break }
            t -= err / d
        }
        var lo = 0.0, hi = 1.0
        t = x
        for _ in 0..<24 {
            let v = coord(t, p1.x, p2.x)
            if abs(v - x) < 1e-6 { break }
            if v < x { lo = t } else { hi = t }
            t = (lo + hi) / 2
        }
        return coord(t, p1.y, p2.y)
    }
}

struct BezierGraphEditor: View {
    @Binding var easing: BezierEasing
    @State private var active: Int?

    var body: some View {
        GeometryReader { geo in
            let side = min(geo.size.width, geo.size.height)
            let rect = CGRect(x: (geo.size.width - side) / 2, y: (geo.size.height - side) / 2, width: side, height: side)

            func toView(_ p: CGPoint) -> CGPoint {
                CGPoint(x: rect.minX + p.x * rect.width, y: rect.maxY - p.y * rect.height)
            }
            func toNorm(_ p: CGPoint) -> CGPoint {
                CGPoint(x: min(max((p.x - rect.minX) / rect.width, 0), 1),   // x clamped to [0,1]
                        y: min(max((rect.maxY - p.y) / rect.height, -0.5), 1.5)) // y may overshoot
            }

            ZStack {
                Canvas { ctx, _ in
                    ctx.stroke(Path(rect), with: .color(.gray.opacity(0.4)), lineWidth: 1)
                    let a = toView(.zero), b = toView(CGPoint(x: 1, y: 1))
                    let c1 = toView(easing.p1), c2 = toView(easing.p2)

                    var handles = Path()
                    handles.move(to: a); handles.addLine(to: c1)
                    handles.move(to: b); handles.addLine(to: c2)
                    ctx.stroke(handles, with: .color(.white.opacity(0.5)), lineWidth: 1)

                    var curve = Path()
                    curve.move(to: a)
                    curve.addCurve(to: b, control1: c1, control2: c2)
                    ctx.stroke(curve, with: .color(.cyan), lineWidth: 3)

                    for p in [c1, c2] {
                        ctx.fill(Path(ellipseIn: CGRect(x: p.x - 8, y: p.y - 8, width: 16, height: 16)), with: .color(.orange))
                    }
                }
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { v in
                            if active == nil {
                                let d1 = hypot(v.startLocation.x - toView(easing.p1).x, v.startLocation.y - toView(easing.p1).y)
                                let d2 = hypot(v.startLocation.x - toView(easing.p2).x, v.startLocation.y - toView(easing.p2).y)
                                active = d1 <= d2 ? 1 : 2
                            }
                            let n = toNorm(v.location)
                            if active == 1 { easing.p1 = n } else { easing.p2 = n }
                        }
                        .onEnded { _ in active = nil }
                )
            }
        }
        .aspectRatio(1, contentMode: .fit)
        .padding()
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 16))
    }
}
