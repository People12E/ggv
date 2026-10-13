#include <metal_stdlib>
#include "BPShaderTypes.h"
using namespace metal;

constexpr sampler linClamp(coord::normalized, address::clamp_to_edge, filter::linear);
constexpr sampler linMirror(coord::normalized, address::mirrored_repeat, filter::linear);
constexpr sampler linRepeat(coord::normalized, address::repeat, filter::linear);

#define PIXEL_GUARD(dst) if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return; \
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());

// ===========================================================================
// DISTORT
// (Displacement Map + chromatic aberration live in Effects.metal.)
// ===========================================================================


static inline float waveShape(float x, uint shape) {
    switch (shape) {
        case 1:  return 1.0 - 4.0 * abs(fract(x / 6.2831853 + 0.25) - 0.5);
        case 2:  return sin(x) >= 0 ? 1.0 : -1.0;
        default: return sin(x);
    }
}

kernel void distortWaveWarp(texture2d<float, access::sample> src [[texture(0)]],
                            texture2d<float, access::write>  dst [[texture(1)]],
                            constant WaveWarpParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 size = float2(dst.get_width(), dst.get_height());
    float2 px = uv * size;
    float2 dir = float2(cos(p.angle), sin(p.angle));
    float2 perp = float2(-dir.y, dir.x);
    float phase = dot(px, dir) / p.wavelength * 6.2831853 + p.time * p.speed;
    float2 offset = perp * p.amplitude * waveShape(phase, p.shape);
    dst.write(src.sample(linClamp, (px + offset) / size), gid);
}

// Polar Coordinates: rect -> polar (mode 0) or polar -> rect (mode 1), interpolated by `amount`.

kernel void distortPolar(texture2d<float, access::sample> src [[texture(0)]],
                         texture2d<float, access::write>  dst [[texture(1)]],
                         constant PolarParams& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 c = uv - 0.5;
    float2 mapped;
    if (p.mode == 0) {      // output pixel (polar) samples rect source at (angle, radius)
        float ang = atan2(c.y, c.x) / 6.2831853 + 0.5;
        float rad = length(c) * 2.0;
        mapped = float2(ang, rad);
    } else {                // output pixel (rect) samples polar source
        float ang = uv.x * 6.2831853 - 3.14159265;
        float rad = uv.y * 0.5;
        mapped = 0.5 + float2(cos(ang), sin(ang)) * rad;
    }
    dst.write(src.sample(linMirror, mix(uv, mapped, p.amount)), gid);
}

// Corner Pin: inverse bilinear. Corners in normalized output space: TL, TR, BL, BR.
// Uses a precomputed homography (row-major 3x3) mapping OUTPUT pixel -> SOURCE uv.

kernel void distortCornerPin(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write>  dst [[texture(1)]],
                             constant CornerPinParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float3 h = p.outToSrc * float3(uv, 1.0);
    float2 s = h.xy / h.z;
    bool inside = all(s >= 0.0) && all(s <= 1.0) && h.z > 0.0;
    dst.write(inside ? src.sample(linClamp, s) : float4(0), gid);
}

// Mesh Warp: bicubic-ish (bilinear) warp from a control grid of displaced points.
// grid is (cols+1) x (rows+1) float2 offsets in normalized uv.

kernel void distortMeshWarp(texture2d<float, access::sample> src [[texture(0)]],
                            texture2d<float, access::write>  dst [[texture(1)]],
                            constant MeshWarpParams& p [[buffer(0)]],
                            device const float2* offsets [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 g = uv * float2(p.cols, p.rows);
    int2 c0 = int2(min(floor(g), float2(p.cols - 1, p.rows - 1)));
    float2 f = g - float2(c0);
    int stride = p.cols + 1;
    float2 o00 = offsets[c0.y * stride + c0.x];
    float2 o10 = offsets[c0.y * stride + c0.x + 1];
    float2 o01 = offsets[(c0.y + 1) * stride + c0.x];
    float2 o11 = offsets[(c0.y + 1) * stride + c0.x + 1];
    float2 o = mix(mix(o00, o10, f.x), mix(o01, o11, f.x), f.y);
    dst.write(src.sample(linClamp, uv + o), gid);
}


kernel void distortOffset(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant OffsetParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 s = uv - p.shift;
    dst.write(p.wrap != 0 ? src.sample(linRepeat, s) : src.sample(linClamp, s), gid);
}


kernel void distortSpherize(texture2d<float, access::sample> src [[texture(0)]],
                            texture2d<float, access::write>  dst [[texture(1)]],
                            constant SpherizeParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float aspect = float(dst.get_width()) / float(dst.get_height());
    float2 d = (uv - p.center) * float2(aspect, 1.0);
    float r = length(d) / max(p.radius, 1e-4);
    float2 s = uv;
    if (r < 1.0) {
        // Sample from a smaller radius near the center => magnified (bulged) look; r=1 stays fixed.
        float srcR = mix(r, asin(r) * 0.63661977, p.amount);
        float2 nd = d / max(length(d), 1e-6) * srcR * p.radius;
        s = p.center + nd / float2(aspect, 1.0);
    }
    dst.write(src.sample(linClamp, s), gid);
}

// ===========================================================================
// GENERATE
// ===========================================================================

static inline float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
static inline float2 hash22(float2 p) {
    float3 p3 = fract(float3(p.xyx) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
static inline float hash21(float2 p) { return hash22(p).x; }

static inline float valueNoise(float2 p) {
    float2 i = floor(p), f = fract(p);
    float2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash21(i), hash21(i + float2(1, 0)), u.x),
               mix(hash21(i + float2(0, 1)), hash21(i + float2(1, 1)), u.x), u.y);
}

// Fractal Noise: fBm with evolution (time) as a z-like offset.

kernel void generateFractalNoise(texture2d<float, access::write> dst [[texture(0)]],
                                 constant FractalNoiseParams& p [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float aspect = float(dst.get_width()) / float(dst.get_height());
    float2 q = uv * float2(aspect, 1.0) * p.scale;
    float amp = 0.5, sum = 0, norm = 0;
    for (int o = 0; o < p.octaves; ++o) {
        float n = valueNoise(q + float2(p.evolution * 0.37 * (o + 1), p.evolution * 0.21 * (o + 1)));
        if (p.turbulent != 0) n = abs(n * 2.0 - 1.0);
        sum += n * amp; norm += amp;
        q *= p.lacunarity; amp *= p.gain;
    }
    float v = sum / max(norm, 1e-5);
    v = saturate((v - 0.5) * p.contrast + 0.5 + p.brightness);
    dst.write(float4(float3(v), 1.0), gid);
}


kernel void generateGrid(texture2d<float, access::write> dst [[texture(0)]],
                         constant GridParams& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 px = float2(gid) + 0.5;
    float2 f = abs(fmod(px, p.cell) - p.cell * 0.5);
    float2 d = p.cell * 0.5 - f;                       // distance to nearest grid line
    float line = min(d.x, d.y);
    float a = 1.0 - smoothstep(p.lineWidth * 0.5, p.lineWidth * 0.5 + p.feather + 1e-3, line);
    dst.write(float4(mix(p.background, p.color, a), 1.0), gid);
}


kernel void generateCheckerboard(texture2d<float, access::write> dst [[texture(0)]],
                                 constant CheckerParams& p [[buffer(0)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    int2 c = int2(floor(float2(gid) / p.size));
    dst.write(float4(((c.x + c.y) & 1) ? p.colorB : p.colorA, 1.0), gid);
}


kernel void generateFourColorGradient(texture2d<float, access::write> dst [[texture(0)]],
                                      constant FourColorParams& p [[buffer(0)]],
                                      uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 q = uv;
    float w0 = 1.0 / (pow(distance(q, p.p0), 2.0) + 1e-4);
    float w1 = 1.0 / (pow(distance(q, p.p1), 2.0) + 1e-4);
    float w2 = 1.0 / (pow(distance(q, p.p2), 2.0) + 1e-4);
    float w3 = 1.0 / (pow(distance(q, p.p3), 2.0) + 1e-4);
    float3 c = (p.c0 * w0 + p.c1 * w1 + p.c2 * w2 + p.c3 * w3) / (w0 + w1 + w2 + w3);
    dst.write(float4(c, 1.0), gid);
}

// Cell Pattern: Worley noise (cells / crystals / plates via `mode`).

kernel void generateCellPattern(texture2d<float, access::write> dst [[texture(0)]],
                                constant CellParams& p [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float aspect = float(dst.get_width()) / float(dst.get_height());
    float2 q = uv * float2(aspect, 1.0) * p.scale;
    float2 i = floor(q), f = fract(q);
    float d1 = 8.0, d2 = 8.0;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            float2 g = float2(x, y);
            float2 o = hash22(i + g);
            o = 0.5 + 0.5 * sin(p.evolution + 6.2831853 * o);
            float d = length(g + o - f);
            if (d < d1) { d2 = d1; d1 = d; } else if (d < d2) { d2 = d; }
        }
    float v = p.mode == 0 ? d1 : (p.mode == 1 ? d2 - d1 : 1.0 - d1);
    v = saturate((v - 0.5) * p.contrast + 0.5);
    dst.write(float4(float3(v), 1.0), gid);
}

// Stroke: outlines a mask (alpha) by distance-from-edge approximation (needs a jump-flood/distance texture
// for exactness; this variant samples a ring neighborhood).

kernel void generateStroke(texture2d<float, access::sample> mask [[texture(0)]],
                           texture2d<float, access::write>  dst  [[texture(1)]],
                           constant StrokeParams& p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 texel = 1.0 / float2(mask.get_width(), mask.get_height());
    float a = mask.sample(linClamp, uv).a;
    float maxNeighbor = 0.0;
    for (int i = 0; i < p.samples; ++i) {
        float ang = 6.2831853 * float(i) / p.samples;
        maxNeighbor = max(maxNeighbor, mask.sample(linClamp, uv + float2(cos(ang), sin(ang)) * p.width * texel).a);
    }
    float outline = saturate(maxNeighbor - a);
    dst.write(float4(p.color, outline * p.opacity), gid);
}

// Audio Spectrum: bars from an FFT magnitude buffer (host fills `bins`, normalized 0..1, log-spaced).

kernel void generateAudioSpectrum(texture2d<float, access::write> dst [[texture(0)]],
                                  constant SpectrumParams& p [[buffer(0)]],
                                  device const float* bins [[buffer(1)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 size = float2(dst.get_width(), dst.get_height());
    float2 a = p.start * size, b = p.end * size;
    float2 ab = b - a;
    float len = length(ab);
    float2 dir = ab / max(len, 1e-3), nrm = float2(-dir.y, dir.x);
    float2 rel = float2(gid) + 0.5 - a;
    float along = dot(rel, dir);
    float across = dot(rel, nrm);
    float4 outc = float4(0);
    if (along >= 0 && along <= len) {
        float bandW = len / p.bandCount;
        int band = clamp(int(along / bandW), 0, p.bandCount - 1);
        float inBand = fmod(along, bandW);
        float h = bins[band] * p.maxHeight;
        if (inBand > p.gap * 0.5 * bandW && inBand < bandW - p.gap * 0.5 * bandW && across >= 0 && across <= h)
            outc = float4(p.color, 1.0);
    }
    dst.write(outc, gid);
}
