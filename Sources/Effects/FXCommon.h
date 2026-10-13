#ifndef FX_COMMON_H
#define FX_COMMON_H

#include <metal_stdlib>
using namespace metal;

// =============================================================================
// B.P effect-library conventions
// -----------------------------------------------------------------------------
// Every generic effect kernel has the same signature (FX_ARGS):
//   texture(0) src  : input of this pass (previous pass output, or the layer)
//   texture(1) dst  : output of this pass
//   texture(2) aux  : secondary input (map layer, clean plate, LUT, or the ORIGINAL
//                     layer for composite passes). A 1x1 dummy when unused.
//   buffer(0)  P    : packed floats, see slot map below
//
// Pixel data is PREMULTIPLIED RGBA, float16 working format. Colour effects un-premultiply
// internally. Values are whatever colour space the compositor works in (display-referred
// gamma by default, like After Effects 8/16bpc; linear if the project is set to linear).
//
// P slot map (filled by EffectRunner):
//   P[0] pass index within the whole effect     P[1] comp time in seconds
//   P[2] repeat index inside a repeated pass    P[3] per-pass tag (e.g. 0 = H, 1 = V)
//   P[4] render scale (proxy ratio, 1 = full)   P[5...] the effect's own parameters
// Pixel-unit parameters are pre-multiplied by render scale on the CPU.
//
// Every kernel is annotated `// @fx <kernel>: name@slot ...` and tools/check_catalog.py
// verifies the annotation, the Swift catalog, and the PARAM() reads all agree.
// =============================================================================

#define FX_RESERVED 5
#define PARAM(i)  P[FX_RESERVED + (i)]
#define PARAM2(i) float2(P[FX_RESERVED + (i)], P[FX_RESERVED + (i) + 1])
#define PARAM3(i) float3(P[FX_RESERVED + (i)], P[FX_RESERVED + (i) + 1], P[FX_RESERVED + (i) + 2])
#define PARAM4(i) float4(P[FX_RESERVED + (i)], P[FX_RESERVED + (i) + 1], P[FX_RESERVED + (i) + 2], P[FX_RESERVED + (i) + 3])

#define FX_ARGS \
    texture2d<float, access::sample> src [[texture(0)]], \
    texture2d<float, access::write>  dst [[texture(1)]], \
    texture2d<float, access::sample> aux [[texture(2)]], \
    constant float*                  P   [[buffer(0)]],  \
    uint2 gid [[thread_position_in_grid]]

#define FX_PROLOGUE \
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return; \
    constexpr sampler s (coord::normalized, address::clamp_to_edge, filter::linear);  \
    constexpr sampler sz(coord::normalized, address::clamp_to_zero, filter::linear);  \
    constexpr sampler sr(coord::normalized, address::repeat,        filter::linear);  \
    constexpr sampler sn(coord::normalized, address::clamp_to_edge, filter::nearest); \
    const float2 size    = float2(dst.get_width(), dst.get_height()); \
    const float2 srcSize = float2(src.get_width(), src.get_height()); \
    const float2 uv      = (float2(gid) + 0.5) / size; \
    const uint   passIdx = uint(P[0]); \
    const float  time    = P[1]; \
    const uint   repIdx  = uint(P[2]); \
    const float  tag     = P[3]; \
    const float  rscale  = max(P[4], 1e-3);

constant float FX_PI  = 3.14159265358979f;
constant float FX_TAU = 6.28318530717959f;

// ---------------------------------------------------------------- small helpers
inline float luma709(float3 c) { return dot(c, float3(0.2126f, 0.7152f, 0.0722f)); }
inline float fx_fmod(float x, float y) { return x - y * floor(x / y); }
inline float3 fx_fmod(float3 x, float y) { return x - y * floor(x / y); }   // hsl2rgb needs the vector form
inline float2 fx_rot(float2 v, float a) { float c = cos(a), s = sin(a); return float2(c * v.x - s * v.y, s * v.x + c * v.y); }
// After Effects angle widgets: 0 deg = up, increasing clockwise. Returns a direction in texture space (y down).
inline float2 fx_ae_dir(float deg) { float a = deg * 0.01745329252f; return float2(sin(a), -cos(a)); }

// Compositing of a straight-colour generator layer over a premultiplied base.
// mode 0 = replace (generator only); 1 normal, 2 multiply, 3 screen, 4 overlay, 5 add, 6 difference.
inline float3 fx_blend_rgb(float3 b, float3 t, int m) {
    switch (m) {
        case 2: return b * t;
        case 3: return 1.0f - (1.0f - b) * (1.0f - t);
        case 4: return select(2.0f * b * t, 1.0f - 2.0f * (1.0f - b) * (1.0f - t), b > 0.5f);
        case 5: return b + t;
        case 6: return fabs(b - t);
        default: return t;
    }
}
inline float4 fx_composite(float4 basePre, float3 topRGB, float topA, int mode) {
    if (mode == 0) return float4(topRGB * topA, topA);
    float4 b = float4(basePre.a > 1e-5f ? basePre.rgb / basePre.a : float3(0.0f), basePre.a);
    float3 mixedRGB = fx_blend_rgb(b.rgb, topRGB, mode);
    float outA = topA + b.a * (1.0f - topA);
    float3 pre = topA * ((1.0f - b.a) * topRGB + b.a * mixedRGB) + (1.0f - topA) * b.a * b.rgb;
    return float4(pre, outA);
}

inline float4 unpremul(float4 c) { return float4(c.a > 1e-5f ? c.rgb / c.a : float3(0.0f), c.a); }
inline float4 premul(float4 c)   { return float4(c.rgb * c.a, c.a); }

// ---------------------------------------------------------------- hashing / noise
inline uint fx_pcg(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word  = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
inline float fx_hash11(uint n) { return float(fx_pcg(n)) * (1.0f / 4294967295.0f); }
inline float fx_hash21(int2 p) { return fx_hash11(uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ 0x9e3779b9u); }
inline float2 fx_hash22(int2 p) {
    uint h = uint(p.x) * 73856093u ^ uint(p.y) * 19349663u ^ 0x85ebca6bu;
    return float2(float(fx_pcg(h)), float(fx_pcg(h + 1u))) * (1.0f / 4294967295.0f);
}
inline float3 fx_grad3(int3 p) {
    uint h = fx_pcg(uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u ^ uint(p.z) * 0xcb1ab31fu);
    return float3(float(fx_pcg(h)), float(fx_pcg(h + 1u)), float(fx_pcg(h + 2u))) * (2.0f / 4294967295.0f) - 1.0f;
}
// Classic gradient (Perlin) noise, roughly in [-1, 1].
inline float fx_gnoise3(float3 x) {
    float3 i = floor(x);
    float3 f = x - i;
    float3 u = f * f * f * (f * (f * 6.0f - 15.0f) + 10.0f);
    int3 b = int3(i);
    float n000 = dot(fx_grad3(b + int3(0, 0, 0)), f - float3(0, 0, 0));
    float n100 = dot(fx_grad3(b + int3(1, 0, 0)), f - float3(1, 0, 0));
    float n010 = dot(fx_grad3(b + int3(0, 1, 0)), f - float3(0, 1, 0));
    float n110 = dot(fx_grad3(b + int3(1, 1, 0)), f - float3(1, 1, 0));
    float n001 = dot(fx_grad3(b + int3(0, 0, 1)), f - float3(0, 0, 1));
    float n101 = dot(fx_grad3(b + int3(1, 0, 1)), f - float3(1, 0, 1));
    float n011 = dot(fx_grad3(b + int3(0, 1, 1)), f - float3(0, 1, 1));
    float n111 = dot(fx_grad3(b + int3(1, 1, 1)), f - float3(1, 1, 1));
    return mix(mix(mix(n000, n100, u.x), mix(n010, n110, u.x), u.y),
               mix(mix(n001, n101, u.x), mix(n011, n111, u.x), u.y), u.z);
}

// ---------------------------------------------------------------- colour spaces
inline float3 srgb2lin(float3 c) { return select(c / 12.92f, pow((c + 0.055f) / 1.055f, 2.4f), c > 0.04045f); }
inline float3 lin2srgb(float3 c) { return select(c * 12.92f, 1.055f * pow(max(c, 0.0f), 1.0f / 2.4f) - 0.055f, c > 0.0031308f); }

inline float3 rgb2hsl(float3 c) {
    float mx = max(c.r, max(c.g, c.b)), mn = min(c.r, min(c.g, c.b));
    float l = 0.5f * (mx + mn), d = mx - mn, h = 0.0f, sat = 0.0f;
    if (d > 1e-6f) {
        sat = d / max(1.0f - fabs(2.0f * l - 1.0f), 1e-6f);
        if (mx == c.r)      h = fx_fmod((c.g - c.b) / d, 6.0f);
        else if (mx == c.g) h = (c.b - c.r) / d + 2.0f;
        else                h = (c.r - c.g) / d + 4.0f;
        h /= 6.0f;
    }
    return float3(h, sat, l);
}
inline float3 hsl2rgb(float3 hsl) {
    float h = fx_fmod(hsl.x, 1.0f);
    float c = (1.0f - fabs(2.0f * hsl.z - 1.0f)) * hsl.y;
    float3 k = fx_fmod(float3(0.0f, 4.0f, 2.0f) + h * 6.0f, 6.0f);
    float3 rgb = clamp(fabs(k - 3.0f) - 1.0f, 0.0f, 1.0f);
    return (hsl.z - 0.5f * c) + c * rgb;
}

// Gamma-encoded sRGB -> CIE Lab (D65). L in [0,100].
inline float3 rgb2lab(float3 c) {
    float3 l = srgb2lin(clamp(c, 0.0f, 1.0f));
    float3 xyz = float3(dot(l, float3(0.4124564f, 0.3575761f, 0.1804375f)),
                        dot(l, float3(0.2126729f, 0.7151522f, 0.0721750f)),
                        dot(l, float3(0.0193339f, 0.1191920f, 0.9503041f))) / float3(0.95047f, 1.0f, 1.08883f);
    float3 f = select(7.787f * xyz + 16.0f / 116.0f, pow(max(xyz, 1e-6f), 1.0f / 3.0f), xyz > 0.008856f);
    return float3(116.0f * f.y - 16.0f, 500.0f * (f.x - f.y), 200.0f * (f.y - f.z));
}

// ---------------------------------------------------------------- misc
// Interleaved gradient noise (Jimenez) for per-pixel sample rotation.
inline float fx_ign(float2 p) { return fract(52.9829189f * fract(dot(p, float2(0.06711056f, 0.00583715f)))); }

// Vogel-disc sample i of n in the unit disc, rotated by `rot`.
inline float2 fx_vogel(int i, int n, float rot) {
    float r = sqrt((float(i) + 0.5f) / float(n));
    float a = float(i) * 2.39996323f + rot;
    return float2(cos(a), sin(a)) * r;
}

#endif
