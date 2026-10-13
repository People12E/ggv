#include <metal_stdlib>
#include "BPShaderTypes.h"
using namespace metal;

constexpr sampler linClamp(coord::normalized, address::clamp_to_edge, filter::linear);

static inline float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

static inline float3 rgb2hsv(float3 c) {
    float4 K = float4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    float4 p = mix(float4(c.bg, K.wz), float4(c.gb, K.xy), step(c.b, c.g));
    float4 q = mix(float4(p.xyw, c.r), float4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return float3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

static inline float3 hsv2rgb(float3 c) {
    float3 p = abs(fract(c.xxx + float3(1.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(float3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}


// The compositor stores premultiplied RGBA. Colour grading must happen on straight colour.
static inline float4 unpremul(float4 c) { return c.a > 1e-5 ? float4(c.rgb / c.a, c.a) : float4(0.0); }
static inline float4 premul(float4 c)   { return float4(c.rgb * c.a, c.a); }

#define PIXEL_GUARD(dst) if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return; \
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());

// ===========================================================================
// COLOR CORRECTION
// ===========================================================================

// Curves: 256x1 LUT texture per channel (RGB master baked on CPU from Bezier/spline points).
kernel void colorCurves(texture2d<float, access::sample> src [[texture(0)]],
                        texture2d<float, access::sample> lut [[texture(1)]],  // 256x1, rgb = per-channel curve
                        texture2d<float, access::write>  dst [[texture(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    constexpr sampler ls(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 c = unpremul(src.sample(linClamp, uv));
    float n = float(lut.get_width());
    float3 u = (saturate(c.rgb) * (n - 1.0) + 0.5) / n;          // sample at texel centres
    float3 o = float3(lut.sample(ls, float2(u.r, 0.5)).r,
                      lut.sample(ls, float2(u.g, 0.5)).g,
                      lut.sample(ls, float2(u.b, 0.5)).b);
    dst.write(premul(float4(o, c.a)), gid);
}


kernel void colorLevels(texture2d<float, access::sample> src [[texture(0)]],
                        texture2d<float, access::write>  dst [[texture(1)]],
                        constant LevelsParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = unpremul(src.sample(linClamp, uv));
    float3 x = saturate((c.rgb - p.inBlack) / max(p.inWhite - p.inBlack, 1e-5));
    x = pow(x, float3(1.0 / max(p.gamma, 1e-3)));
    x = mix(float3(p.outBlack), float3(p.outWhite), x);
    dst.write(premul(float4(x, c.a)), gid);
}


kernel void colorHueSaturation(texture2d<float, access::sample> src [[texture(0)]],
                               texture2d<float, access::write>  dst [[texture(1)]],
                               constant HueSatParams& p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = unpremul(src.sample(linClamp, uv));
    float3 hsv = rgb2hsv(c.rgb);
    hsv.x = fract(hsv.x + p.hueShift);
    hsv.y = saturate(hsv.y * p.saturation);
    float3 rgb = hsv2rgb(hsv);
    rgb = p.lightness >= 0 ? mix(rgb, float3(1), p.lightness) : mix(rgb, float3(0), -p.lightness);
    dst.write(premul(float4(rgb, c.a)), gid);
}

// Lumetri-style grade: exposure, contrast, highlights/shadows, whites/blacks, temperature/tint,
// vibrance, saturation, then an optional 3D creative LUT (e.g. 33^3).

kernel void colorLumetri(texture2d<float, access::sample> src [[texture(0)]],
                         texture3d<float, access::sample> lut [[texture(1)]],
                         texture2d<float, access::write>  dst [[texture(2)]],
                         constant LumetriParams& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    constexpr sampler ls(coord::normalized, address::clamp_to_edge, filter::linear);
    float4 c = unpremul(src.sample(linClamp, uv));
    float3 x = c.rgb * exp2(p.exposure);

    // white balance (warm/cool on R/B, green/magenta on G)
    x *= float3(1.0 + p.temperature * 0.1, 1.0 - p.tint * 0.1, 1.0 - p.temperature * 0.1);

    float l = luma(x);
    float shadowMask = 1.0 - smoothstep(0.0, 0.5, l);
    float highMask   = smoothstep(0.5, 1.0, l);
    x += x * (p.shadows * shadowMask + p.highlights * highMask) * 0.5;
    x += p.blacks * 0.1 * (1.0 - smoothstep(0.0, 0.25, l));
    x += p.whites * 0.1 * smoothstep(0.75, 1.0, l);

    x = (x - 0.5) * (1.0 + p.contrast) + 0.5;

    float lum = luma(x);
    float sat = max(x.r, max(x.g, x.b)) - min(x.r, min(x.g, x.b));
    x = mix(float3(lum), x, 1.0 + p.vibrance * (1.0 - sat));   // vibrance boosts low-sat pixels more
    x = mix(float3(lum), x, p.saturation);
    x = saturate(x);

    if (p.hasLUT != 0) {
        float3 ln = float3(lut.get_width(), lut.get_height(), lut.get_depth());
        float3 lc = (saturate(x) * (ln - 1.0) + 0.5) / ln;       // sample at texel centres
        x = mix(x, lut.sample(ls, lc).rgb, p.lutIntensity);
    }
    dst.write(premul(float4(x, c.a)), gid);
}


kernel void colorPhotoFilter(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write>  dst [[texture(1)]],
                             constant PhotoFilterParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = unpremul(src.sample(linClamp, uv));
    float3 o = mix(c.rgb, c.rgb * p.filterColor * 2.0 , p.density);   // multiply-style filter
    if (p.preserveLuminosity != 0) o *= luma(c.rgb) / max(luma(o), 1e-4);
    dst.write(premul(float4(saturate(o), c.a)), gid);
}


kernel void colorBalance(texture2d<float, access::sample> src [[texture(0)]],
                         texture2d<float, access::write>  dst [[texture(1)]],
                         constant ColorBalanceParams& p [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = unpremul(src.sample(linClamp, uv));
    float l = luma(c.rgb);
    float ws = saturate((0.333 - l) / 0.333 + 0.5) * (1.0 - smoothstep(0.0, 0.5, l));
    float wh = smoothstep(0.5, 1.0, l);
    float wm = saturate(1.0 - ws - wh);
    float3 o = c.rgb + p.shadows * ws + p.midtones * wm + p.highlights * wh;
    if (p.preserveLuminosity != 0) o *= l / max(luma(o), 1e-4);
    dst.write(premul(float4(saturate(o), c.a)), gid);
}


kernel void colorTint(texture2d<float, access::sample> src [[texture(0)]],
                      texture2d<float, access::write>  dst [[texture(1)]],
                      constant TintParams& p [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = unpremul(src.sample(linClamp, uv));
    float3 t = mix(p.mapBlack, p.mapWhite, luma(c.rgb));
    dst.write(premul(float4(mix(c.rgb, t, p.amount), c.a)), gid);
}

// ===========================================================================
// STYLIZE
// ===========================================================================

// Saber-style edge glow: Sobel edges (or alpha edges) feed a glow source, tinted and boosted.

kernel void stylizeEdgeGlowSource(texture2d<float, access::sample> src [[texture(0)]],
                                  texture2d<float, access::write>  dst [[texture(1)]],
                                  constant SaberParams& p [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 t = 1.0 / float2(src.get_width(), src.get_height());
    float tl = luma(src.sample(linClamp, uv + t * float2(-1, -1)).rgb);
    float tc = luma(src.sample(linClamp, uv + t * float2( 0, -1)).rgb);
    float tr = luma(src.sample(linClamp, uv + t * float2( 1, -1)).rgb);
    float ml = luma(src.sample(linClamp, uv + t * float2(-1,  0)).rgb);
    float mr = luma(src.sample(linClamp, uv + t * float2( 1,  0)).rgb);
    float bl = luma(src.sample(linClamp, uv + t * float2(-1,  1)).rgb);
    float bc = luma(src.sample(linClamp, uv + t * float2( 0,  1)).rgb);
    float br = luma(src.sample(linClamp, uv + t * float2( 1,  1)).rgb);
    float gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
    float gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
    float e = smoothstep(p.threshold, p.threshold + 0.2, length(float2(gx, gy)) * p.edgeStrength);
    dst.write(float4(p.color * e * p.intensity, e), gid);
}
// Then: blur the source with kawaseBlur (Effects.metal) several times and composite with glowComposite.

// Find Edges

kernel void stylizeFindEdges(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write>  dst [[texture(1)]],
                             constant FindEdgesParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 t = 1.0 / float2(src.get_width(), src.get_height());
    float3 gx = float3(0), gy = float3(0);
    const float kx[9] = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    const float ky[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            float3 s = src.sample(linClamp, uv + float2(i, j) * t).rgb;
            int k = (j + 1) * 3 + (i + 1);
            gx += s * kx[k]; gy += s * ky[k];
        }
    float3 e = saturate(sqrt(gx * gx + gy * gy));
    e = p.invert != 0 ? 1.0 - e : e;
    float4 c = src.sample(linClamp, uv);
    dst.write(float4(mix(c.rgb, e, p.blend), c.a), gid);
}


kernel void stylizeThreshold(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write>  dst [[texture(1)]],
                             constant ThresholdParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    dst.write(float4(float3(step(p.level, luma(c.rgb))), c.a), gid);
}


kernel void stylizeMosaic(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant MosaicParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 b = float2(max(p.blocks.x, 1), max(p.blocks.y, 1));
    float2 cell = floor(uv * b);
    float2 center = (cell + 0.5) / b;
    float4 c = src.sample(linClamp, center);   // sharpColors=0: bilinear sample at center averages 2x2
    dst.write(c, gid);
}

// Strobe light: every `period` frames, flash toward a color (or invert) for `duration` frames.

kernel void stylizeStrobe(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant StrobeParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    bool on = fmod(p.time, p.period) < p.duration;
    float3 flash = p.invert != 0 ? 1.0 - c.rgb : p.color;
    dst.write(float4(on ? mix(c.rgb, flash, p.intensity) : c.rgb, c.a), gid);
}

// Cartoon: posterize + edge lines.

kernel void stylizeCartoon(texture2d<float, access::sample> src [[texture(0)]],
                           texture2d<float, access::write>  dst [[texture(1)]],
                           constant CartoonParams& p [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 t = p.edgeWidth / float2(src.get_width(), src.get_height());
    float4 c = src.sample(linClamp, uv);
    float3 q = floor(c.rgb * p.levels + 0.5) / p.levels;

    float l  = luma(c.rgb);
    float dx = luma(src.sample(linClamp, uv + float2(t.x, 0)).rgb) - luma(src.sample(linClamp, uv - float2(t.x, 0)).rgb);
    float dy = luma(src.sample(linClamp, uv + float2(0, t.y)).rgb) - luma(src.sample(linClamp, uv - float2(0, t.y)).rgb);
    float edge = step(p.edgeThreshold, length(float2(dx, dy)));
    dst.write(float4(mix(q, p.lineColor, edge), c.a), gid);
}
