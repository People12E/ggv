#include <metal_stdlib>
#include "BPShaderTypes.h"
using namespace metal;

constexpr sampler linClamp(coord::normalized, address::clamp_to_edge, filter::linear);

static inline float luma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

#define PIXEL_GUARD(dst) if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return; \
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());

// ===========================================================================
// PERSPECTIVE
// ===========================================================================

// Drop Shadow (2D): offset + blurred alpha, composited under the layer.
// Run a Gaussian (Blur.metal) on the alpha first; `blurredAlpha` is that result.

kernel void perspectiveDropShadow(texture2d<float, access::sample> src          [[texture(0)]],
                                  texture2d<float, access::sample> blurredAlpha [[texture(1)]],
                                  texture2d<float, access::write>  dst          [[texture(2)]],
                                  constant DropShadowParams& p [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    float sa = blurredAlpha.sample(linClamp, uv - p.offset).a * p.opacity;
    float4 shadow = float4(p.color * sa, sa);
    if (p.shadowOnly != 0) { dst.write(shadow, gid); return; }
    // layer over shadow (premultiplied)
    float4 outc = float4(c.rgb + shadow.rgb * (1.0 - c.a), c.a + shadow.a * (1.0 - c.a));
    dst.write(outc, gid);
}

// Bevel Alpha / Bevel Edges: lighting from the alpha (or luma) height field.

// Height field used by both bevels. mode 0: alpha, 1: luma, 2: distance to the layer rectangle (Bevel Edges).
static inline float bevelHeight(texture2d<float, access::sample> hm, float2 uv, constant BevelParams& p) {
    if (p.mode == 2) {
        float2 size = float2(hm.get_width(), hm.get_height());
        float2 px = uv * size;
        float e = min(min(px.x, size.x - px.x), min(px.y, size.y - px.y));
        return saturate(e / max(p.edgeWidth, 1e-3));
    }
    float4 s = hm.sample(linClamp, uv);
    return p.mode == 1 ? luma(s.rgb) : s.a;
}

kernel void perspectiveBevel(texture2d<float, access::sample> src       [[texture(0)]],
                             texture2d<float, access::sample> heightMap [[texture(1)]],   // blurred alpha for a soft Bevel Alpha; ignored by mode 2
                             texture2d<float, access::write>  dst       [[texture(2)]],
                             constant BevelParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float2 t = 1.0 / float2(heightMap.get_width(), heightMap.get_height());
    float dx = bevelHeight(heightMap, uv + float2(t.x, 0), p) - bevelHeight(heightMap, uv - float2(t.x, 0), p);
    float dy = bevelHeight(heightMap, uv + float2(0, t.y), p) - bevelHeight(heightMap, uv - float2(0, t.y), p);
    float3 n = normalize(float3(-dx * p.depth, -dy * p.depth, 1.0));
    float3 L = normalize(float3(p.lightDir, 0.7));
    float lit = dot(n, L) - L.z;                 // 0 on flat areas, +/- on slopes
    float4 c = src.sample(linClamp, uv);
    float3 tint = lit > 0 ? p.highlight : p.shadow;
    float3 o = c.rgb + tint * abs(lit) * p.intensity * c.a;
    dst.write(float4(o, c.a), gid);
}

// "Basic 3D": plane in 3D with rotation (swivel/tilt) and perspective, backward-mapped.
// viewProjInverseRows: homography from output pixel to the plane's local uv (host computes
// from camera FOV, plane transform).  Same machinery as Corner Pin, plus back-face / specular.

kernel void perspectiveBasic3D(texture2d<float, access::sample> src [[texture(0)]],
                               texture2d<float, access::write>  dst [[texture(1)]],
                               constant Basic3DParams& p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float3 h = p.outToPlane * float3(uv, 1.0);
    if (h.z <= 0) { dst.write(float4(0), gid); return; }
    float2 s = h.xy / h.z;
    if (any(s < 0.0) || any(s > 1.0)) { dst.write(float4(0), gid); return; }
    float4 c = src.sample(linClamp, s);
    float3 V = float3(0, 0, 1);
    float3 Hh = normalize(normalize(p.lightDir) + V);
    float spec = pow(saturate(dot(normalize(p.normal), Hh)), p.shine) * p.specular;
    dst.write(float4(c.rgb + spec * c.a, c.a), gid);
}

// Depth-aware 3D layer compositing: write layer depth so later layers/shadows can test against it.

kernel void perspectiveWriteDepth(texture2d<float, access::sample> color [[texture(0)]],
                                  texture2d<float, access::read_write> depthBuf [[texture(1)]],  // r32Float
                                  texture2d<float, access::read_write> accum    [[texture(2)]],
                                  constant LayerDepthParams& p [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= accum.get_width() || gid.y >= accum.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(accum.get_width(), accum.get_height());
    float4 c = color.sample(linClamp, uv);
    if (c.a < 0.01) return;
    float d = depthBuf.read(gid).r;
    if (p.depth < d) {                    // nearer than what's stored
        depthBuf.write(float4(p.depth), gid);
        float4 below = accum.read(gid);
        accum.write(float4(c.rgb + below.rgb * (1.0 - c.a), c.a + below.a * (1.0 - c.a)), gid);
    }
}

// Shadow map comparison (PCF) for a directional/spot light: depth stored from the light's view.

kernel void perspectiveApplyShadow(texture2d<float, access::sample> color      [[texture(0)]],
                                   texture2d<float, access::sample> worldPos   [[texture(1)]],  // rgb = world-space position
                                   depth2d<float, access::sample>   shadowMap  [[texture(2)]],
                                   texture2d<float, access::write>  dst        [[texture(3)]],
                                   constant ShadowParams& p [[buffer(0)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    constexpr sampler cmp(coord::normalized, address::clamp_to_edge, filter::linear, compare_func::less_equal);
    float4 c = color.sample(linClamp, uv);
    float3 wp = worldPos.sample(linClamp, uv).rgb;
    float4 lp = p.lightViewProj * float4(wp, 1.0);
    float3 ndc = lp.xyz / lp.w;
    float2 suv = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    float lit = 0.0;
    float2 texel = p.softness / float2(shadowMap.get_width(), shadowMap.get_height());
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x)
            lit += shadowMap.sample_compare(cmp, suv + float2(x, y) * texel, ndc.z - p.bias);
    lit /= 9.0;
    float shade = mix(1.0, lit, p.strength);
    dst.write(float4(c.rgb * shade, c.a), gid);
}

// ===========================================================================
// KEYING
// ===========================================================================

// Keylight-style chroma key (difference of key-channel vs. other two) with clip black/white,
// plus screen-balance and optional despill. Output: RGBA with matte in alpha.

kernel void keyChroma(texture2d<float, access::sample> src [[texture(0)]],
                      texture2d<float, access::write>  dst [[texture(1)]],
                      constant KeyParams& p [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);

    // Determine dominant key channel (green screen default: index 1).
    float3 k = normalize(p.keyColor + 1e-4);
    int ch = (k.g >= k.r && k.g >= k.b) ? 1 : (k.b >= k.r ? 2 : 0);
    float key  = ch == 0 ? c.r : (ch == 1 ? c.g : c.b);
    float o1   = ch == 0 ? c.g : (ch == 1 ? c.r : c.r);
    float o2   = ch == 0 ? c.b : (ch == 1 ? c.b : c.g);
    float other = mix(o1, o2, p.screenBalance);          // balance between the two non-key channels

    float screenMatte = (key - other) * p.screenGain;    // high = more screen-colored => more transparent
    float alpha = 1.0 - saturate(screenMatte);
    alpha = saturate((alpha - p.clipBlack) / max(p.clipWhite - p.clipBlack, 1e-4));

    // Despill: clamp key channel to the average of the other two.
    float3 rgb = c.rgb;
    float limit = (o1 + o2) * 0.5;
    float spill = max(key - limit, 0.0) * p.despill;
    if (ch == 0) rgb.r -= spill; else if (ch == 1) rgb.g -= spill; else rgb.b -= spill;

    dst.write(float4(rgb * alpha, alpha * c.a), gid);    // premultiplied
}


kernel void keyLuma(texture2d<float, access::sample> src [[texture(0)]],
                    texture2d<float, access::write>  dst [[texture(1)]],
                    constant LumaKeyParams& p [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    float l = luma(c.rgb);
    float d = p.keyOutBrighter != 0 ? (l - p.threshold) : (p.threshold - l);   // >0 means key out
    float a = 1.0 - smoothstep(-p.edgeFeather, p.edgeFeather + p.tolerance, d);
    dst.write(float4(c.rgb * a, c.a * a), gid);
}

// Color Range key in Lab-ish space (approximated by YCbCr distance box).

static inline float3 rgb2ycc(float3 c) {
    float y = luma(c);
    return float3(y, (c.b - y) * 0.5389 + 0.5, (c.r - y) * 0.6350 + 0.5);
}

kernel void keyColorRange(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant ColorRangeParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    float3 y = rgb2ycc(c.rgb);
    float3 inside = smoothstep(p.minYCC - p.feather, p.minYCC, y) * (1.0 - smoothstep(p.maxYCC, p.maxYCC + p.feather, y));
    float key = inside.x * inside.y * inside.z;
    float a = 1.0 - key;
    dst.write(float4(c.rgb * a, c.a * a), gid);
}


kernel void keyDifferenceMatte(texture2d<float, access::sample> src       [[texture(0)]],
                               texture2d<float, access::sample> reference [[texture(1)]],
                               texture2d<float, access::write>  dst       [[texture(2)]],
                               constant DiffMatteParams& p [[buffer(0)]],
                               uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    float d = length(c.rgb - reference.sample(linClamp, uv).rgb);
    float a = smoothstep(p.tolerance, p.tolerance + p.softness + 1e-4, d);
    dst.write(float4(c.rgb * a, c.a * a), gid);
}

// Spill Suppressor: remove residual key color from edges without touching alpha.

kernel void keySpillSuppress(texture2d<float, access::sample> src [[texture(0)]],
                             texture2d<float, access::write>  dst [[texture(1)]],
                             constant SpillParams& p [[buffer(0)]],
                             uint2 gid [[thread_position_in_grid]])
{
    PIXEL_GUARD(dst)
    float4 c = src.sample(linClamp, uv);
    // Key weights: the dominant key channel(s) vs. the mean of the remaining channel(s).
    // Neutral colors (r=g=b) give exactly 0 spill, so grays are never tinted.
    float3 k = p.keyColor / max(p.keyColor.r + p.keyColor.g + p.keyColor.b, 1e-4);   // sums to 1
    float3 o = (float3(1.0) - k);
    float keyVal   = dot(c.rgb, k);
    float otherVal = dot(c.rgb, o) / max(o.r + o.g + o.b, 1e-4);
    float amountOfKey = max(keyVal - otherVal, 0.0);
    float3 rgb = c.rgb - k * amountOfKey * p.amount;
    rgb = mix(rgb, float3(luma(rgb)), saturate(amountOfKey * p.amount * 0.5));   // desaturate leftover tint
    dst.write(float4(max(rgb, 0.0), c.a), gid);
}

// ---------------------------------------------------------------------------
// 3D Camera: a layer is a plane in world space, viewed through a perspective (or orthographic) camera.
// Per output pixel: unproject a ray, intersect the layer plane, sample the layer, and write a normalized
// depth (0 = near, 1 = far / no hit) that blurLens / depth-of-field and the shadow pass can consume.
// Output colour is premultiplied.
// ---------------------------------------------------------------------------
kernel void perspectiveCamera3D(texture2d<float, access::sample> layer    [[texture(0)]],
                                texture2d<float, access::write>  dst      [[texture(1)]],
                                texture2d<float, access::write>  depthOut [[texture(2)]],   // r32Float
                                constant Camera3DParams& p [[buffer(0)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 size = float2(dst.get_width(), dst.get_height());
    float2 pix = float2(gid) + 0.5;
    float2 ndc = float2(pix.x / size.x * 2.0 - 1.0, 1.0 - pix.y / size.y * 2.0);

    float4 n4 = p.invViewProj * float4(ndc, 0.0, 1.0);
    float4 f4 = p.invViewProj * float4(ndc, 1.0, 1.0);
    float3 o = n4.xyz / n4.w;
    float3 d = normalize(f4.xyz / f4.w - o);

    float3 ol = (p.worldToLayer * float4(o, 1.0)).xyz;
    float3 dl = (p.worldToLayer * float4(d, 0.0)).xyz;

    float t = -ol.z / (abs(dl.z) < 1e-8 ? 1e-8 : dl.z);
    float3 N = normalize(p.normal);
    bool backFacing = dot(N, d) > 0.0;
    bool miss = (abs(dl.z) < 1e-8) || (t <= 0.0) || (backFacing && p.twoSided == 0);

    float2 lp = ol.xy + dl.xy * t;
    float2 edge = min(lp, p.layerSize - lp);
    float cover = saturate(min(edge.x, edge.y) + 0.5);       // 1px anti-aliased silhouette
    if (miss || cover <= 0.0) {
        dst.write(float4(0), gid);
        depthOut.write(float4(1.0), gid);
        return;
    }

    if (backFacing) N = -N;
    float4 c = layer.sample(linClamp, lp / p.layerSize);
    float3 L = normalize(p.lightDir);
    float3 H = normalize(L - d);                              // view vector is -d
    float spec = pow(saturate(dot(N, H)), p.shine) * p.specular;

    float a = cover * p.opacity;
    dst.write(float4((c.rgb + spec * c.a) * a, c.a * a), gid);
    depthOut.write(float4(saturate((t - p.depthNear) / max(p.depthFar - p.depthNear, 1e-4))), gid);
}
