#include "FXCommon.h"

// =============================================================================
// Perspective: Basic 3D, Bevel Alpha, Bevel Edges, Drop Shadow, plus the 3D camera/layer draw
// (shadow-mapped, depth-writing) used by the compositor for real 3D layers.
// =============================================================================

// ---- Basic 3D: swivel (Y) / tilt (X) / distance, perspective ray-plane intersection ----------------
// @fx fx_basic3d: swivel@0 tilt@1 distance@2 fieldOfView@3 specular@4
kernel void fx_basic3d(FX_ARGS)
{
    FX_PROLOGUE
    float swivel = PARAM(0) * FX_PI / 180.0f;
    float tilt   = PARAM(1) * FX_PI / 180.0f;
    float dist   = PARAM(2);
    float fov    = clamp(PARAM(3), 5.0f, 150.0f) * FX_PI / 180.0f;
    float spec   = saturate(PARAM(4) * 0.01f);

    float f = 0.5f * size.x / tan(0.5f * fov);                   // focal length in pixels
    float3 ray = float3((uv - 0.5f) * size, f);                  // camera at origin looking +z, y down
    float cy = cos(swivel), sy = sin(swivel), cx = cos(tilt), sx = sin(tilt);
    float3x3 Ry = float3x3(float3(cy, 0.0f, -sy), float3(0.0f, 1.0f, 0.0f), float3(sy, 0.0f, cy));
    float3x3 Rx = float3x3(float3(1.0f, 0.0f, 0.0f), float3(0.0f, cx, sx), float3(0.0f, -sx, cx));
    float3x3 R = Rx * Ry;
    float3 C = float3(0.0f, 0.0f, f + dist);
    float3 N = R * float3(0.0f, 0.0f, 1.0f);

    float denom = dot(ray, N);
    if (fabs(denom) < 1e-5f) { dst.write(float4(0.0f), gid); return; }
    float t = dot(C, N) / denom;
    if (t <= 0.0f) { dst.write(float4(0.0f), gid); return; }
    float3 H = ray * t;
    float3 local = transpose(R) * (H - C);
    float4 c = src.sample(sz, local.xy / size + 0.5f);

    float3 toCam = normalize(-H);
    float3 toLight = normalize(float3(-0.3f, -0.5f, -1.0f));
    float3 hv = normalize(toCam + toLight);
    float highlight = pow(saturate(dot(-N, hv)), 50.0f) * spec;
    c.rgb += highlight * c.a;
    dst.write(c, gid);
}

// ---- Bevel helpers ---------------------------------------------------------------------------------
inline float3 fx_bevel_apply(float3 rgb, float a, float shade, float intensity, float4 lightColor, float mask)
{
    float lit  = max(shade, 0.0f) * intensity * mask;
    float dark = max(-shade, 0.0f) * intensity * mask;
    return (rgb + lightColor.rgb * lightColor.a * lit * a) * (1.0f - dark);
}

// ---- Bevel Alpha: light/dark edges from the blurred alpha gradient --------------------------------------------
// @fx fx_bevel_alpha: edgeThickness@0 lightAngle@1 lightIntensity@2 lightColor@3:4
kernel void fx_bevel_alpha(FX_ARGS)
{
    FX_PROLOGUE
    float R = max(PARAM(0), 0.5f);
    float2 L2 = fx_ae_dir(PARAM(1));              // direction towards the light
    float inten = PARAM(2) * 0.01f;
    float4 lc = PARAM4(3);

    float4 o = src.sample(s, uv);
    const int N = 32;
    float rot = fx_ign(float2(gid)) * FX_TAU;
    float2 g = float2(0.0f);
    float blurA = 0.0f;
    for (int i = 0; i < N; ++i) {
        float2 v = fx_vogel(i, N, rot);
        float a = src.sample(sz, uv + v * R / srcSize).a;
        g += v * a;
        blurA += a;
    }
    g /= float(N); blurA /= float(N);

    float mask = o.a * (1.0f - smoothstep(0.85f, 1.0f, blurA));            // only near the inner edge
    float3 n = normalize(float3(-g * 3.0f, 1.0f));
    float3 Lv = normalize(float3(L2 * 0.8f, 0.6f));
    float shade = dot(n, Lv) - Lv.z;
    dst.write(float4(fx_bevel_apply(o.rgb, o.a, shade, inten, lc, mask), o.a), gid);
}

// ---- Bevel Edges: bevel of the layer's rectangular border ---------------------------------------------------------------
// @fx fx_bevel_edges: edgeThickness@0 lightAngle@1 lightIntensity@2 lightColor@3:4
kernel void fx_bevel_edges(FX_ARGS)
{
    FX_PROLOGUE
    float R = max(PARAM(0), 0.5f);
    float2 L2 = fx_ae_dir(PARAM(1));
    float inten = PARAM(2) * 0.01f;
    float4 lc = PARAM4(3);

    float4 o = src.sample(s, uv);
    float2 p = uv * size;
    float2 dEdge = min(p, size - p);                                  // distance to nearest edge per axis
    float2 slope = 1.0f - smoothstep(0.0f, R, dEdge);
    float2 inward = float2(p.x < size.x * 0.5f ? 1.0f : -1.0f, p.y < size.y * 0.5f ? 1.0f : -1.0f);
    float3 n = normalize(float3(-inward * slope * 1.5f, 1.0f));
    float3 Lv = normalize(float3(L2 * 0.8f, 0.6f));
    float shade = dot(n, Lv) - Lv.z;
    dst.write(float4(fx_bevel_apply(o.rgb, o.a, shade, inten, lc, 1.0f), o.a), gid);
}

// ---- Drop Shadow (Vogel-disc soft shadow; direction uses the After Effects angle convention) --------------------------------
// @fx fx_drop_shadow: shadowColor@0:4 opacity@4 direction@5 distance@6 softness@7 shadowOnly@8
kernel void fx_drop_shadow(FX_ARGS)
{
    FX_PROLOGUE
    float4 col = PARAM4(0);
    float opacity = saturate(PARAM(4) * 0.01f);
    float2 off = fx_ae_dir(PARAM(5)) * PARAM(6) / size;
    float soft = max(PARAM(7), 0.0f);
    bool only = PARAM(8) > 0.5f;

    float sa;
    if (soft < 0.5f) {
        sa = src.sample(sz, uv - off).a;
    } else {
        const int N = 48;
        float rot = fx_ign(float2(gid)) * FX_TAU;
        sa = 0.0f;
        for (int i = 0; i < N; ++i) sa += src.sample(sz, uv - off + fx_vogel(i, N, rot) * soft / srcSize).a;
        sa /= float(N);
    }
    float sA = col.a * opacity * sa;
    float3 sC = col.rgb * sA;
    float4 o = src.sample(s, uv);
    dst.write(only ? float4(sC, sA) : float4(o.rgb + sC * (1.0f - o.a), o.a + sA * (1.0f - o.a)), gid);
}

// =============================================================================
// Real 3D layers (drawn with a render pipeline, not the compute convention above).
// Pipeline: shadow pass (depth only, light view) -> main pass with colour(0) = premultiplied RGBA
// and colour(1) = linear depth / far (feeds the Camera DOF effect through `aux`).
// Layer planes face -z by default (the camera looks down +z); lighting is two-sided.
// =============================================================================
struct Layer3DUniforms {
    float4x4 mvp;
    float4x4 model;
    float4x4 lightViewProj;     // world -> light clip, used for lookups and the shadow pass
    float4   lightPosDir;       // xyz: position (point) or direction towards the light (directional); w: 0 dir, 1 point
    float4   lightColor;        // rgb * intensity
    float4   ambient;           // rgb
    float4   misc;              // x opacity, y shadow bias, z receivesLight, w receivesShadows
    float4   sizeFar;           // xy layer size in px, z far plane
};

struct L3DVert {
    float4 position [[position]];
    float2 uv;
    float3 world;
    float4 lightClip;
    float3 normal;
    float  viewZ;
};

inline float4 fx_l3d_corner(uint vid, constant Layer3DUniforms& U, thread float2& uv)
{
    float2 corner = float2((vid & 1u) ? 1.0f : -1.0f, (vid & 2u) ? 1.0f : -1.0f);
    uv = corner * 0.5f + 0.5f;                                   // layer space is y-down
    return float4(corner.x * 0.5f * U.sizeFar.x, corner.y * 0.5f * U.sizeFar.y, 0.0f, 1.0f);
}

vertex L3DVert fx_layer3d_vertex(uint vid [[vertex_id]], constant Layer3DUniforms& U [[buffer(0)]])
{
    L3DVert o;
    float2 uv;
    float4 local = fx_l3d_corner(vid, U, uv);
    float4 w = U.model * local;
    o.position = U.mvp * local;
    o.uv = uv;
    o.world = w.xyz;
    o.lightClip = U.lightViewProj * w;
    o.normal = normalize((U.model * float4(0.0f, 0.0f, -1.0f, 0.0f)).xyz);
    o.viewZ = o.position.w;
    return o;
}

struct L3DOut {
    float4 color [[color(0)]];
    float  depth [[color(1)]];
};

fragment L3DOut fx_layer3d_fragment(L3DVert in [[stage_in]],
                                    constant Layer3DUniforms& U [[buffer(0)]],
                                    texture2d<float, access::sample> tex [[texture(0)]],
                                    depth2d<float, access::sample> shadowMap [[texture(1)]])
{
    constexpr sampler ls(coord::normalized, address::clamp_to_zero, filter::linear);
    constexpr sampler shs(coord::normalized, address::clamp_to_edge, filter::linear, compare_func::less_equal);

    float4 c = tex.sample(ls, in.uv) * U.misc.x;
    float3 lit = float3(1.0f);
    if (U.misc.z > 0.5f) {
        float3 L = (U.lightPosDir.w > 0.5f) ? normalize(U.lightPosDir.xyz - in.world) : normalize(U.lightPosDir.xyz);
        float diff = fabs(dot(normalize(in.normal), L));
        float shadow = 1.0f;
        if (U.misc.w > 0.5f) {
            float3 ndc = in.lightClip.xyz / in.lightClip.w;
            float2 suv = float2(ndc.x * 0.5f + 0.5f, 0.5f - ndc.y * 0.5f);
            if (suv.x >= 0.0f && suv.x <= 1.0f && suv.y >= 0.0f && suv.y <= 1.0f) {
                float2 texel = 1.0f / float2(shadowMap.get_width(), shadowMap.get_height());
                float sum = 0.0f;
                for (int j = -1; j <= 1; ++j)
                    for (int i = -1; i <= 1; ++i)
                        sum += shadowMap.sample_compare(shs, suv + float2(i, j) * texel, ndc.z - U.misc.y);
                shadow = sum / 9.0f;
            }
        }
        lit = U.ambient.rgb + U.lightColor.rgb * diff * shadow;
    }
    L3DOut res;
    res.color = float4(c.rgb * lit, c.a);
    res.depth = in.viewZ / max(U.sizeFar.z, 1.0f);
    return res;
}

// Shadow pass: depth only from the light; alpha-tested so transparent layer regions don't cast.
struct L3DShadowVert { float4 position [[position]]; float2 uv; };

vertex L3DShadowVert fx_shadow_vertex(uint vid [[vertex_id]], constant Layer3DUniforms& U [[buffer(0)]])
{
    L3DShadowVert o;
    float2 uv;
    float4 local = fx_l3d_corner(vid, U, uv);
    o.position = U.lightViewProj * (U.model * local);
    o.uv = uv;
    return o;
}

fragment void fx_shadow_fragment(L3DShadowVert in [[stage_in]], texture2d<float, access::sample> tex [[texture(0)]])
{
    constexpr sampler ls(coord::normalized, address::clamp_to_zero, filter::linear);
    if (tex.sample(ls, in.uv).a < 0.5f) discard_fragment();
}
