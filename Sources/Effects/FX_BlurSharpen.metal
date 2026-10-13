#include "FXCommon.h"

// =============================================================================
// Blur & Sharpen
// =============================================================================

// 1-D symmetric gather used by Gaussian / Box / Unsharp. `radius` is the half-extent in texels.
// Beyond 48 taps the spacing grows and the bilinear filter fills the gaps.
inline float4 fx_blur_line(texture2d<float, access::sample> t, float2 uv, float2 dirTexel,
                           float radius, bool gaussian, bool repeatEdge)
{
    constexpr sampler sc(coord::normalized, address::clamp_to_edge, filter::linear);
    constexpr sampler sb(coord::normalized, address::clamp_to_zero, filter::linear);
    int taps = int(min(ceil(radius), 48.0f));
    float stride = taps > 0 ? radius / float(taps) : 0.0f;
    float sigma = max(radius / 3.0f, 1e-3f);
    float4 acc = repeatEdge ? t.sample(sc, uv) : t.sample(sb, uv);
    float wsum = 1.0f;
    for (int i = 1; i <= taps; ++i) {
        float d = float(i) * stride;
        float w = gaussian ? exp(-0.5f * d * d / (sigma * sigma)) : 1.0f;
        float2 o = dirTexel * d;
        float4 a = repeatEdge ? t.sample(sc, uv + o) : t.sample(sb, uv + o);
        float4 b = repeatEdge ? t.sample(sc, uv - o) : t.sample(sb, uv - o);
        acc += w * (a + b);
        wsum += 2.0f * w;
    }
    return acc / wsum;
}

// ---- Gaussian Blur: two passes, tag 0 = horizontal, tag 1 = vertical ----------
// @fx fx_gaussian: blurriness@0 dimensions@1 repeatEdge@2
kernel void fx_gaussian(FX_ARGS)
{
    FX_PROLOGUE
    float blurriness = PARAM(0);
    int   dims       = int(PARAM(1));          // 0 both, 1 horizontal, 2 vertical
    bool  repeatEdge = PARAM(2) > 0.5f;
    bool  horizontal = tag < 0.5f;
    if (blurriness < 0.05f || (horizontal && dims == 2) || (!horizontal && dims == 1)) {
        dst.write(src.sample(s, uv), gid);
        return;
    }
    float2 dir = (horizontal ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f)) / srcSize;
    dst.write(fx_blur_line(src, uv, dir, blurriness * 1.5f, true, repeatEdge), gid);
}

// ---- Fast Box Blur: the runner repeats both passes `iterations` times ---------
// @fx fx_box: blurriness@0 dimensions@1 repeatEdge@3
kernel void fx_box(FX_ARGS)
{
    FX_PROLOGUE
    float blurriness = PARAM(0);
    int   dims       = int(PARAM(1));
    bool  repeatEdge = PARAM(3) > 0.5f;
    bool  horizontal = tag < 0.5f;
    if (blurriness < 0.05f || (horizontal && dims == 2) || (!horizontal && dims == 1)) {
        dst.write(src.sample(s, uv), gid);
        return;
    }
    float2 dir = (horizontal ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f)) / srcSize;
    dst.write(fx_blur_line(src, uv, dir, blurriness * 0.5f, false, repeatEdge), gid);
}

// ---- Directional Blur -----------------------------------------------------------
// @fx fx_directional: direction@0 length@1
kernel void fx_directional(FX_ARGS)
{
    FX_PROLOGUE
    float len = PARAM(1);
    float2 step = fx_ae_dir(PARAM(0)) / srcSize;     // After Effects convention: 0 deg = up, clockwise
    const int N = 64;
    float jitter = fx_ign(float2(gid));
    float4 acc = float4(0.0f);
    for (int i = 0; i < N; ++i) {
        float t = (float(i) + jitter) / float(N) - 0.5f;
        acc += src.sample(s, uv + step * (len * t));
    }
    dst.write(acc / float(N), gid);
}

// ---- Radial Blur (spin or zoom) ----------------------------------------------------
// @fx fx_radial: type@0 amount@1 center@2:2
kernel void fx_radial(FX_ARGS)
{
    FX_PROLOGUE
    int    type   = int(PARAM(0));            // 0 spin, 1 zoom
    float  amount = PARAM(1);                 // spin: degrees, zoom: percent
    float2 c      = PARAM2(2);                // normalized 0..1
    const int N = 48;
    float jitter = fx_ign(float2(gid));
    float2 d = (uv - c) * size;               // pixel space so rotation keeps aspect
    float4 acc = float4(0.0f);
    for (int i = 0; i < N; ++i) {
        float t = (float(i) + jitter) / float(N) - 0.5f;
        float2 p;
        if (type == 0) p = c + fx_rot(d, amount * FX_PI / 180.0f * t) / size;
        else           p = c + (uv - c) * (1.0f + amount * 0.01f * t);
        acc += src.sample(s, p);
    }
    dst.write(acc / float(N), gid);
}

// ---- Smart Blur: edge-preserving (bilateral) ------------------------------------
// @fx fx_smart_blur: radius@0 threshold@1 mode@2
kernel void fx_smart_blur(FX_ARGS)
{
    FX_PROLOGUE
    float radius = max(PARAM(0), 0.0f);
    float thr    = max(PARAM(1) * 0.01f, 1e-4f);     // 0..100 -> 0..1 colour distance
    int   mode   = int(PARAM(2));                     // 0 normal, 1 edge only, 2 overlay edge
    float4 centre = src.sample(s, uv);
    int r = int(min(ceil(radius), 5.0f));
    float stride = r > 0 ? radius / float(r) : 0.0f;
    float sigmaS = max(radius * 0.5f, 0.5f);
    float4 acc = centre;
    float wsum = 1.0f;
    for (int dy = -r; dy <= r; ++dy) {
        for (int dx = -r; dx <= r; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float2 o = float2(dx, dy) * stride;
            float4 c = src.sample(s, uv + o / srcSize);
            float ws = exp(-dot(o, o) / (2.0f * sigmaS * sigmaS));
            float dc = length(c - centre);
            float wr = 1.0f - smoothstep(thr * 0.5f, thr, dc);
            float w = ws * wr;
            acc += w * c;
            wsum += w;
        }
    }
    float4 blurred = acc / wsum;
    float edge = saturate(length(centre.rgb - blurred.rgb) * 4.0f);
    float4 outc = blurred;
    if (mode == 1)      outc = float4(float3(edge) * centre.a, centre.a);
    else if (mode == 2) outc = float4(mix(blurred.rgb, float3(centre.a), edge), blurred.a);
    dst.write(outc, gid);
}

// ---- Camera Lens Blur (iris shape, optional depth map in aux, highlight boost) ----
// @fx fx_lens_blur: radius@0 blades@1 rotation@2 curvature@3 depthSource@4 focalDistance@5 invertDepth@6 highlightGain@7 highlightThreshold@8
kernel void fx_lens_blur(FX_ARGS)
{
    FX_PROLOGUE
    float maxR      = max(PARAM(0), 0.0f);
    float blades    = PARAM(1);
    float rot       = PARAM(2) * FX_PI / 180.0f;
    float curv      = saturate(PARAM(3));
    bool  useDepth  = PARAM(4) > 0.5f;
    float focal     = PARAM(5);
    bool  invDepth  = PARAM(6) > 0.5f;
    float hlGain    = PARAM(7);
    float hlThresh  = PARAM(8);

    float4 centre = src.sample(s, uv);
    if (maxR < 0.5f) { dst.write(centre, gid); return; }

    float zP = 0.0f, cocP = maxR;
    if (useDepth) {
        float d = aux.sample(s, uv).r;
        zP = invDepth ? 1.0f - d : d;
        cocP = maxR * min(1.0f, fabs(zP - focal) * 2.0f);
    }

    const int N = 96;
    float spin = fx_ign(float2(gid)) * FX_TAU;
    float4 acc = centre;
    float wsum = 1.0f;
    for (int i = 0; i < N; ++i) {
        float2 v = fx_vogel(i, N, spin);                    // unit disc
        if (blades >= 3.0f) {
            float th = atan2(v.y, v.x) - rot;
            float seg = FX_TAU / blades;
            float polyR = cos(FX_PI / blades) / cos(fx_fmod(th, seg) - seg * 0.5f);
            v *= mix(polyR, 1.0f, curv);
        }
        float2 off = v * maxR;
        float dist = length(off);
        float2 q = uv + off / srcSize;
        float w = 1.0f;
        if (useDepth) {
            float dq = aux.sample(s, q).r;
            float zQ = invDepth ? 1.0f - dq : dq;
            float cocQ = maxR * min(1.0f, fabs(zQ - focal) * 2.0f);
            // Sample q may blur over p if its disc reaches p; a farther q must not bleed over a sharper p.
            float reach = (zQ <= zP + 0.02f) ? cocQ : min(cocQ, cocP);
            w = saturate(reach - dist + 1.0f);
        }
        float4 c = src.sample(s, q);
        w *= 1.0f + hlGain * smoothstep(hlThresh, 1.0f, luma709(c.rgb));
        acc += w * c;
        wsum += w;
    }
    dst.write(acc / wsum, gid);
}

// ---- Unsharp Mask: pass 0/1 blur (H, V), pass 2 combine with the original in aux ------
// @fx fx_unsharp_blur: radius@0
kernel void fx_unsharp_blur(FX_ARGS)
{
    FX_PROLOGUE
    bool horizontal = tag < 0.5f;
    float2 dir = (horizontal ? float2(1.0f, 0.0f) : float2(0.0f, 1.0f)) / srcSize;
    dst.write(fx_blur_line(src, uv, dir, max(PARAM(0), 0.0f) * 1.5f, true, true), gid);
}

// @fx fx_unsharp_apply: amount@1 threshold@2
kernel void fx_unsharp_apply(FX_ARGS)
{
    FX_PROLOGUE
    float amount = PARAM(1) * 0.01f;                  // percent
    float thr    = PARAM(2) / 255.0f;
    float4 blurred  = src.sample(s, uv);
    float4 original = aux.sample(s, uv);
    float3 diff = original.rgb - blurred.rgb;
    float3 gate = step(float3(thr), fabs(diff));
    float3 rgb = clamp(original.rgb + diff * amount * gate, 0.0f, max(original.a, 0.0f));
    dst.write(float4(rgb, original.a), gid);
}

// ---- Kawase blur step (shared by Glow and Saber; offset grows with the repeat index) ----
// @fx fx_kawase:
kernel void fx_kawase(FX_ARGS)
{
    FX_PROLOGUE
    float2 texel = 1.0f / srcSize;
    float2 o = (float(repIdx) + 0.5f) * texel;
    float4 sum  = src.sample(s, uv + float2( o.x,  o.y));
    sum        += src.sample(s, uv + float2(-o.x,  o.y));
    sum        += src.sample(s, uv + float2( o.x, -o.y));
    sum        += src.sample(s, uv + float2(-o.x, -o.y));
    dst.write(sum * 0.25f, gid);
}
