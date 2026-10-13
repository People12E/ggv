#include "FXCommon.h"

// =============================================================================
// Keying. Inputs/outputs are premultiplied; keying works on straight colour and writes
// alpha = matte * original alpha.
// =============================================================================

// Reference value for the primary screen channel `pi`, mixing the other two by `balance`
// (balance 0 = max of the others, 1 = min, 0.5 = their midpoint).
inline float fx_screen_ref(float3 v, int pi, float balance)
{
    float o1 = v[(pi + 1) % 3], o2 = v[(pi + 2) % 3];
    return mix(max(o1, o2), min(o1, o2), balance);
}

inline int fx_primary_channel(float3 K)
{
    return (K.g >= K.r && K.g >= K.b) ? 1 : ((K.b >= K.r) ? 2 : 0);
}

// ---- Keylight-style screen keyer -----------------------------------------------------------------------------
// Not Foundry's algorithm: a screen-difference matte normalised by the screen colour, with gain, balance,
// clip black/white and primary-channel despill.
// @fx fx_keylight: screenColor@0:4 screenGain@4 screenBalance@5 clipBlack@6 clipWhite@7 despill@8 viewMode@9
kernel void fx_keylight(FX_ARGS)
{
    FX_PROLOGUE
    float3 K = PARAM4(0).rgb;
    float gain = PARAM(4) * 0.01f;
    float bal  = saturate(PARAM(5) * 0.01f);
    float cb = PARAM(6) * 0.01f, cw = PARAM(7) * 0.01f;
    float despill = saturate(PARAM(8) * 0.01f);
    int   view = int(PARAM(9));                   // 0 final, 1 matte, 2 despilled source

    float4 c = unpremul(src.sample(s, uv));
    int pi = fx_primary_channel(K);
    float kd = max(K[pi] - fx_screen_ref(K, pi, bal), 1e-3f);

    float3 v = c.rgb;
    float refv = fx_screen_ref(v, pi, bal);
    float d = v[pi] - refv;
    float alpha = 1.0f - saturate(gain * d / kd);
    alpha = saturate((alpha - cb) / max(cw - cb, 1e-3f));

    float3 rgb = v;
    rgb[pi] = mix(rgb[pi], min(rgb[pi], refv), despill);

    if (view == 1)      dst.write(float4(float3(alpha) * c.a, c.a), gid);
    else if (view == 2) dst.write(float4(rgb * c.a, c.a), gid);
    else                dst.write(float4(rgb * alpha * c.a, alpha * c.a), gid);
}

// ---- Luma Key ----------------------------------------------------------------------------------------------------
inline float fx_luma_alpha(float l, int type, float thr, float tol)
{
    tol = max(tol, 1e-4f);
    switch (type) {
        case 0:  return 1.0f - smoothstep(thr - tol, thr + tol, l);          // key out brighter
        case 1:  return smoothstep(thr - tol, thr + tol, l);                 // key out darker
        case 2:  return smoothstep(0.0f, tol, fabs(l - thr));                // key out similar
        default: return 1.0f - smoothstep(0.0f, tol, fabs(l - thr));         // key out dissimilar
    }
}

// @fx fx_luma_key: keyType@0 threshold@1 tolerance@2 edgeThin@3 edgeFeather@4
kernel void fx_luma_key(FX_ARGS)
{
    FX_PROLOGUE
    int   type = int(PARAM(0));
    float thr = PARAM(1) * 0.01f, tol = PARAM(2) * 0.01f;
    float thin = PARAM(3), feather = PARAM(4);

    float4 c = unpremul(src.sample(s, uv));
    float a = fx_luma_alpha(luma709(c.rgb), type, thr, tol);
    float rot = fx_ign(float2(gid)) * FX_TAU;

    if (thin > 0.5f) {                                             // erode the matte
        for (int i = 0; i < 8; ++i) {
            float2 v = float2(cos(rot + float(i) * FX_TAU / 8.0f), sin(rot + float(i) * FX_TAU / 8.0f));
            float4 q = unpremul(src.sample(s, uv + v * thin / srcSize));
            a = min(a, fx_luma_alpha(luma709(q.rgb), type, thr, tol));
        }
    }
    if (feather > 0.5f) {                                          // soften the matte
        float acc = a;
        for (int i = 0; i < 16; ++i) {
            float4 q = unpremul(src.sample(s, uv + fx_vogel(i, 16, rot) * feather / srcSize));
            acc += fx_luma_alpha(luma709(q.rgb), type, thr, tol);
        }
        a = acc / 17.0f;
    }
    dst.write(float4(c.rgb * a * c.a, a * c.a), gid);
}

// ---- Color Range -----------------------------------------------------------------------------------------------------
inline float3 fx_to_space(float3 rgb, int cs)
{
    if (cs == 0) {                                                  // Lab, remapped to roughly 0..1
        float3 lab = rgb2lab(rgb);
        return float3(lab.x / 100.0f, (lab.y + 128.0f) / 255.0f, (lab.z + 128.0f) / 255.0f);
    }
    if (cs == 1) {                                                  // YUV
        float y = luma709(rgb);
        return float3(y, (rgb.b - y) * 0.492f + 0.5f, (rgb.r - y) * 0.877f + 0.5f);
    }
    return rgb;
}

// @fx fx_color_range: colorSpace@0 minColor@1:4 maxColor@5:4 fuzziness@9 invert@10
kernel void fx_color_range(FX_ARGS)
{
    FX_PROLOGUE
    int cs = int(PARAM(0));                                         // 0 Lab, 1 YUV, 2 RGB
    float3 mn = fx_to_space(PARAM4(1).rgb, cs), mx = fx_to_space(PARAM4(5).rgb, cs);
    float fz = max(PARAM(9) * 0.01f * 0.5f, 1e-4f);
    bool invert = PARAM(10) > 0.5f;

    float3 lo = min(mn, mx), hi = max(mn, mx);
    float4 c = unpremul(src.sample(s, uv));
    float3 x = fx_to_space(c.rgb, cs);
    float3 inside = smoothstep(lo - fz, lo, x) * (1.0f - smoothstep(hi, hi + fz, x));
    float matte = inside.x * inside.y * inside.z;                   // 1 = inside the range
    float a = invert ? matte : 1.0f - matte;                        // selected colours are keyed out
    dst.write(float4(c.rgb * a * c.a, a * c.a), gid);
}

// ---- Difference Matte (clean plate / reference layer in aux) ----------------------------------------------------------
// @fx fx_difference_matte: differenceMode@0 tolerance@1 softness@2 invert@3
kernel void fx_difference_matte(FX_ARGS)
{
    FX_PROLOGUE
    int mode = int(PARAM(0));                      // 0 colours, 1 luminance
    float tol = PARAM(1) * 0.01f, soft = PARAM(2) * 0.01f;
    bool invert = PARAM(3) > 0.5f;
    float4 c = unpremul(src.sample(s, uv));
    float4 r = unpremul(aux.sample(s, uv));
    float diff = (mode == 0) ? length(c.rgb - r.rgb) / 1.7320508f : fabs(luma709(c.rgb) - luma709(r.rgb));
    float a = smoothstep(tol, tol + max(soft, 1e-4f), diff);
    if (invert) a = 1.0f - a;
    dst.write(float4(c.rgb * a * c.a, a * c.a), gid);
}

// ---- Spill Suppressor ------------------------------------------------------------------------------------------------------
// @fx fx_spill_suppress: suppressColor@0:4 amount@4 mode@5 preserveLuminance@6
kernel void fx_spill_suppress(FX_ARGS)
{
    FX_PROLOGUE
    float3 K = PARAM4(0).rgb;
    float amount = saturate(PARAM(4) * 0.01f);
    int   mode = int(PARAM(5));                    // 0 average of the other channels, 1 their maximum
    bool  keepLuma = PARAM(6) > 0.5f;

    float4 c = unpremul(src.sample(s, uv));
    int pi = fx_primary_channel(K);
    float3 v = c.rgb;
    float refv = fx_screen_ref(v, pi, mode == 0 ? 0.5f : 0.0f);
    float excess = max(v[pi] - refv, 0.0f);
    float3 rgb = v;
    rgb[pi] -= excess * amount;
    if (keepLuma) rgb += (luma709(c.rgb) - luma709(rgb));
    c.rgb = max(rgb, 0.0f);
    dst.write(premul(c), gid);
}
