#include "FXCommon.h"

// =============================================================================
// Stylize
// =============================================================================

// ---- Glow: pass 0 extract (half res), pass 1... Kawase (repeat count derived from radius),
//      final pass composite with the ORIGINAL layer in aux ------------------------------------------------------
// @fx fx_glow_extract: threshold@0 softness@1
kernel void fx_glow_extract(FX_ARGS)
{
    FX_PROLOGUE
    float threshold = PARAM(0);
    float knee = max(PARAM(1), 1e-3f);
    float4 c = src.sample(s, uv);
    float l = max(c.r, max(c.g, c.b));
    float soft = clamp(l - threshold + knee, 0.0f, 2.0f * knee);
    soft = soft * soft / (4.0f * knee + 1e-5f);
    float w = max(soft, l - threshold) / max(l, 1e-5f);
    dst.write(float4(c.rgb * max(w, 0.0f), c.a * saturate(w)), gid);
}

// @fx fx_glow_composite: intensity@3 colorMode@4 colorA@5:4 colorB@9:4 operation@13
kernel void fx_glow_composite(FX_ARGS)
{
    FX_PROLOGUE
    float intensity = PARAM(3);
    int   colorMode = int(PARAM(4));              // 0 original colours, 1 A & B colours
    float4 cA = PARAM4(5), cB = PARAM4(9);
    int   op = int(PARAM(13));                    // 0 add, 1 screen

    float4 glow = src.sample(s, uv);              // blurred glow (lower res, bilinear upsampled)
    float4 o = aux.sample(s, uv);                 // original layer
    float g = luma709(glow.rgb);
    float3 gc = glow.rgb;
    if (colorMode == 1) gc = mix(cA.rgb, cB.rgb, saturate(g)) * g;
    gc *= intensity;
    float3 rgb = (op == 1) ? 1.0f - (1.0f - o.rgb) * (1.0f - saturate(gc)) : o.rgb + gc;
    float glowA = saturate(max(gc.r, max(gc.g, gc.b)));
    dst.write(float4(rgb, max(o.a, glowA)), gid);
}

// ---- Saber-style energy glow ---------------------------------------------------------------------------------------------------------
// Passes: core extract -> Kawase (outer glow) -> composite with original in aux (adds core + inner glow + flicker/turbulence).
// @fx fx_saber_core: threshold@0 softness@1
kernel void fx_saber_core(FX_ARGS)
{
    FX_PROLOGUE
    float4 c = src.sample(s, uv);
    float m = luma709(unpremul(c).rgb);                      // brightness drives the core
    float core = smoothstep(PARAM(0), PARAM(0) + max(PARAM(1), 1e-3f), m) * c.a;
    dst.write(float4(float3(core), core), gid);
}

// @fx fx_saber_composite: threshold@0 softness@1 coreIntensity@2 innerRadius@3 glowIntensity@5 coreColor@6:4 glowColor@10:4 turbulence@14 flicker@15
kernel void fx_saber_composite(FX_ARGS)
{
    FX_PROLOGUE
    float thr = PARAM(0), soft = max(PARAM(1), 1e-3f);
    float coreK = PARAM(2), innerR = max(PARAM(3), 0.0f), glowK = PARAM(5);
    float4 coreCol = PARAM4(6), glowCol = PARAM4(10);
    float turb = PARAM(14), flick = PARAM(15);

    float4 outer = src.sample(s, uv);                         // blurred core
    float2 jitter = float2(fx_gnoise3(float3(uv * size / 40.0f, time * 3.0f)),
                           fx_gnoise3(float3(uv * size / 40.0f + 7.3f, time * 3.0f))) * turb;
    float2 uvd = uv + jitter / size;
    float4 base = aux.sample(s, uv);
    float4 oc = aux.sample(s, uvd);
    float core = smoothstep(thr, thr + soft, luma709(oc.rgb)) * oc.a;

    // inner glow: soft gather of the (distorted) core mask
    float inner = 0.0f;
    const int N = 24;
    float rot = fx_ign(float2(gid)) * FX_TAU;
    for (int i = 0; i < N; ++i) {
        float4 t = aux.sample(s, uvd + fx_vogel(i, N, rot) * innerR / size);
        inner += smoothstep(thr, thr + soft, luma709(t.rgb)) * t.a;
    }
    inner /= float(N);

    float fl = 1.0f + flick * fx_gnoise3(float3(time * 18.0f, 3.1f, 0.7f));
    float3 glow = glowCol.rgb * (inner * 1.5f + outer.r * 2.0f) * glowK * fl;
    float3 hot  = mix(glowCol.rgb, coreCol.rgb, saturate(core * coreK)) * core * coreK * fl;
    float3 add  = glow * glowCol.a + hot * coreCol.a;
    float a = saturate(max(add.r, max(add.g, add.b)));
    dst.write(float4(base.rgb + add, max(base.a, a)), gid);
}

// ---- Strobe Light (stateless: depends only on comp time) -----------------------------------------------------------------------
// @fx fx_strobe: strobeColor@0:4 blendWithOriginal@4 strobeDuration@5 strobePeriod@6 probability@7 operator@8 seed@9
kernel void fx_strobe(FX_ARGS)
{
    FX_PROLOGUE
    float4 col = PARAM4(0);
    float blend = saturate(PARAM(4) * 0.01f);
    float duration = PARAM(5), period = max(PARAM(6), 1e-3f);
    float probability = saturate(PARAM(7) * 0.01f);
    int   op = int(PARAM(8));                    // 0 operate on colour, 1 make layer transparent
    uint  seed = uint(PARAM(9));

    float k = floor(time / period);
    float phase = time - k * period;
    bool on = phase < duration;
    on = on && (fx_hash11(uint(k) * 2654435761u + seed) < probability);      // chance that this period flashes
    float4 c = src.sample(s, uv);
    if (!on) { dst.write(c, gid); return; }
    if (op == 1) { dst.write(float4(0.0f), gid); return; }
    float3 flash = col.rgb * c.a;
    dst.write(float4(mix(flash, c.rgb, blend), c.a), gid);
}

// ---- Mosaic -----------------------------------------------------------------------------------------------------------------------
// @fx fx_mosaic: horizontalBlocks@0 verticalBlocks@1 sharpColors@2
kernel void fx_mosaic(FX_ARGS)
{
    FX_PROLOGUE
    float2 blocks = max(float2(PARAM(0), PARAM(1)), 1.0f);
    bool sharp = PARAM(2) > 0.5f;
    float2 cell = floor(uv * blocks);
    float2 centre = (cell + 0.5f) / blocks;
    if (sharp) { dst.write(src.sample(sn, centre), gid); return; }
    float4 acc = float4(0.0f);
    for (int j = 0; j < 4; ++j)
        for (int i = 0; i < 4; ++i)
            acc += src.sample(s, (cell + (float2(i, j) + 0.5f) / 4.0f) / blocks);
    dst.write(acc / 16.0f, gid);
}

// ---- Find Edges (Sobel per channel) ----------------------------------------------------------------------------------------------
// @fx fx_find_edges: invert@0 blendWithOriginal@1
kernel void fx_find_edges(FX_ARGS)
{
    FX_PROLOGUE
    bool invert = PARAM(0) > 0.5f;
    float blend = saturate(PARAM(1) * 0.01f);
    float2 t = 1.0f / srcSize;
    float3 tl = unpremul(src.sample(s, uv + float2(-t.x, -t.y))).rgb, tc = unpremul(src.sample(s, uv + float2(0.0f, -t.y))).rgb, tr = unpremul(src.sample(s, uv + float2(t.x, -t.y))).rgb;
    float3 ml = unpremul(src.sample(s, uv + float2(-t.x, 0.0f))).rgb,                                         mr = unpremul(src.sample(s, uv + float2(t.x, 0.0f))).rgb;
    float3 bl = unpremul(src.sample(s, uv + float2(-t.x,  t.y))).rgb, bc = unpremul(src.sample(s, uv + float2(0.0f,  t.y))).rgb, br = unpremul(src.sample(s, uv + float2(t.x,  t.y))).rgb;
    float3 gx = (tr + 2.0f * mr + br) - (tl + 2.0f * ml + bl);
    float3 gy = (bl + 2.0f * bc + br) - (tl + 2.0f * tc + tr);
    float3 e = saturate(sqrt(gx * gx + gy * gy));
    if (invert) e = 1.0f - e;
    float4 o = src.sample(s, uv);
    float3 rgb = mix(e * o.a, o.rgb, blend);
    dst.write(float4(rgb, o.a), gid);
}

// ---- Threshold ---------------------------------------------------------------------------------------------------------------------------
// @fx fx_threshold: level@0 softness@1
kernel void fx_threshold(FX_ARGS)
{
    FX_PROLOGUE
    float level = PARAM(0), soft = max(PARAM(1), 1e-4f);
    float4 c = unpremul(src.sample(s, uv));
    float v = smoothstep(level - soft, level + soft, luma709(c.rgb));
    dst.write(float4(float3(v) * c.a, c.a), gid);
}

// ---- Cartoon: smoothed + posterised fill, Sobel ink lines ---------------------------------------------------------------------
// @fx fx_cartoon: shadingSteps@0 shadingSmoothness@1 detailRadius@2 detailThreshold@3 edgeThreshold@4 edgeWidth@5 edgeSoftness@6 edgeOpacity@7 edgeColor@8:4
kernel void fx_cartoon(FX_ARGS)
{
    FX_PROLOGUE
    float steps   = max(PARAM(0), 2.0f);
    float smooth_ = clamp(PARAM(1) * 0.01f, 0.02f, 1.0f);
    float radius  = max(PARAM(2), 0.5f);
    float thr     = max(PARAM(3) * 0.01f, 1e-3f);
    float eThr    = PARAM(4), eW = max(PARAM(5), 0.5f), eSoft = max(PARAM(6), 1e-3f);
    float eOp     = saturate(PARAM(7) * 0.01f);
    float4 eCol   = PARAM4(8);

    float4 centre = unpremul(src.sample(s, uv));
    // 3x3 bilateral smoothing at `radius` spacing
    float3 acc = centre.rgb; float wsum = 1.0f;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            if (i == 0 && j == 0) continue;
            float3 c = unpremul(src.sample(s, uv + float2(i, j) * radius / srcSize)).rgb;
            float w = 1.0f - smoothstep(thr * 0.5f, thr, length(c - centre.rgb));
            acc += w * c; wsum += w;
        }
    float3 sm = acc / wsum;

    float l = luma709(sm);
    float lq = (floor(l * steps) + smoothstep(0.5f - smooth_ * 0.5f, 0.5f + smooth_ * 0.5f, fract(l * steps))) / steps;
    float3 fill = sm * (lq / max(l, 1e-3f));

    float2 t = eW / srcSize;
    float a00 = luma709(unpremul(src.sample(s, uv + float2(-t.x, -t.y))).rgb), a10 = luma709(unpremul(src.sample(s, uv + float2(0.0f, -t.y))).rgb), a20 = luma709(unpremul(src.sample(s, uv + float2(t.x, -t.y))).rgb);
    float a01 = luma709(unpremul(src.sample(s, uv + float2(-t.x, 0.0f))).rgb),                                                              a21 = luma709(unpremul(src.sample(s, uv + float2(t.x, 0.0f))).rgb);
    float a02 = luma709(unpremul(src.sample(s, uv + float2(-t.x,  t.y))).rgb), a12 = luma709(unpremul(src.sample(s, uv + float2(0.0f,  t.y))).rgb), a22 = luma709(unpremul(src.sample(s, uv + float2(t.x,  t.y))).rgb);
    float gx = (a20 + 2.0f * a21 + a22) - (a00 + 2.0f * a01 + a02);
    float gy = (a02 + 2.0f * a12 + a22) - (a00 + 2.0f * a10 + a20);
    float edge = smoothstep(eThr, eThr + eSoft, sqrt(gx * gx + gy * gy));
    float3 rgb = mix(fill, eCol.rgb, edge * eOp * eCol.a);
    dst.write(premul(float4(rgb, centre.a)), gid);
}
