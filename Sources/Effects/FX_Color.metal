#include "FXCommon.h"

// =============================================================================
// Colour correction. All kernels un-premultiply, work on straight colour, re-premultiply.
// Levels / luminance weights assume values in 0..1 of the working colour space.
// =============================================================================

// Shadow / midtone / highlight weights from luminance (smooth, sum to 1).
inline float3 fx_tonal_weights(float l) {
    float sh = 1.0f - smoothstep(0.0f, 0.5f, l);
    float hi = smoothstep(0.5f, 1.0f, l);
    return float3(sh, saturate(1.0f - sh - hi), hi);
}

// Keep the original luminance after a colour shift.
inline float3 fx_preserve_luma(float3 shifted, float3 original) {
    float ls = luma709(shifted);
    return ls > 1e-5f ? shifted * (luma709(original) / ls) : shifted;
}

// ---- Curves: aux is a 256x1 RGBA16F LUT (R,G,B = per-channel curves with the master curve
//      already composed in by CurveLUT.build on the CPU) ----------------------------------
// @fx fx_curves: amount@0
kernel void fx_curves(FX_ARGS)
{
    FX_PROLOGUE
    float amount = saturate(PARAM(0) * 0.01f);
    float4 c = unpremul(src.sample(s, uv));
    float3 x = saturate(c.rgb) * (255.0f / 256.0f) + 0.5f / 256.0f;
    float3 mapped = float3(aux.sample(s, float2(x.r, 0.5f)).r,
                           aux.sample(s, float2(x.g, 0.5f)).g,
                           aux.sample(s, float2(x.b, 0.5f)).b);
    c.rgb = mix(c.rgb, mapped, amount);
    dst.write(premul(c), gid);
}

// ---- Levels (master) ----------------------------------------------------------------
// @fx fx_levels: inputBlack@0 inputWhite@1 gamma@2 outputBlack@3 outputWhite@4
kernel void fx_levels(FX_ARGS)
{
    FX_PROLOGUE
    float inB = PARAM(0), inW = PARAM(1), gamma = max(PARAM(2), 0.01f);
    float outB = PARAM(3), outW = PARAM(4);
    float4 c = unpremul(src.sample(s, uv));
    float3 v = saturate((c.rgb - inB) / max(inW - inB, 1e-5f));
    v = pow(v, float3(1.0f / gamma));
    c.rgb = mix(float3(outB), float3(outW), v);
    dst.write(premul(c), gid);
}

// ---- Hue / Saturation (HSL based, with Colorize) ----------------------------------------------
// @fx fx_hue_sat: hue@0 saturation@1 lightness@2 colorize@3 colorizeHue@4 colorizeSaturation@5 colorizeLightness@6
kernel void fx_hue_sat(FX_ARGS)
{
    FX_PROLOGUE
    float hue = PARAM(0) / 360.0f;
    float sat = PARAM(1) * 0.01f;
    float lit = PARAM(2) * 0.01f;
    bool  colorize = PARAM(3) > 0.5f;
    float cHue = PARAM(4) / 360.0f, cSat = PARAM(5) * 0.01f, cLit = PARAM(6) * 0.01f;

    float4 c = unpremul(src.sample(s, uv));
    float3 hsl = rgb2hsl(saturate(c.rgb));
    if (colorize) {
        hsl.x = cHue;
        hsl.y = cSat;
        hsl.z = cLit >= 0.0f ? mix(hsl.z, 1.0f, cLit) : mix(hsl.z, 0.0f, -cLit);
    } else {
        hsl.x += hue;
        hsl.y = saturate(hsl.y * (1.0f + sat));
        hsl.z = lit >= 0.0f ? mix(hsl.z, 1.0f, lit) : mix(hsl.z, 0.0f, -lit);
    }
    c.rgb = hsl2rgb(hsl);
    dst.write(premul(c), gid);
}

// ---- Lumetri-style basic correction + three colour wheels ------------------------------------
// Wheel colours: neutral = any grey; hue/saturation of the swatch pushes that tonal range,
// alpha of the swatch is the strength.
// @fx fx_lumetri: exposure@0 contrast@1 highlights@2 shadows@3 whites@4 blacks@5 temperature@6 tint@7 saturation@8 vibrance@9 shadowWheel@10:4 midWheel@14:4 highlightWheel@18:4
kernel void fx_lumetri(FX_ARGS)
{
    FX_PROLOGUE
    float exposure = PARAM(0);
    float contrast = PARAM(1) * 0.01f;
    float highlights = PARAM(2) * 0.01f, shadows = PARAM(3) * 0.01f;
    float whites = PARAM(4) * 0.01f, blacks = PARAM(5) * 0.01f;
    float temp = PARAM(6) * 0.01f, tintv = PARAM(7) * 0.01f;
    float satur = PARAM(8) * 0.01f, vib = PARAM(9) * 0.01f;
    float4 wS = PARAM4(10), wM = PARAM4(14), wH = PARAM4(18);

    float4 c = unpremul(src.sample(s, uv));
    float3 rgb = c.rgb * exp2(exposure);

    rgb *= float3(1.0f + 0.25f * temp, 1.0f - 0.15f * tintv, 1.0f - 0.25f * temp);   // white balance
    rgb = (rgb - 0.5f) * (1.0f + contrast) + 0.5f;

    float l = luma709(rgb);
    rgb *= 1.0f + highlights * 0.5f * smoothstep(0.5f, 1.0f, l);
    rgb += shadows * 0.25f * (1.0f - smoothstep(0.0f, 0.5f, l));
    rgb *= 1.0f + whites * 0.25f * smoothstep(0.75f, 1.0f, l);
    rgb += blacks * 0.10f * (1.0f - smoothstep(0.0f, 0.25f, l));

    l = luma709(rgb);
    float3 w = fx_tonal_weights(clamp(l, 0.0f, 1.0f));
    rgb += (wS.rgb - luma709(wS.rgb)) * wS.a * w.x * 0.5f;
    rgb += (wM.rgb - luma709(wM.rgb)) * wM.a * w.y * 0.5f;
    rgb += (wH.rgb - luma709(wH.rgb)) * wH.a * w.z * 0.5f;

    float g = luma709(rgb);
    float chroma = max(rgb.r, max(rgb.g, rgb.b)) - min(rgb.r, min(rgb.g, rgb.b));
    rgb = mix(float3(g), rgb, satur * (1.0f + vib * (1.0f - saturate(chroma))));
    c.rgb = max(rgb, 0.0f);
    dst.write(premul(c), gid);
}

// ---- Look LUT (.cube style 3D LUT bound to texture 2) -----------------------------------------
// Runs after fx_lumetri in the same effect, so its parameter lives after Lumetri's own slots (0...21).
// @fx fx_lut3d: lookIntensity@22
kernel void fx_lut3d(texture2d<float, access::sample> src [[texture(0)]],
                     texture2d<float, access::write>  dst [[texture(1)]],
                     texture3d<float, access::sample> lut [[texture(2)]],
                     constant float* P [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    FX_PROLOGUE
    float k = saturate(PARAM(22) * 0.01f);
    float4 c = unpremul(src.sample(s, uv));
    float n = float(lut.get_width());
    float3 coord = saturate(c.rgb) * ((n - 1.0f) / n) + 0.5f / n;
    float3 graded = lut.sample(s, coord).rgb;
    c.rgb = mix(c.rgb, graded, k);
    dst.write(premul(c), gid);
}

// ---- Photo Filter -------------------------------------------------------------------------------------
// @fx fx_photo_filter: filterColor@0:4 density@4 preserveLuminosity@5
kernel void fx_photo_filter(FX_ARGS)
{
    FX_PROLOGUE
    float3 f = PARAM4(0).rgb;
    float density = saturate(PARAM(4) * 0.01f);
    bool keepLuma = PARAM(5) > 0.5f;
    float4 c = unpremul(src.sample(s, uv));
    float3 tinted = c.rgb * (f / max(max(f.r, max(f.g, f.b)), 1e-4f));
    if (keepLuma) tinted = fx_preserve_luma(tinted, c.rgb);
    c.rgb = mix(c.rgb, tinted, density);
    dst.write(premul(c), gid);
}

// ---- Color Balance (shadows / midtones / highlights, per channel -100..100) -------------------
// @fx fx_color_balance: shadowR@0 shadowG@1 shadowB@2 midR@3 midG@4 midB@5 highR@6 highG@7 highB@8 preserveLuminosity@9
kernel void fx_color_balance(FX_ARGS)
{
    FX_PROLOGUE
    float3 sh = PARAM3(0), mi = PARAM3(3), hi = PARAM3(6);
    bool keepLuma = PARAM(9) > 0.5f;
    float4 c = unpremul(src.sample(s, uv));
    float3 w = fx_tonal_weights(clamp(luma709(c.rgb), 0.0f, 1.0f));
    float3 shifted = c.rgb + (sh * w.x + mi * w.y + hi * w.z) * 0.01f * 0.3f;
    shifted = max(shifted, 0.0f);
    c.rgb = keepLuma ? fx_preserve_luma(shifted, c.rgb) : shifted;
    dst.write(premul(c), gid);
}

// ---- Tint (map black / white) ----------------------------------------------------------------------------
// @fx fx_tint: mapBlackTo@0:4 mapWhiteTo@4:4 amount@8
kernel void fx_tint(FX_ARGS)
{
    FX_PROLOGUE
    float3 blackTo = PARAM4(0).rgb, whiteTo = PARAM4(4).rgb;
    float amount = saturate(PARAM(8) * 0.01f);
    float4 c = unpremul(src.sample(s, uv));
    float3 mapped = mix(blackTo, whiteTo, saturate(luma709(c.rgb)));
    c.rgb = mix(c.rgb, mapped, amount);
    dst.write(premul(c), gid);
}
