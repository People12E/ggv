#include "FXCommon.h"

// =============================================================================
// Generate. Every generator composites over the incoming layer with fx_composite:
// blendMode 0 = generator only, 1 normal, 2 multiply, 3 screen, 4 overlay, 5 add, 6 difference.
// =============================================================================

// ---- Grid -----------------------------------------------------------------------------------
// @fx fx_grid: anchor@0:2 cellWidth@2 cellHeight@3 border@4 feather@5 color@6:4 blendMode@10
kernel void fx_grid(FX_ARGS)
{
    FX_PROLOGUE
    float2 anchor = PARAM2(0);                  // pixels from top-left
    float2 cell   = max(float2(PARAM(2), PARAM(3)), 2.0f);
    float border  = PARAM(4), feather = PARAM(5);
    float4 col    = PARAM4(6);
    int mode      = int(PARAM(10));

    float2 p = uv * size - anchor;
    float2 m = float2(fx_fmod(p.x, cell.x), fx_fmod(p.y, cell.y));
    float2 dist = min(m, cell - m);                         // distance to nearest line per axis
    float hw = border * 0.5f;
    float2 cov2 = 1.0f - smoothstep(max(hw - 0.5f, 0.0f), hw + 0.5f + feather, dist);
    float cov = max(cov2.x, cov2.y);
    dst.write(fx_composite(src.sample(s, uv), col.rgb, col.a * cov, mode), gid);
}

// ---- 4-Color Gradient (inverse-distance weighting) --------------------------------------------
// @fx fx_gradient4: point1@0:2 point2@2:2 point3@4:2 point4@6:2 color1@8:4 color2@12:4 color3@16:4 color4@20:4 blend@24 jitter@25 opacity@26 blendMode@27
kernel void fx_gradient4(FX_ARGS)
{
    FX_PROLOGUE
    float aspect = size.x / size.y;
    float2 pts[4] = { PARAM2(0), PARAM2(2), PARAM2(4), PARAM2(6) };
    float4 cols[4] = { PARAM4(8), PARAM4(12), PARAM4(16), PARAM4(20) };
    float blend   = saturate(PARAM(24) * 0.01f);
    float jitter  = PARAM(25) * 0.01f;
    float opacity = saturate(PARAM(26) * 0.01f);
    int mode      = int(PARAM(27));

    float expo = mix(4.0f, 1.0f, blend);                 // larger blend = softer, wider falloff
    float4 acc = float4(0.0f);
    float wsum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        float2 d = (uv - pts[i]) * float2(aspect, 1.0f);
        float w = 1.0f / (pow(length(d), expo) + 1e-6f);
        acc += w * cols[i];
        wsum += w;
    }
    float4 g = acc / wsum;
    g.rgb += (fx_ign(float2(gid)) - 0.5f) * jitter / 255.0f * 4.0f;     // dither to hide banding
    dst.write(fx_composite(src.sample(s, uv), g.rgb, g.a * opacity, mode), gid);
}

// ---- Fractal Noise ------------------------------------------------------------------------------------
// @fx fx_fractal_noise: noiseType@0 contrast@1 brightness@2 scale@3 complexity@4 subInfluence@5 subScale@6 evolution@7 evolutionSpeed@8 offset@9:2 invert@11 opacity@12 blendMode@13
kernel void fx_fractal_noise(FX_ARGS)
{
    FX_PROLOGUE
    int   type     = int(PARAM(0));               // 0 basic, 1 turbulent, 2 billow, 3 ridged
    float contrast = PARAM(1) * 0.01f;
    float bright   = PARAM(2) * 0.01f;
    float scale    = max(PARAM(3), 1.0f);
    int   octaves  = clamp(int(PARAM(4)), 1, 8);
    float subInf   = PARAM(5) * 0.01f;
    float subScale = max(PARAM(6) * 0.01f, 0.1f);
    float evo      = PARAM(7) / 360.0f + time * PARAM(8) / 360.0f;
    float2 offset  = PARAM2(9);
    bool  invert   = PARAM(11) > 0.5f;
    float opacity  = saturate(PARAM(12) * 0.01f);
    int   mode     = int(PARAM(13));

    float2 p = (uv * size - offset) / scale;
    float sum = 0.0f, norm = 0.0f, amp = 1.0f, freq = 1.0f;
    for (int o = 0; o < octaves; ++o) {
        float n = fx_gnoise3(float3(p * freq, evo * 4.0f + float(o) * 13.7f));
        if (type == 1)      n = fabs(n) * 2.0f - 1.0f;
        else if (type == 2) n = 1.0f - 2.0f * fabs(n);
        else if (type == 3) { float r = 1.0f - fabs(n); n = r * r * 2.0f - 1.0f; }
        sum += n * amp;
        norm += amp;
        amp *= subInf;
        freq /= subScale;
    }
    float v = sum / max(norm, 1e-5f) * 0.5f + 0.5f;
    v = (v - 0.5f) * contrast + 0.5f + bright * 0.5f;
    v = saturate(v);
    if (invert) v = 1.0f - v;
    dst.write(fx_composite(src.sample(s, uv), float3(v), opacity, mode), gid);
}

// ---- Stroke along a polyline: buffer(1) = float3 (x, y normalized, cumulative arc length 0..1) ----
// Up to 256 points. Cost is O(points) per pixel, so restrict the dispatch to the path's bounding box for big paths.
// @fx fx_stroke: pointCount@0 brushSize@1 brushHardness@2 opacity@3 start@4 end@5 color@6:4 paintStyle@10
kernel void fx_stroke(FX_ARGS, constant float3* pts [[buffer(1)]])
{
    FX_PROLOGUE
    int   count    = clamp(int(PARAM(0)), 0, 256);
    float radius   = max(PARAM(1) * 0.5f, 0.25f);
    float hardness = saturate(PARAM(2) * 0.01f);
    float opacity  = saturate(PARAM(3) * 0.01f);
    float t0 = PARAM(4) * 0.01f, t1 = PARAM(5) * 0.01f;
    float4 col = PARAM4(6);
    int   style = int(PARAM(10));                 // 0 over original, 1 on transparent, 2 reveal original

    float2 p = uv * size;
    float best = 1e9f;
    for (int i = 0; i + 1 < count; ++i) {
        float3 a = pts[i], b = pts[i + 1];
        float2 pa = a.xy * size, pb = b.xy * size;
        float2 ab = pb - pa;
        float tt = saturate(dot(p - pa, ab) / max(dot(ab, ab), 1e-6f));
        float cum = mix(a.z, b.z, tt);
        if (cum < t0 || cum > t1) continue;
        best = min(best, length(p - (pa + ab * tt)));
    }
    float inner = radius * hardness;
    float cov = (count > 1) ? 1.0f - smoothstep(max(inner - 0.5f, 0.0f), radius + 0.5f, best) : 0.0f;
    float4 base = src.sample(s, uv);
    float a = col.a * opacity * cov;
    if (style == 1)      dst.write(float4(col.rgb * a, a), gid);
    else if (style == 2) dst.write(base * cov * opacity, gid);
    else                 dst.write(fx_composite(base, col.rgb, a, 1), gid);
}

// ---- Audio Spectrum: buffer(1) = band magnitudes 0..1 (AudioSpectrumAnalyzer) --------------------
// Path is the line start->end. Style 0 bars, 1 smooth line, 2 radial bars around start with radius |end-start|.
// @fx fx_audio_spectrum: bandCount@0 style@1 start@2:2 end@4:2 thickness@6 maxHeight@7 color@8:4 paintStyle@12
kernel void fx_audio_spectrum(FX_ARGS, constant float* bins [[buffer(1)]])
{
    FX_PROLOGUE
    int   bands  = clamp(int(PARAM(0)), 1, 256);
    int   style  = int(PARAM(1));
    float2 A = PARAM2(2) * size, B = PARAM2(4) * size;
    float thick  = max(PARAM(6), 1.0f);
    float maxH   = PARAM(7);
    float4 col   = PARAM4(8);
    int   paint  = int(PARAM(12));                // 0 over original, 1 on transparent

    float2 p = uv * size;
    float cov = 0.0f;
    if (style == 2) {
        float2 d = p - A;
        float R0 = length(B - A);
        float ang = fract(atan2(d.x, -d.y) / FX_TAU);
        int bi = min(int(ang * float(bands)), bands - 1);
        float u = fract(ang * float(bands));
        float h = bins[bi] * maxH;
        float r = length(d) - R0;
        float barCov = smoothstep(0.0f, 0.5f, u) * (1.0f - smoothstep(0.5f, 1.0f, u)) * 2.0f;   // gap between bars
        cov = barCov * step(0.0f, r) * (1.0f - smoothstep(h - 0.5f, h + 0.5f, r));
    } else {
        float2 axis = B - A;
        float len = max(length(axis), 1.0f);
        float2 a = axis / len;
        float2 n = float2(a.y, -a.x);                       // "up" relative to the line
        float2 d = p - A;
        float u = dot(d, a), v = dot(d, n);
        if (u >= 0.0f && u <= len) {
            float f = u / len * float(bands);
            if (style == 0) {
                int bi = min(int(f), bands - 1);
                float inBar = fract(f);
                float h = bins[bi] * maxH;
                float bar = smoothstep(0.1f, 0.2f, inBar) * (1.0f - smoothstep(0.8f, 0.9f, inBar));
                cov = bar * step(0.0f, v) * (1.0f - smoothstep(h - 0.5f, h + 0.5f, v));
            } else {
                float fc = max(f - 0.5f, 0.0f);
                int i0 = min(int(fc), bands - 1), i1 = min(i0 + 1, bands - 1);
                float h = mix(bins[i0], bins[i1], fract(fc)) * maxH;
                cov = 1.0f - smoothstep(thick * 0.5f - 0.5f, thick * 0.5f + 0.5f, fabs(v - h));
            }
        }
    }
    float4 base = src.sample(s, uv);
    float a = col.a * cov;
    dst.write(paint == 1 ? float4(col.rgb * a, a) : fx_composite(base, col.rgb, a, 1), gid);
}

// ---- Cell Pattern (Worley) --------------------------------------------------------------------------------------
// @fx fx_cell_pattern: pattern@0 cellSize@1 contrast@2 offset@3:2 evolution@5 evolutionSpeed@6 invert@7 opacity@8 blendMode@9
kernel void fx_cell_pattern(FX_ARGS)
{
    FX_PROLOGUE
    int   pattern  = int(PARAM(0));         // 0 bubbles, 1 crystals, 2 plates, 3 pillow
    float cellSize = max(PARAM(1), 4.0f);
    float contrast = PARAM(2) * 0.01f;
    float2 offset  = PARAM2(3);
    float evo      = PARAM(5) / 360.0f + time * PARAM(6) / 360.0f;
    bool  invert   = PARAM(7) > 0.5f;
    float opacity  = saturate(PARAM(8) * 0.01f);
    int   mode     = int(PARAM(9));

    float2 p = (uv * size - offset) / cellSize;
    int2 base = int2(floor(p));
    float f1 = 1e9f, f2 = 1e9f;
    int2 id1 = base;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            int2 c = base + int2(i, j);
            float2 h = fx_hash22(c);
            float2 pos = float2(c) + 0.5f + 0.45f * float2(sin(FX_TAU * (evo + h.x)), cos(FX_TAU * (evo + h.y)));
            float d = length(p - pos);
            if (d < f1) { f2 = f1; f1 = d; id1 = c; }
            else if (d < f2) { f2 = d; }
        }
    }
    float v;
    if (pattern == 0)      { float r = 0.25f + 0.2f * fx_hash21(id1); v = 1.0f - smoothstep(r - 0.03f, r, f1); }
    else if (pattern == 1) { v = fx_hash21(id1 + int2(17, 31)); }
    else if (pattern == 2) { v = smoothstep(0.0f, 0.18f, f2 - f1); }
    else                   { v = saturate(1.0f - f1 * 1.4f); v = sqrt(max(v, 0.0f)); }
    v = saturate((v - 0.5f) * contrast + 0.5f);
    if (invert) v = 1.0f - v;
    dst.write(fx_composite(src.sample(s, uv), float3(v), opacity, mode), gid);
}

// ---- Checkerboard -----------------------------------------------------------------------------------------------------
// @fx fx_checkerboard: anchor@0:2 cellWidth@2 cellHeight@3 feather@4 color@5:4 blendMode@9
kernel void fx_checkerboard(FX_ARGS)
{
    FX_PROLOGUE
    float2 anchor = PARAM2(0);
    float2 cell = max(float2(PARAM(2), PARAM(3)), 2.0f);
    float feather = PARAM(4);
    float4 col = PARAM4(5);
    int mode = int(PARAM(9));

    float2 q = (uv * size - anchor) / cell;
    float2 fw = max(float2(feather) / cell, float2(1.0f) / cell);       // at least ~1 px of anti-aliasing
    float2 sq = clamp(sin(FX_PI * q) / (FX_PI * fw), -1.0f, 1.0f) * 0.5f + 0.5f;
    float cov = sq.x * sq.y + (1.0f - sq.x) * (1.0f - sq.y);
    dst.write(fx_composite(src.sample(s, uv), col.rgb, col.a * cov, mode), gid);
}
