#include "FXCommon.h"

// =============================================================================
// Distort. All are backward mappings: for each output pixel decide where to read the source.
// =============================================================================

inline float fx_channel(float4 c, int ch) {
    switch (ch) {
        case 0: return c.r;
        case 1: return c.g;
        case 2: return c.b;
        case 3: return c.a;
        default: return luma709(c.rgb);
    }
}

// ---- Displacement Map (map layer in aux) -------------------------------------------------------------
// @fx fx_displace: mapChannelX@0 mapChannelY@1 maxHorizontal@2 maxVertical@3 mapMode@4 edgeBehavior@5
kernel void fx_displace(FX_ARGS)
{
    FX_PROLOGUE
    int chX = int(PARAM(0)), chY = int(PARAM(1));           // 0 R, 1 G, 2 B, 3 A, 4 luminance
    float2 maxPx = float2(PARAM(2), PARAM(3));              // pixels
    int mapMode = int(PARAM(4));                            // 0 stretch, 1 centre, 2 tile
    int edge = int(PARAM(5));                               // 0 clamp, 1 wrap

    float2 auxSize = float2(aux.get_width(), aux.get_height());
    float2 muv = uv;
    if (mapMode == 1) muv = (uv - 0.5f) * size / auxSize + 0.5f;
    else if (mapMode == 2) muv = fract(uv * size / auxSize);

    float4 m = unpremul(aux.sample(s, muv));
    float2 disp = (float2(fx_channel(m, chX), fx_channel(m, chY)) - 0.5f) * 2.0f * maxPx;
    float2 p = uv + disp / size;
    dst.write(edge == 1 ? src.sample(sr, p) : src.sample(s, p), gid);
}

// ---- Wave Warp -------------------------------------------------------------------------------------------------
// @fx fx_wave_warp: waveType@0 waveHeight@1 waveWidth@2 direction@3 speed@4 phase@5
kernel void fx_wave_warp(FX_ARGS)
{
    FX_PROLOGUE
    int   type   = int(PARAM(0));    // 0 sine, 1 square, 2 triangle, 3 sawtooth, 4 noise
    float height = PARAM(1);
    float width  = max(PARAM(2), 1.0f);
    float speed  = PARAM(4);
    float phase  = PARAM(5) / 360.0f;

    float2 p = uv * size;
    float2 d = fx_ae_dir(PARAM(3));                // direction of travel (0 deg = up, clockwise)
    float2 n = float2(-d.y, d.x);                  // displacement axis
    float ph = dot(p, d) / width + phase + time * speed;

    float w;
    if (type == 0)      w = sin(FX_TAU * ph);
    else if (type == 1) w = clamp(sin(FX_TAU * ph) * 8.0f, -1.0f, 1.0f);
    else if (type == 2) w = 2.0f * fabs(2.0f * (ph - floor(ph + 0.5f))) - 1.0f;
    else if (type == 3) w = 2.0f * fract(ph) - 1.0f;
    else                w = fx_gnoise3(float3(ph * 2.0f, 0.37f, 0.11f)) * 1.6f;

    float2 q = (p + n * (w * height)) / size;
    dst.write(src.sample(s, q), gid);
}

// ---- Polar Coordinates ------------------------------------------------------------------------------------------------
// @fx fx_polar: conversion@0 interpolation@1
kernel void fx_polar(FX_ARGS)
{
    FX_PROLOGUE
    int   conv = int(PARAM(0));                 // 0 rect to polar, 1 polar to rect
    float k    = saturate(PARAM(1) * 0.01f);
    float aspect = size.x / size.y;
    float2 mapped;
    if (conv == 0) {
        float2 v = uv - 0.5f;
        float2 vv = float2(v.x * aspect, v.y);
        float theta = atan2(vv.x, -vv.y);                    // 0 at top, clockwise
        mapped = float2(theta / FX_TAU + 0.5f, length(vv) * 2.0f);
    } else {
        float theta = (uv.x - 0.5f) * FX_TAU;
        float rho = uv.y * 0.5f;
        float2 vv = rho * float2(sin(theta), -cos(theta));
        mapped = float2(vv.x / aspect, vv.y) + 0.5f;
    }
    float2 p = mix(uv, mapped, k);
    float4 c = (conv == 0) ? src.sample(sr, float2(p.x, clamp(p.y, 0.0f, 1.0f))) : src.sample(sz, p);
    dst.write(c, gid);
}

// ---- Corner Pin: 3x3 homography (row-major) taking OUTPUT uv to SOURCE uv, built on the CPU ----------------
// @fx fx_corner_pin: homography@0:9
kernel void fx_corner_pin(FX_ARGS)
{
    FX_PROLOGUE
    float3 r0 = PARAM3(0), r1 = PARAM3(3), r2 = PARAM3(6);
    float3 q = float3(uv, 1.0f);
    float3 h = float3(dot(r0, q), dot(r1, q), dot(r2, q));
    if (fabs(h.z) < 1e-6f) { dst.write(float4(0.0f), gid); return; }
    float2 p = h.xy / h.z;
    dst.write(src.sample(sz, p), gid);
}

// ---- Mesh Warp: control-grid offsets (normalized, row-major) in buffer(1), Catmull-Rom interpolated ----------
// Approximation: the interpolated offset field is used directly as the backward map.
inline float fx_cr(float a, float b, float c, float d, float t) {
    return 0.5f * ((2.0f * b) + (-a + c) * t + (2.0f * a - 5.0f * b + 4.0f * c - d) * t * t + (-a + 3.0f * b - 3.0f * c + d) * t * t * t);
}
inline float2 fx_grid_at(constant float2* g, int cols, int rows, int x, int y) {
    return g[clamp(y, 0, rows - 1) * cols + clamp(x, 0, cols - 1)];
}

// @fx fx_mesh_warp: columns@0 rows@1
kernel void fx_mesh_warp(FX_ARGS, constant float2* grid [[buffer(1)]])
{
    FX_PROLOGUE
    int cols = max(int(PARAM(0)), 2), rows = max(int(PARAM(1)), 2);
    float gx = uv.x * float(cols - 1), gy = uv.y * float(rows - 1);
    int ix = min(int(floor(gx)), cols - 2), iy = min(int(floor(gy)), rows - 2);
    float tx = gx - float(ix), ty = gy - float(iy);
    float2 rowv[4];
    for (int j = 0; j < 4; ++j) {
        float2 a = fx_grid_at(grid, cols, rows, ix - 1, iy - 1 + j);
        float2 b = fx_grid_at(grid, cols, rows, ix,     iy - 1 + j);
        float2 c = fx_grid_at(grid, cols, rows, ix + 1, iy - 1 + j);
        float2 d = fx_grid_at(grid, cols, rows, ix + 2, iy - 1 + j);
        rowv[j] = float2(fx_cr(a.x, b.x, c.x, d.x, tx), fx_cr(a.y, b.y, c.y, d.y, tx));
    }
    float2 off = float2(fx_cr(rowv[0].x, rowv[1].x, rowv[2].x, rowv[3].x, ty),
                        fx_cr(rowv[0].y, rowv[1].y, rowv[2].y, rowv[3].y, ty));
    dst.write(src.sample(s, uv - off), gid);
}

// ---- Offset (shift with optional wrap) ------------------------------------------------------------------------------------
// @fx fx_offset: shiftX@0 shiftY@1 wrapAround@2 blendWithOriginal@3
kernel void fx_offset(FX_ARGS)
{
    FX_PROLOGUE
    float2 shift = float2(PARAM(0), PARAM(1)) / size;
    bool wrap = PARAM(2) > 0.5f;
    float k = saturate(PARAM(3) * 0.01f);
    float4 shifted = wrap ? src.sample(sr, uv - shift) : src.sample(sz, uv - shift);
    dst.write(mix(shifted, src.sample(s, uv), k), gid);
}

// ---- Spherize -----------------------------------------------------------------------------------------------------------------------------
// @fx fx_spherize: radius@0 amount@1 center@2:2
kernel void fx_spherize(FX_ARGS)
{
    FX_PROLOGUE
    float R = max(PARAM(0), 1.0f);
    float amt = clamp(PARAM(1) * 0.01f, -1.0f, 1.0f);
    float2 c = PARAM2(2);
    float2 d = (uv - c) * size;
    float r = length(d);
    float2 p = uv;
    if (r < R) {
        float n = r / R;
        float target = (amt >= 0.0f) ? mix(n, asin(min(n, 1.0f)) * (2.0f / FX_PI), amt)
                                     : mix(n, sin(n * FX_PI * 0.5f), -amt);
        p = c + (r > 1e-4f ? d / r : float2(0.0f)) * (target * R) / size;
    }
    dst.write(src.sample(s, p), gid);
}
