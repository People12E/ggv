#include <metal_stdlib>
#include "BPShaderTypes.h"
using namespace metal;

constexpr sampler linClamp(coord::normalized, address::clamp_to_edge, filter::linear);

static inline float2 pixelUV(uint2 gid, texture2d<float, access::write> dst) {
    return (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
}

// ---------------------------------------------------------------------------
// Gaussian blur: separable. Dispatch horizontal (dir = (1,0)) then vertical (0,1).
// Weights are precomputed on the CPU (normalized, center first, `radius`+1 entries).
// Fast Box Blur reuses this kernel with uniform weights.
// ---------------------------------------------------------------------------

kernel void blurSeparable(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant SeparableParams& p [[buffer(0)]],
                          constant float* weights     [[buffer(1)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float2 texel = p.direction / float2(src.get_width(), src.get_height());
    float4 sum = src.sample(linClamp, uv) * weights[0];
    for (int i = 1; i <= p.radius; ++i) {
        sum += src.sample(linClamp, uv + texel * i) * weights[i];
        sum += src.sample(linClamp, uv - texel * i) * weights[i];
    }
    dst.write(sum, gid);
}

// ---------------------------------------------------------------------------
// Directional (motion) blur: one-pass line integral.
// ---------------------------------------------------------------------------

kernel void blurDirectional(texture2d<float, access::sample> src [[texture(0)]],
                            texture2d<float, access::write>  dst [[texture(1)]],
                            constant DirectionalParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float2 size = float2(src.get_width(), src.get_height());
    float2 dir = float2(cos(p.angle), sin(p.angle)) * p.length / size;
    float4 sum = float4(0);
    int n = max(p.samples, 1);
    for (int i = 0; i < n; ++i) {
        float t = (float(i) / max(n - 1, 1)) - 0.5;
        sum += src.sample(linClamp, uv + dir * t);
    }
    dst.write(sum / n, gid);
}

// ---------------------------------------------------------------------------
// Radial blur: spin or zoom around a center.
// ---------------------------------------------------------------------------

kernel void blurRadial(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write>  dst [[texture(1)]],
                       constant RadialParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float2 d = uv - p.center;
    float4 sum = float4(0);
    int n = max(p.samples, 1);
    for (int i = 0; i < n; ++i) {
        float t = (float(i) / max(n - 1, 1)) - 0.5;
        float2 q;
        if (p.zoom != 0) {
            q = p.center + d * (1.0 + t * p.amount);
        } else {
            float a = t * p.amount;                  // radians
            float c = cos(a), s = sin(a);
            q = p.center + float2(c * d.x - s * d.y, s * d.x + c * d.y);
        }
        sum += src.sample(linClamp, q);
    }
    dst.write(sum / n, gid);
}

// ---------------------------------------------------------------------------
// Smart blur: edge-preserving (bilateral) blur.
// ---------------------------------------------------------------------------

kernel void blurSmart(texture2d<float, access::sample> src [[texture(0)]],
                      texture2d<float, access::write>  dst [[texture(1)]],
                      constant SmartParams& p [[buffer(0)]],
                      uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float4 c = src.sample(linClamp, uv);
    float4 sum = float4(0); float wsum = 0;
    float inv2s = 1.0 / (2.0 * p.spatialSigma * p.spatialSigma + 1e-6);
    float invT = 1.0 / (p.threshold * p.threshold + 1e-6);
    for (int y = -p.radius; y <= p.radius; ++y) {
        for (int x = -p.radius; x <= p.radius; ++x) {
            float4 s4 = src.sample(linClamp, uv + float2(x, y) * texel);
            float3 dc = s4.rgb - c.rgb;
            float w = exp(-float(x * x + y * y) * inv2s) * exp(-dot(dc, dc) * invT);
            sum += s4 * w; wsum += w;
        }
    }
    dst.write(sum / max(wsum, 1e-6), gid);
}

// ---------------------------------------------------------------------------
// Camera lens blur: gather bokeh with per-pixel radius from a depth map.
// Disc kernel with golden-angle sampling; iris shape blend via `blades`.
// ---------------------------------------------------------------------------

kernel void blurLens(texture2d<float, access::sample> src   [[texture(0)]],
                     texture2d<float, access::sample> depth [[texture(1)]],   // 0 near .. 1 far
                     texture2d<float, access::write>  dst   [[texture(2)]],
                     constant LensParams& p [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float d = depth.sample(linClamp, uv).r;
    float coc = clamp(abs(d - p.focalDepth) / max(p.focalRange, 1e-4), 0.0, 1.0) * p.maxRadius;

    float4 sum = src.sample(linClamp, uv); float wsum = 1.0;
    const float golden = 2.39996323;
    for (int i = 1; i < p.samples; ++i) {
        float r = sqrt(float(i) / p.samples) * coc;
        float a = float(i) * golden;
        float2 o = float2(cos(a), sin(a));
        if (p.blades >= 3) {                          // polygonal iris: pull samples in toward flat edges
            float seg = 6.28318530718 / p.blades;
            float am = fmod(a, seg) - seg * 0.5;
            o *= cos(seg * 0.5) / max(cos(am), 1e-3);
            o *= 0.9;
        }
        float4 s4 = src.sample(linClamp, uv + o * r * texel);
        float sd = depth.sample(linClamp, uv + o * r * texel).r;
        float w = (sd >= d - 0.02) ? 1.0 : 0.0;     // don't leak sharp foreground into background blur
        w *= 1.0 + dot(s4.rgb, float3(0.333)) * p.brightBoost;
        sum += s4 * w; wsum += w;
    }
    dst.write(sum / wsum, gid);
}

// ---------------------------------------------------------------------------
// Unsharp mask: dst = src + amount * (src - blurred), thresholded.
// Feed the output of a Gaussian pass as `blurred`.
// ---------------------------------------------------------------------------

kernel void unsharpMask(texture2d<float, access::sample> src     [[texture(0)]],
                        texture2d<float, access::sample> blurred [[texture(1)]],
                        texture2d<float, access::write>  dst     [[texture(2)]],
                        constant UnsharpParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = pixelUV(gid, dst);
    float4 a = src.sample(linClamp, uv);
    float3 diff = a.rgb - blurred.sample(linClamp, uv).rgb;
    float3 gate = step(float3(p.threshold), abs(diff));
    dst.write(float4(a.rgb + diff * p.amount * gate, a.a), gid);
}
