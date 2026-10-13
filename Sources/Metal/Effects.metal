#include <metal_stdlib>
using namespace metal;

// ============================================================================
// 1. Displacement map + radial chromatic aberration (single fused pass)
// ============================================================================
struct DisplaceParams {
    float2 strength;      // pixels, x/y
    float  caAmount;      // radial chromatic aberration, in normalized radius units
    float2 center;        // normalized CA center
    uint   mapChannelX;   // 0=R 1=G 2=B 3=A 4=luma
    uint   mapChannelY;
};

static inline float pickChannel(float4 c, uint ch) {
    switch (ch) {
        case 0: return c.r;
        case 1: return c.g;
        case 2: return c.b;
        case 3: return c.a;
        default: return dot(c.rgb, float3(0.2126, 0.7152, 0.0722));
    }
}

kernel void displaceChromatic(texture2d<float, access::sample> src  [[texture(0)]],
                              texture2d<float, access::sample> map  [[texture(1)]],
                              texture2d<float, access::write>  dst  [[texture(2)]],
                              constant DisplaceParams& p            [[buffer(0)]],
                              uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);

    float2 size = float2(dst.get_width(), dst.get_height());
    float2 uv = (float2(gid) + 0.5) / size;

    float4 m = map.sample(s, uv);
    float2 disp = float2(pickChannel(m, p.mapChannelX), pickChannel(m, p.mapChannelY)) - 0.5;
    float2 uvD = uv + disp * 2.0 * p.strength / size;

    // Radial CA: R pushed outward, B inward, G unchanged.
    float2 dir = uvD - p.center;
    float r2 = dot(dir, dir);
    float2 off = dir * p.caAmount * r2;

    float cr = src.sample(s, uvD + off).r;
    float4 g = src.sample(s, uvD);
    float cb = src.sample(s, uvD - off).b;
    dst.write(float4(cr, g.g, cb, g.a), gid);
}

// ============================================================================
// 2. Multi-pass Kawase glow
//    Pass order: threshold -> N x kawaseBlur (iteration offsets) -> composite
// ============================================================================
struct GlowParams {
    float threshold;
    float knee;
    float intensity;
    float3 tint;
};

kernel void glowThreshold(texture2d<float, access::sample> src [[texture(0)]],
                          texture2d<float, access::write>  dst [[texture(1)]],
                          constant GlowParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float4 c = src.sample(s, uv);
    float l = max(c.r, max(c.g, c.b));
    float soft = clamp(l - p.threshold + p.knee, 0.0, 2.0 * p.knee);
    soft = soft * soft / (4.0 * p.knee + 1e-5);
    float w = max(soft, l - p.threshold) / max(l, 1e-5);
    dst.write(float4(c.rgb * w, c.a), gid);
}

struct KawaseParams { float iteration; };   // 0,1,2,3... offset = iteration + 0.5 texels

kernel void kawaseBlur(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write>  dst [[texture(1)]],
                       constant KawaseParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 texel = 1.0 / float2(src.get_width(), src.get_height());
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 o = (p.iteration + 0.5) * texel;

    float4 sum = src.sample(s, uv + float2( o.x,  o.y));
    sum       += src.sample(s, uv + float2(-o.x,  o.y));
    sum       += src.sample(s, uv + float2( o.x, -o.y));
    sum       += src.sample(s, uv + float2(-o.x, -o.y));
    dst.write(sum * 0.25, gid);
}

kernel void glowComposite(texture2d<float, access::sample> original [[texture(0)]],
                          texture2d<float, access::sample> glow     [[texture(1)]],
                          texture2d<float, access::write>  dst      [[texture(2)]],
                          constant GlowParams& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float4 o = original.sample(s, uv);
    float3 g = glow.sample(s, uv).rgb * p.tint * p.intensity;
    dst.write(float4(o.rgb + g, o.a), gid);   // additive (screen-like) composite
}

// ============================================================================
// 3. ARAP mesh deformation solver (2D, local/global Jacobi) for Lockdown tracking
//    Rest mesh p, deformed positions q. Handles (tracked points) are pinned to targets.
//    Adjacency stored in CSR form. Host dispatches: arapLocal -> arapGlobal, ping-pong
//    q buffers, repeat 10-30 iterations per frame (warm-started from previous frame).
// ============================================================================
struct ArapParams {
    uint vertexCount;
};

kernel void arapLocal(device const float2* rest      [[buffer(0)]],
                      device const float2* cur       [[buffer(1)]],
                      device const uint*   rowStart  [[buffer(2)]],   // vertexCount + 1
                      device const uint*   neighbors [[buffer(3)]],
                      device const float*  weights   [[buffer(4)]],   // cotangent / uniform weights
                      device float2*       rotations [[buffer(5)]],   // (cos, sin) per vertex
                      constant ArapParams& P         [[buffer(6)]],
                      uint i [[thread_position_in_grid]])
{
    if (i >= P.vertexCount) return;
    float s00 = 0, s01 = 0, s10 = 0, s11 = 0;
    for (uint k = rowStart[i]; k < rowStart[i + 1]; ++k) {
        uint j = neighbors[k];
        float w = weights[k];
        float2 e  = rest[i] - rest[j];
        float2 e2 = cur[i]  - cur[j];
        // S = sum w * e * e2^T
        s00 += w * e.x * e2.x;  s01 += w * e.x * e2.y;
        s10 += w * e.y * e2.x;  s11 += w * e.y * e2.y;
    }
    // Closed-form 2D polar decomposition: best rotation maximizing tr(R^T S).
    float a = s00 + s11;
    float b = s10 - s01;
    float len = max(sqrt(a * a + b * b), 1e-8);
    // R = [c -s; s c] applied to rest edges, c = a/len, s = b/len ... sign chosen so R e ≈ e2.
    rotations[i] = float2(a / len, -b / len);
}

kernel void arapGlobal(device const float2* rest      [[buffer(0)]],
                       device const float2* cur       [[buffer(1)]],
                       device float2*       next      [[buffer(2)]],
                       device const uint*   rowStart  [[buffer(3)]],
                       device const uint*   neighbors [[buffer(4)]],
                       device const float*  weights   [[buffer(5)]],
                       device const float2* rotations [[buffer(6)]],
                       device const float2* pinTarget [[buffer(7)]],
                       device const uchar*  isPinned  [[buffer(8)]],
                       constant ArapParams& P         [[buffer(9)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= P.vertexCount) return;
    if (isPinned[i]) { next[i] = pinTarget[i]; return; }

    float2 acc = float2(0);
    float wsum = 0;
    float2 Ri = rotations[i];
    for (uint k = rowStart[i]; k < rowStart[i + 1]; ++k) {
        uint j = neighbors[k];
        float w = weights[k];
        float2 Rj = rotations[j];
        float2 e = rest[i] - rest[j];
        // 0.5 * (Ri + Rj) * e, with R = [c -s; s c]
        float2 re = 0.5 * float2(Ri.x * e.x - Ri.y * e.y + Rj.x * e.x - Rj.y * e.y,
                                 Ri.y * e.x + Ri.x * e.y + Rj.y * e.x + Rj.x * e.y);
        acc  += w * (cur[j] + re);
        wsum += w;
    }
    next[i] = wsum > 0 ? acc / wsum : cur[i];
}

// Renders the deformed mesh's warp as a per-pixel backward map for compositing:
// each pixel finds its owning triangle (precomputed index texture) and barycentric-interpolates.
struct WarpVertex { float2 rest; float2 cur; };

kernel void meshWarpApply(texture2d<float, access::sample> src       [[texture(0)]],
                          texture2d<uint,  access::read>   triIndex  [[texture(1)]], // triangle id per output pixel
                          texture2d<float, access::write>  dst       [[texture(2)]],
                          device const uint3*   tris   [[buffer(0)]],
                          device const float2*  rest   [[buffer(1)]],
                          device const float2*  cur    [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(coord::pixel, address::clamp_to_edge, filter::linear);
    uint t = triIndex.read(gid).r;
    if (t == 0xFFFFFFFFu) { dst.write(float4(0), gid); return; }

    uint3 v = tris[t];
    float2 a = cur[v.x], b = cur[v.y], c = cur[v.z];
    float2 p = float2(gid) + 0.5;
    float2 v0 = b - a, v1 = c - a, v2 = p - a;
    float d00 = dot(v0, v0), d01 = dot(v0, v1), d11 = dot(v1, v1);
    float d20 = dot(v2, v0), d21 = dot(v2, v1);
    float denom = max(d00 * d11 - d01 * d01, 1e-8);
    float bv = (d11 * d20 - d01 * d21) / denom;
    float bw = (d00 * d21 - d01 * d20) / denom;
    float bu = 1.0 - bv - bw;
    float2 srcPos = bu * rest[v.x] + bv * rest[v.y] + bw * rest[v.z];
    dst.write(src.sample(s, srcPos), gid);
}
