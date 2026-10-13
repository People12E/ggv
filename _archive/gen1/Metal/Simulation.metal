#include <metal_stdlib>
#include "BPShaderTypes.h"
using namespace metal;

constexpr sampler linClamp(coord::normalized, address::clamp_to_edge, filter::linear);

// ===========================================================================
// PARTICLES (shared by CC Particle World / Particle Systems II / Particle Playground)
//
// Determinism: scrubbing the timeline must give the same result every time. The host either
//  (a) re-simulates from frame 0 with fixed dt, or (b) caches checkpoints every N frames.
//  Randomness comes from hash(particleIndex, seed), never from wall-clock.
// ===========================================================================



static inline uint pcgHash(uint v) {
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
static inline float rand01(uint h) { return float(pcgHash(h)) / 4294967295.0; }

static inline float3 curlish(float3 p, float t) {   // cheap divergence-light swirl for turbulence
    return float3(sin(p.y * 1.7 + t) + cos(p.z * 1.3 - t),
                  sin(p.z * 1.9 - t) + cos(p.x * 1.1 + t),
                  sin(p.x * 1.5 + t) + cos(p.y * 1.7 - t));
}

kernel void particleUpdate(device Particle* particles [[buffer(0)]],
                           constant EmitterParams& E  [[buffer(1)]],
                           uint i [[thread_position_in_grid]])
{
    if (i >= E.particleCount) return;
    Particle p = particles[i];

    if (p.life <= 0.0) {
        // Respawn schedule: particle i is born at time (i / birthRate) + k * (particleCount / birthRate).
        float cycle = float(E.particleCount) / max(E.birthRate, 1e-3);
        float birth = fmod(float(i) / max(E.birthRate, 1e-3), cycle);
        float tt = fmod(E.time, cycle);
        float since = tt - birth;
        if (since < 0.0) since += cycle;
        if (since > E.dt) { particles[i] = p; return; }       // not due yet

        uint base = i * 7919u + E.seed + uint(floor(E.time / cycle)) * 104729u;
        float r0 = rand01(base + 1), r1 = rand01(base + 2), r2 = rand01(base + 3), r3 = rand01(base + 4);

        float3 offset = float3(0);
        if (E.shape == 1) {
            float3 d = normalize(float3(r0, r1, r2) * 2.0 - 1.0 + 1e-4);
            offset = d * E.emitterSize.x * pow(r3, 1.0 / 3.0);
        } else if (E.shape == 2) {
            offset = (float3(r0, r1, r2) - 0.5) * E.emitterSize;
        }
        float theta = r1 * 6.2831853;
        float phi = r0 * E.spreadAngle;
        float3 dir = float3(sin(phi) * cos(theta), cos(phi), sin(phi) * sin(theta));
        float spd = E.speed * (1.0 + (rand01(base + 5) - 0.5) * 2.0 * E.speedJitter);

        p.position = E.emitterPos + offset;
        p.velocity = dir * spd;
        p.life = E.lifetime * (1.0 + (rand01(base + 6) - 0.5) * 2.0 * E.lifetimeJitter);
        p.age = 0;
        p.rotation = rand01(base + 7) * 6.2831853;
    }

    float3 accel = E.gravity + curlish(p.position, E.time) * E.turbulence;
    p.velocity += accel * E.dt;
    p.velocity *= (1.0 - E.drag * E.dt);
    p.position += p.velocity * E.dt;

    if (E.floorBounce != 0 && p.position.y < E.floorY) {
        p.position.y = E.floorY;
        p.velocity.y = -p.velocity.y * E.restitution;
    }

    p.age += E.dt;
    p.life -= E.dt;
    float t = saturate(p.age / max(p.age + max(p.life, 0.0), 1e-4));
    p.color = mix(E.color0, E.color1, t);
    p.size  = mix(E.size0, E.size1, t);
    particles[i] = p;
}

// Splat particles into an accumulation texture (additive soft discs). One thread per particle writes
// a small footprint using atomics-free approach: each thread handles a tile pixel via a gather instead.
// For production use a render pipeline with point sprites; this compute gather is simple and portable.

kernel void particleSplatGather(device const Particle* particles [[buffer(0)]],
                                constant SplatParams& S           [[buffer(1)]],
                                texture2d<float, access::write>  dst [[texture(0)]],
                                uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 pix = float2(gid) + 0.5;
    float4 acc = float4(0);
    for (uint i = 0; i < S.particleCount; ++i) {
        Particle p = particles[i];
        if (p.life <= 0) continue;
        float4 clip = S.viewProj * float4(p.position, 1.0);
        if (clip.w <= 0) continue;
        float2 ndc = clip.xy / clip.w;
        float2 sp = float2(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * S.viewport;
        float r = p.size / clip.w * S.viewport.y * 0.5;
        float d = distance(pix, sp);
        if (d < r) {
            float a = (1.0 - smoothstep(r * (1.0 - clamp(S.softness, 0.0, 0.99)), r, d)) * p.color.a;
            acc.rgb += p.color.rgb * a;
            acc.a = max(acc.a, a);
        }
    }
    dst.write(acc, gid);
}

// ===========================================================================
// SHATTER (2D): Voronoi cells, each cell pushed out and rotated by a blast radius around a center.
// Per-pixel: find nearest seed, compute that cell's transform, backward-map into the source.
// ===========================================================================

static inline float2 hash22s(float2 p, uint seed) {
    float3 p3 = fract(float3(p.xyx + float(seed) * 0.013) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}

kernel void simShatter(texture2d<float, access::sample> src [[texture(0)]],
                       texture2d<float, access::write>  dst [[texture(1)]],
                       constant ShatterParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float aspect = float(dst.get_width()) / float(dst.get_height());
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());

    // Because displaced cells overlap/leave gaps, search a window of candidate cells and keep the
    // front-most one that actually covers this pixel after its transform.
    float2 q = uv * float2(aspect, 1.0) * p.cellScale;
    float2 base = floor(q);
    float4 best = float4(0);
    float bestZ = -1e9;

    for (int y = -3; y <= 3; ++y)
        for (int x = -3; x <= 3; ++x) {
            float2 cellId = base + float2(x, y);
            float2 seed = (cellId + hash22s(cellId, p.seed)) / (float2(aspect, 1.0) * p.cellScale);   // seed in uv
            float dist = distance(seed, p.blastCenter);
            float push = saturate(1.0 - dist / max(p.radius, 1e-4)) * p.force * p.time;
            float2 dir = normalize(seed - p.blastCenter + 1e-5);
            float2 move = dir * push + float2(0, p.gravity * p.time * p.time * 0.5);
            float ang = (hash22s(cellId, p.seed + 7u).x - 0.5) * p.rotation * p.time;

            // Map this pixel back into the cell's rest frame.
            float2 local = uv - seed - move;
            float c = cos(-ang), s = sin(-ang);
            float2 rest = seed + float2(c * local.x - s * local.y, s * local.x + c * local.y);

            // Is `rest` inside this Voronoi cell (i.e. nearest seed to `rest` is `seed`)?
            float2 rq = rest * float2(aspect, 1.0) * p.cellScale;
            float2 rb = floor(rq);
            float dmin = 1e9; float2 nearest = float2(0);
            for (int j = -1; j <= 1; ++j)
                for (int i = -1; i <= 1; ++i) {
                    float2 cid = rb + float2(i, j);
                    float2 sd = (cid + hash22s(cid, p.seed)) / (float2(aspect, 1.0) * p.cellScale);
                    float d = distance(rest, sd);
                    if (d < dmin) { dmin = d; nearest = sd; }
                }
            if (distance(nearest, seed) < 1e-6) {
                float z = push;                                  // cells blown further out are drawn on top
                if (z > bestZ && all(rest >= 0.0) && all(rest <= 1.0)) { bestZ = z; best = src.sample(linClamp, rest); }
            }
        }
    dst.write(best, gid);
}

// ===========================================================================
// WAVE WORLD: 2D wave equation on a height field. State = (h_now, h_prev) in an RG32Float texture.
// Host ping-pongs textures each step, injects impulses with `waveImpulse`.
// ===========================================================================
static inline float waveH(texture2d<float, access::read> t, int2 c) {
    int2 size = int2(t.get_width(), t.get_height());
    return t.read(uint2(clamp(c, int2(0), size - 1))).r;      // clamped reads = reflecting walls
}

kernel void waveStep(texture2d<float, access::read>  stateIn  [[texture(0)]],   // r = h, g = h_prev
                     texture2d<float, access::write> stateOut [[texture(1)]],
                     constant WaveStepParams& p [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= stateIn.get_width() || gid.y >= stateIn.get_height()) return;
    int2 c = int2(gid);
    float2 s = stateIn.read(gid).rg;
    float lap = waveH(stateIn, c + int2(1, 0)) + waveH(stateIn, c + int2(-1, 0))
              + waveH(stateIn, c + int2(0, 1)) + waveH(stateIn, c + int2(0, -1)) - 4.0 * s.x;
    float c2 = p.speed * p.speed * p.dt * p.dt;         // keep c2 <= 0.5 for stability
    float next = 2.0 * s.x - s.y + c2 * lap;
    next *= (1.0 - p.damping);
    stateOut.write(float4(next, s.x, 0, 0), gid);
}

// Adds a raised-cosine impulse while copying state (no in-place read_write: RG32Float does not support it everywhere).
kernel void waveImpulseStep(texture2d<float, access::read>  stateIn  [[texture(0)]],
                            texture2d<float, access::write> stateOut [[texture(1)]],
                            constant ImpulseParams& p [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= stateIn.get_width() || gid.y >= stateIn.get_height()) return;
    float4 s = stateIn.read(gid);
    float d = distance(float2(gid), p.position);
    if (d <= p.radius) s.r += p.strength * 0.5 * (1.0 + cos(3.14159265 * d / max(p.radius, 1e-3)));
    stateOut.write(s, gid);
}

// Render the height field as refraction + specular over the source image.
kernel void waveRender(texture2d<float, access::sample> src   [[texture(0)]],
                       texture2d<float, access::read>   state [[texture(1)]],
                       texture2d<float, access::write>  dst   [[texture(2)]],
                       constant WaveRenderParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]])
{
    uint w = dst.get_width(), h = dst.get_height();
    if (gid.x >= w || gid.y >= h) return;
    int2 c = int2(gid);
    float2 grad = float2(waveH(state, c + int2(1, 0)) - waveH(state, c + int2(-1, 0)),
                         waveH(state, c + int2(0, 1)) - waveH(state, c + int2(0, -1))) * 0.5;
    float3 n = normalize(float3(-grad * 8.0, 1.0));
    float2 uv = (float2(gid) + 0.5) / float2(w, h);
    float4 col = src.sample(linClamp, uv + grad * p.refraction);
    float3 Hv = normalize(normalize(p.lightDir) + float3(0, 0, 1));
    float spec = pow(saturate(dot(n, Hv)), p.shine) * p.specular;
    dst.write(float4(col.rgb + spec, col.a), gid);
}

// ===========================================================================
// FOAM: packed bubbles via Worley cells shaded as lit spheres that refract the source.
// ===========================================================================

kernel void simFoam(texture2d<float, access::sample> src [[texture(0)]],
                    texture2d<float, access::write>  dst [[texture(1)]],
                    constant FoamParams& p [[buffer(0)]],
                    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float aspect = float(dst.get_width()) / float(dst.get_height());
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float2 q = uv * float2(aspect, 1.0) * p.scale;
    float2 i = floor(q), f = fract(q);

    float best = 1e9; float2 bestVec = float2(0); float bestR = 0.5;
    for (int y = -1; y <= 1; ++y)
        for (int x = -1; x <= 1; ++x) {
            float2 g = float2(x, y);
            float2 h = hash22s(i + g, p.seed);
            float2 o = 0.5 + 0.35 * sin(p.time * p.viscosity + 6.2831853 * h);   // bubbles drift slowly
            float2 v = g + o - f;
            float r = mix(0.25, 0.5, h.x) * p.bubbleSize;
            float d = length(v) / r;
            if (d < best) { best = d; bestVec = v; bestR = r; }
        }

    float4 base = src.sample(linClamp, uv);
    if (best >= 1.0) { dst.write(base * 0.85, gid); return; }     // gaps between bubbles
    float z = sqrt(1.0 - best * best);
    float3 n = normalize(float3(bestVec / bestR, z));             // sphere normal
    float2 refr = -n.xy * p.refraction * (1.0 - z);
    float4 c = src.sample(linClamp, uv + refr);
    float3 Hv = normalize(normalize(p.lightDir) + float3(0, 0, 1));
    float spec = pow(saturate(dot(n, Hv)), 40.0);
    float rim = pow(1.0 - z, 3.0) * 0.25;
    dst.write(float4(c.rgb + spec * 0.8 + rim, c.a), gid);
}

// ===========================================================================
// PARTICLE SPRITES (render pipeline). One instanced quad per particle, read straight from the
// simulation buffer: cost is O(particles), not O(pixels x particles) like particleSplatGather.
// Blend on the host: additive (One, One) for glows, or premultiplied (One, OneMinusSourceAlpha).
// ===========================================================================
struct SpriteOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

constant float2 kSpriteQuad[6] = { float2(-1, -1), float2(1, -1), float2(-1, 1),
                                   float2(-1,  1), float2(1, -1), float2(1,  1) };

vertex SpriteOut particleSpriteVertex(uint vid [[vertex_id]],
                                      uint iid [[instance_id]],
                                      device const Particle* particles [[buffer(0)]],
                                      constant SplatParams& S           [[buffer(1)]])
{
    SpriteOut o;
    Particle p = particles[iid];
    float2 corner = kSpriteQuad[vid];
    o.uv = corner;
    o.color = p.color;

    float4 clip = S.viewProj * float4(p.position, 1.0);
    if (p.life <= 0.0 || clip.w <= 1e-4) {            // dead or behind the camera: push outside the clip volume
        o.position = float4(0.0, 0.0, -2.0, 1.0);
        o.color = float4(0.0);
        return o;
    }
    float r = p.size / clip.w * S.viewport.y * 0.5;   // radius in pixels (same convention as the gather kernel)
    float cs = cos(p.rotation), sn = sin(p.rotation);
    float2 rc = float2(cs * corner.x - sn * corner.y, sn * corner.x + cs * corner.y);
    clip.xy += rc * r * 2.0 / S.viewport * clip.w;
    o.position = clip;
    return o;
}

fragment float4 particleSpriteFragment(SpriteOut in [[stage_in]],
                                       constant SplatParams& S [[buffer(0)]])
{
    float d = length(in.uv);
    if (d >= 1.0) discard_fragment();
    float soft = clamp(S.softness, 0.01, 1.0);
    float a = (1.0 - smoothstep(1.0 - soft, 1.0, d)) * in.color.a;
    return float4(in.color.rgb * a, a);               // premultiplied
}
