#include "FXCommon.h"

// =============================================================================
// Simulation. Everything here is STATELESS: a frame is a pure function of (comp time, parameters),
// so scrubbing, looping and multi-frame export all give identical results with no cached state.
// Trade-off: no inter-particle collisions or true fluid dynamics. Wave World is analytic ripples,
// not a PDE solve.
// =============================================================================

// ---- Particle engine (CC Particle World / Particle Playground / CC Particle Systems II) -----------------------
// One instanced draw: 4 vertices (triangle strip) per particle, `instanceCount` particles.
// Particle slot i is reborn every `longevity` seconds with a hashed phase and a per-cycle random seed.
// buffer(0) = P, buffer(1) = float2 render-target size in px, texture(0) = source layer (colour sampling).
//
// Slot map: see tools/check_catalog.py. birthRate (1) and showSource (34) are consumed by the runner.
struct ParticleVert {
    float4 position [[position]];
    float2 q;
    float4 color;
};

// @fx fx_particle_vertex: emitterMode@0 longevity@2 emitterPosition@3:2 emitterZ@5 emitterRadius@6 gridColumns@7 gridRows@8 velocity@9 velocityVariation@10 direction@11 spread@12 gravity@13 resistance@14 turbulence@15 depthSpread@16 birthSize@17 deathSize@18 sizeVariation@19 birthColor@20:4 deathColor@24:4 fadeOut@28 cameraDistance@29 perspective@30 colorFromLayer@33 seed@35 preroll@36
vertex ParticleVert fx_particle_vertex(uint vid [[vertex_id]],
                                       uint iid [[instance_id]],
                                       constant float* P [[buffer(0)]],
                                       constant float2& viewSize [[buffer(1)]],
                                       texture2d<float, access::sample> src [[texture(0)]])
{
    ParticleVert o;
    o.position = float4(2.0f, 2.0f, 0.0f, 1.0f);       // off-screen = culled
    o.q = float2(0.0f);
    o.color = float4(0.0f);

    constexpr sampler ls(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 corner = float2((vid & 1u) ? 1.0f : -1.0f, (vid & 2u) ? 1.0f : -1.0f);
    float time = P[1];
    if (time < 0.0f) return o;

    int   mode = int(PARAM(0));                       // 0 point, 1 disc, 2 grid from layer, 3 line
    float life = max(PARAM(2), 0.05f);
    uint  seed = uint(PARAM(35));
    bool  preroll = PARAM(36) > 0.5f;
    uint  i = iid;

    float phase = fx_hash11(i * 2654435761u ^ (seed * 668265263u)) * life;
    float tt = time + phase;
    float cyc = floor(tt / life);
    float age = tt - cyc * life;
    if (!preroll && (cyc * life - phase) < 0.0f) return o;           // not yet born

    uint h = fx_pcg(i * 747796405u + uint(cyc) * 2891336453u + seed * 69069u);
    float r[8];
    for (int k = 0; k < 8; ++k) r[k] = fx_hash11(h + uint(k) * 0x9e3779b9u);

    float2 vs = viewSize;
    float2 E = float2(PARAM(3), PARAM(4)) * vs;
    float2 sampleUV = E / vs;
    if (mode == 1) {
        float a = r[0] * FX_TAU, rad = sqrt(r[1]) * PARAM(6);
        E += float2(cos(a), sin(a)) * rad;
    } else if (mode == 2) {
        uint cols = max(uint(PARAM(7)), 1u), rows = max(uint(PARAM(8)), 1u);
        uint gi = i % (cols * rows);
        float2 g = float2(float(gi % cols), float(gi / cols));
        E = (g + 0.5f) / float2(float(cols), float(rows)) * vs;
        sampleUV = E / vs;
    } else if (mode == 3) {
        E.x += (r[0] * 2.0f - 1.0f) * PARAM(6);
    }

    float t01 = age / life;
    float dirDeg = PARAM(11) + (r[2] - 0.5f) * PARAM(12);
    float speed = PARAM(9) * (1.0f + (r[3] * 2.0f - 1.0f) * PARAM(10) * 0.01f);
    float2 v0 = fx_ae_dir(dirDeg) * speed;
    float drag = max(PARAM(14), 0.0f);
    float2 travel = (drag > 1e-3f) ? v0 * (1.0f - exp(-drag * age)) / drag : v0 * age;
    float2 grav = float2(0.0f, 0.5f * PARAM(13) * age * age);
    float2 wob = PARAM(15) * float2(sin(age * 3.0f + r[4] * FX_TAU), cos(age * 2.3f + r[5] * FX_TAU)) * min(age, 1.0f);
    float2 pos = E + travel + grav + wob;
    float z = PARAM(5) + (r[6] * 2.0f - 1.0f) * PARAM(16) * age;
    float sz = mix(PARAM(17), PARAM(18), t01) * (1.0f + (r[7] * 2.0f - 1.0f) * PARAM(19) * 0.01f);

    if (PARAM(30) > 0.5f) {                                          // perspective camera looking at the comp centre
        float camD = max(PARAM(29), 1.0f);
        float denom = camD + z;
        if (denom < 1.0f) return o;
        float k = camD / denom;
        float2 c = vs * 0.5f;
        pos = c + (pos - c) * k;
        sz *= k;
    }

    float4 col = mix(PARAM4(20), PARAM4(24), t01);
    if (PARAM(33) > 0.5f) {
        float4 lc = unpremul(src.sample(ls, sampleUV, level(0)));
        col = float4(col.rgb * lc.rgb, col.a * lc.a);
    }
    float fadeStart = 1.0f - max(saturate(PARAM(28) * 0.01f), 1e-3f);
    col.a *= (1.0f - smoothstep(fadeStart, 1.0f, t01)) * smoothstep(0.0f, 0.05f, t01);

    float2 ndc = float2(pos.x / vs.x * 2.0f - 1.0f, 1.0f - pos.y / vs.y * 2.0f) + corner * sz / vs;
    o.position = float4(ndc, 0.0f, 1.0f);
    o.q = corner;
    o.color = col;
    return o;
}

// @fx fx_particle_fragment: shape@31 additive@32
fragment float4 fx_particle_fragment(ParticleVert in [[stage_in]], constant float* P [[buffer(0)]])
{
    int shape = int(PARAM(31));                       // 0 hard disc, 1 soft glow, 2 square
    float add = saturate(PARAM(32));                  // 1 = additive (alpha written as 0)
    float r = length(in.q);
    float a = (shape == 0) ? 1.0f - smoothstep(0.85f, 1.0f, r)
            : (shape == 1) ? pow(saturate(1.0f - r), 2.0f)
            : 1.0f;
    float ca = in.color.a * a;
    return float4(in.color.rgb * ca, ca * (1.0f - add));
}

// ---- Shatter ----------------------------------------------------------------------------------------------------------------------
// Jittered-grid Voronoi pieces. Each output pixel tests the 5x5 neighbouring pieces: it inverts that piece's
// transform (move + spin) and keeps it if the un-moved point still belongs to the piece. The fastest-moving
// owner wins, so flying pieces overlap the ones they leave behind.
// Limit: pieces can travel about 2 piece-sizes from home (displacement is clamped to the search window).
inline float2 fx_shat_point(int2 c, float S, float jit, uint seed)
{
    int2 cs = c + int2(int(seed % 1000u) * 37, int(seed % 1000u) * 91);
    float2 h = fx_hash22(cs);
    return (float2(c) + 0.5f + (h - 0.5f) * jit) * S;
}
inline int2 fx_shat_owner(float2 q, float S, float jit, uint seed)
{
    int2 base = int2(floor(q / S));
    int2 best = base;
    float bd = 1e9f;
    for (int j = -1; j <= 1; ++j)
        for (int i = -1; i <= 1; ++i) {
            int2 c = base + int2(i, j);
            float d = length(q - fx_shat_point(c, S, jit, seed));
            if (d < bd) { bd = d; best = c; }
        }
    return best;
}

// @fx fx_shatter: pieceSize@0 progress@1 strength@2 rotation@3 gravity@4 origin@5:2 seed@7 irregularity@8
kernel void fx_shatter(FX_ARGS)
{
    FX_PROLOGUE
    float S = max(PARAM(0), 4.0f);
    float prog = PARAM(1) * 0.01f;
    float strength = PARAM(2), rotDeg = PARAM(3), gravity = PARAM(4);
    float2 E = PARAM2(5) * size;
    uint seed = uint(PARAM(7));
    float jit = saturate(PARAM(8));

    float2 p = uv * size;
    int2 base = int2(floor(p / S));
    float bestPri = -1.0f;
    float4 bestC = float4(0.0f);
    float maxDim = max(size.x, size.y);

    for (int dy = -2; dy <= 2; ++dy) {
        for (int dx = -2; dx <= 2; ++dx) {
            int2 c = base + int2(dx, dy);
            float2 C = fx_shat_point(c, S, jit, seed);
            float2 hh = fx_hash22(c + int2(int(seed % 977u), 5));
            float2 dir = C - E;
            float dl = length(dir);
            dir = dl > 1e-3f ? dir / dl : float2(0.0f, -1.0f);
            float falloff = 1.0f / (1.0f + dl / (0.5f * maxDim));
            float2 d = dir * strength * (0.5f + hh.x) * falloff * prog + float2(0.0f, gravity * prog * prog);
            float dl2 = length(d);
            float maxD = 1.9f * S;
            if (dl2 > maxD) d *= maxD / dl2;
            float th = (hh.y * 2.0f - 1.0f) * rotDeg * 0.01745329252f * prog;
            float2 q = C + fx_rot(p - C - d, -th);
            int2 owner = fx_shat_owner(q, S, jit, seed);
            if (owner.x != c.x || owner.y != c.y) continue;
            float pri = length(d) + hh.x * 1e-3f;
            if (pri > bestPri) { bestPri = pri; bestC = src.sample(sz, q / size); }
        }
    }
    dst.write(bestPri < 0.0f ? float4(0.0f) : bestC, gid);
}

// ---- Wave World (analytic ripples / plane waves, not a PDE solve) ----------------------------------------------------------
inline float fx_wave_height(float2 p, float2 size, int type, float amp, float wl, float spd, float damp,
                            int count, uint seed, float2 origin, float t)
{
    float h = 0.0f;
    wl = max(wl, 2.0f);
    if (type == 1) {                                          // three crossing plane waves
        for (int i = 0; i < 3; ++i) {
            float2 d = fx_ae_dir(fx_hash11(uint(i) * 977u + seed * 31u) * 360.0f);
            float ph = dot(p, d) / wl - t * spd / wl * (0.7f + 0.3f * float(i));
            h += amp * sin(FX_TAU * ph) / float(1 + i);
        }
        return h;
    }
    for (int j = 0; j < count; ++j) {
        float2 pos = (j == 0) ? origin * size
                              : float2(fx_hash11(uint(j) * 7919u + seed * 17u), fx_hash11(uint(j) * 104729u + seed * 29u)) * size;
        float tj = t - 0.35f * float(j);
        if (tj <= 0.0f) continue;
        float d = length(p - pos);
        float front = spd * tj;
        if (d >= front) continue;                             // wave has not reached this radius yet
        float age = (front - d) / max(spd, 1.0f);
        float env = exp(-damp * age) * exp(-d / max(size.x, 1.0f));
        h += amp * sin(FX_TAU * (d - front) / wl) * env;
    }
    return h;
}

// @fx fx_wave_world: waveType@0 amplitude@1 wavelength@2 speed@3 damping@4 sourceCount@5 seed@6 origin@7:2 output@9 refraction@10 lightAngle@11
kernel void fx_wave_world(FX_ARGS)
{
    FX_PROLOGUE
    int   type = int(PARAM(0));                              // 0 ripples, 1 plane waves
    float amp = PARAM(1), wl = PARAM(2), spd = PARAM(3), damp = PARAM(4);
    int   count = clamp(int(PARAM(5)), 1, 8);
    uint  seed = uint(PARAM(6));
    float2 origin = PARAM2(7);
    int   outMode = int(PARAM(9));                           // 0 height, 1 refracted layer, 2 normal map
    float refr = PARAM(10);
    float2 L2 = fx_ae_dir(PARAM(11));

    float2 p = uv * size;
    float e = 1.5f;
    float hC = fx_wave_height(p, size, type, amp, wl, spd, damp, count, seed, origin, time);
    float hx = fx_wave_height(p + float2(e, 0.0f), size, type, amp, wl, spd, damp, count, seed, origin, time)
             - fx_wave_height(p - float2(e, 0.0f), size, type, amp, wl, spd, damp, count, seed, origin, time);
    float hy = fx_wave_height(p + float2(0.0f, e), size, type, amp, wl, spd, damp, count, seed, origin, time)
             - fx_wave_height(p - float2(0.0f, e), size, type, amp, wl, spd, damp, count, seed, origin, time);
    float2 grad = float2(hx, hy) / (2.0f * e);
    float3 n = normalize(float3(-grad * 4.0f, 1.0f));

    if (outMode == 0) {
        float v = saturate(hC / max(2.0f * amp, 1e-3f) + 0.5f);
        dst.write(float4(float3(v), 1.0f), gid);
    } else if (outMode == 2) {
        dst.write(float4(n * 0.5f + 0.5f, 1.0f), gid);
    } else {
        float4 c = src.sample(s, uv + grad * refr / size);
        float3 Lv = normalize(float3(L2 * 0.6f, 0.8f));
        float diffuse = saturate(dot(n, Lv));
        float spec = pow(saturate(dot(reflect(-Lv, n), float3(0.0f, 0.0f, 1.0f))), 40.0f);
        c.rgb = c.rgb * (0.75f + 0.5f * diffuse) + spec * 0.6f * c.a;
        dst.write(c, gid);
    }
}

// ---- Foam (rising bubbles with refraction, rim and specular) ---------------------------------------------------------------------
// Bubbles live in vertical columns; each column hosts 4 bubbles per depth layer that rise and wrap.
// @fx fx_foam: bubbleSize@0 sizeVariation@1 density@2 riseSpeed@3 wobble@4 refraction@5 rimStrength@6 specular@7 seed@8 viewMode@9
kernel void fx_foam(FX_ARGS)
{
    FX_PROLOGUE
    float bsize = max(PARAM(0), 4.0f);
    float sizeVar = saturate(PARAM(1) * 0.01f);
    float density = saturate(PARAM(2) * 0.01f);
    float rise = PARAM(3), wobble = PARAM(4);
    float refr = PARAM(5), rimK = PARAM(6) * 0.01f, specK = PARAM(7) * 0.01f;
    uint  seed = uint(PARAM(8));
    bool  bubblesOnly = PARAM(9) > 0.5f;

    float2 p = uv * size;
    float4 bg = src.sample(s, uv);
    float3 resultRGB = bubblesOnly ? float3(0.0f) : bg.rgb;
    float  resultA   = bubblesOnly ? 0.0f : bg.a;
    float3 Ls = normalize(float3(-0.5f, -0.6f, 0.7f));

    for (int layer = 2; layer >= 0; --layer) {                       // back to front
        float sl = (layer == 0) ? 1.0f : ((layer == 1) ? 0.65f : 0.4f);
        float W = bsize * sl * 1.6f;
        int col = int(floor(p.x / W));
        for (int dc = -1; dc <= 1; ++dc) {
            int ci = col + dc;
            for (int k = 0; k < 4; ++k) {
                uint hid = uint(ci) * 73856093u ^ uint(k) * 19349663u ^ uint(layer) * 83492791u ^ seed * 2654435761u;
                if (fx_hash11(hid) > density) continue;
                float hx = fx_hash11(hid + 1u), hy = fx_hash11(hid + 2u), hr = fx_hash11(hid + 3u), hs = fx_hash11(hid + 4u);
                float r = bsize * sl * 0.5f * (1.0f + (hr - 0.5f) * 2.0f * sizeVar);
                float H = size.y + 4.0f * r;
                float y = size.y + 2.0f * r - fx_fmod(hy * H + rise * sl * (0.7f + 0.6f * hs) * time, H);
                float x = (float(ci) + hx) * W + wobble * r * sin(time * 2.0f + hs * FX_TAU);
                float2 d = p - float2(x, y);
                float dist = length(d);
                if (dist > r + 1.0f) continue;
                float cover = 1.0f - smoothstep(r - 1.0f, r + 1.0f, dist);
                float2 nxy = d / r;
                float nz = sqrt(max(1.0f - dot(nxy, nxy), 0.0f));
                float3 n = float3(nxy, nz);
                float rim = pow(1.0f - nz, 3.0f) * rimK;
                float spec = pow(saturate(dot(reflect(-Ls, n), float3(0.0f, 0.0f, 1.0f))), 30.0f) * specK;
                if (bubblesOnly) {
                    float a = saturate(cover * (0.12f + rim + spec));
                    resultRGB = resultRGB * (1.0f - a) + float3(rim + spec) * a;
                    resultA = resultA + a * (1.0f - resultA);
                } else {
                    float4 refrC = src.sample(s, uv - nxy * (1.0f - nz) * refr * r / size);
                    float3 bubbleRGB = refrC.rgb + rim + spec;
                    resultRGB = mix(resultRGB, bubbleRGB, cover);
                }
            }
        }
    }
    dst.write(float4(resultRGB, resultA), gid);
}
