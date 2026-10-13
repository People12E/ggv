// BPShaderTypes.h
// Single source of truth for every parameter struct shared by Metal shaders and Swift.
//
// Include from MSL with  #include "BPShaderTypes.h"
// Expose to Swift by importing it in the target's bridging header (see BP-Bridging-Header.h).
//
// Layout rules that matter (identical in MSL, C and Swift):
//   - vector_float3 is 16 bytes and 16-byte aligned, so a float that follows it starts at +16.
//   - vector_float2 is 8-byte aligned, matrix_float3x3 is 48 bytes, matrix_float4x4 is 64 bytes.
//   - Use `unsigned int` / `int` for flags and counts, never bool.
// Because Swift imports these exact C structs, there are no hand-mirrored copies to drift out of sync.

#ifndef BPShaderTypes_h
#define BPShaderTypes_h

#include <simd/simd.h>

// ---------------------------------------------------------------------------
// Displacement map / glow / ARAP (Effects.metal)
// ---------------------------------------------------------------------------
typedef struct {
    vector_float2 strength;      // displacement in pixels (x, y)
    float caAmount;              // radial chromatic aberration amount
    vector_float2 center;        // normalized CA centre
    unsigned int mapChannelX;    // 0=R 1=G 2=B 3=A 4=luma
    unsigned int mapChannelY;
} DisplaceParams;

typedef struct {
    float threshold;
    float knee;
    float intensity;
    vector_float3 tint;
} GlowParams;

typedef struct { float iteration; } KawaseParams;      // offset = iteration + 0.5 texels

typedef struct { unsigned int vertexCount; } ArapParams;

// ---------------------------------------------------------------------------
// Blur & sharpen (Blur.metal)
// ---------------------------------------------------------------------------
typedef struct { vector_float2 direction; int radius; } SeparableParams;   // weights in buffer(1)
typedef struct { float angle; float length; int samples; } DirectionalParams;   // angle radians, length px
typedef struct { vector_float2 center; float amount; unsigned int zoom; int samples; } RadialParams; // amount: radians (spin) or fraction (zoom)
typedef struct { int radius; float spatialSigma; float threshold; } SmartParams;
typedef struct { float maxRadius; float focalDepth; float focalRange; int samples; int blades; float brightBoost; } LensParams;
typedef struct { float amount; float threshold; } UnsharpParams;

// ---------------------------------------------------------------------------
// Colour correction & stylize (ColorStylize.metal)
// ---------------------------------------------------------------------------
typedef struct { float inBlack; float inWhite; float gamma; float outBlack; float outWhite; } LevelsParams;
typedef struct { float hueShift; float saturation; float lightness; } HueSatParams;      // hue in turns (-0.5...0.5); saturation is a multiplier
typedef struct {
    float exposure; float contrast; float highlights; float shadows; float whites; float blacks;
    float temperature; float tint; float vibrance; float saturation; float lutIntensity;
    unsigned int hasLUT;
} LumetriParams;
typedef struct { vector_float3 filterColor; float density; unsigned int preserveLuminosity; } PhotoFilterParams;
typedef struct { vector_float3 shadows; vector_float3 midtones; vector_float3 highlights; unsigned int preserveLuminosity; } ColorBalanceParams;
typedef struct { vector_float3 mapBlack; vector_float3 mapWhite; float amount; } TintParams;

typedef struct { vector_float3 color; float edgeStrength; float threshold; float intensity; } SaberParams;
typedef struct { unsigned int invert; float blend; } FindEdgesParams;
typedef struct { float level; } ThresholdParams;
typedef struct { vector_int2 blocks; unsigned int sharpColors; } MosaicParams;
typedef struct { vector_float3 color; float intensity; float period; float duration; float time; unsigned int invert; } StrobeParams;
typedef struct { float levels; float edgeThreshold; float edgeWidth; vector_float3 lineColor; } CartoonParams;

// ---------------------------------------------------------------------------
// Distort & generate (DistortGenerate.metal)
// ---------------------------------------------------------------------------
typedef struct { float amplitude; float wavelength; float speed; float time; float angle; unsigned int shape; } WaveWarpParams; // shape 0 sine, 1 triangle, 2 square
typedef struct { float amount; unsigned int mode; } PolarParams;                     // mode 0 rect->polar, 1 polar->rect
typedef struct { matrix_float3x3 outToSrc; } CornerPinParams;                        // output uv -> source uv (homography)
typedef struct { int cols; int rows; } MeshWarpParams;                               // (cols+1)*(rows+1) float2 uv offsets in buffer(1)
typedef struct { vector_float2 shift; unsigned int wrap; } OffsetParams;             // shift in uv
typedef struct { vector_float2 center; float radius; float amount; } SpherizeParams; // radius in uv-height units

typedef struct { float scale; int octaves; float lacunarity; float gain; float contrast; float brightness; float evolution; unsigned int turbulent; } FractalNoiseParams;
typedef struct { vector_float2 cell; float lineWidth; vector_float3 color; vector_float3 background; float feather; } GridParams;
typedef struct { vector_float2 size; vector_float3 colorA; vector_float3 colorB; } CheckerParams;
typedef struct { vector_float2 p0; vector_float2 p1; vector_float2 p2; vector_float2 p3;
                 vector_float3 c0; vector_float3 c1; vector_float3 c2; vector_float3 c3; float blend; } FourColorParams;
typedef struct { float scale; float evolution; unsigned int mode; float contrast; } CellParams;  // mode 0 F1, 1 F2-F1, 2 inverted F1
typedef struct { vector_float3 color; float width; float opacity; int samples; } StrokeParams;
typedef struct { int bandCount; vector_float2 start; vector_float2 end; float maxHeight; float thickness; vector_float3 color; float gap; } SpectrumParams; // bins in buffer(1)

// ---------------------------------------------------------------------------
// Perspective & keying (PerspectiveKeying.metal)
// ---------------------------------------------------------------------------
typedef struct { vector_float2 offset; vector_float3 color; float opacity; unsigned int shadowOnly; } DropShadowParams; // offset in uv
typedef struct {
    float depth;                 // height-field gradient scale
    vector_float2 lightDir;
    vector_float3 highlight;
    vector_float3 shadow;
    float intensity;
    unsigned int mode;           // 0 = Bevel Alpha (alpha height), 1 = luma height, 2 = Bevel Edges (distance to layer rectangle)
    float edgeWidth;             // mode 2: bevel width in pixels
} BevelParams;
typedef struct { matrix_float3x3 outToPlane; vector_float3 normal; vector_float3 lightDir; float specular; float shine; } Basic3DParams;
typedef struct { float depth; } LayerDepthParams;
typedef struct { matrix_float4x4 lightViewProj; float bias; float softness; float strength; } ShadowParams;
typedef struct {
    matrix_float4x4 invViewProj;   // clip -> world
    matrix_float4x4 worldToLayer;  // world -> layer pixel space (z = 0 is the layer plane, origin top-left)
    vector_float3 normal;          // world-space layer normal
    vector_float3 lightDir;        // world-space direction towards the light
    vector_float2 layerSize;       // layer size in pixels
    float depthNear;
    float depthFar;
    float specular;
    float shine;
    float opacity;
    unsigned int twoSided;
} Camera3DParams;

typedef struct { vector_float3 keyColor; float screenGain; float screenBalance; float clipBlack; float clipWhite; float despill; } KeyParams;
typedef struct { float threshold; float tolerance; float edgeFeather; unsigned int keyOutBrighter; } LumaKeyParams;
typedef struct { vector_float3 minYCC; vector_float3 maxYCC; float feather; } ColorRangeParams;
typedef struct { float tolerance; float softness; float blend; } DiffMatteParams;
typedef struct { vector_float3 keyColor; float amount; } SpillParams;

// ---------------------------------------------------------------------------
// Simulation (Simulation.metal)
// ---------------------------------------------------------------------------
typedef struct {
    vector_float3 position;
    float life;                  // remaining seconds; <= 0 means dead / unborn
    vector_float3 velocity;
    float age;
    vector_float4 color;
    float size;
    float rotation;
    vector_float2 _pad;
} Particle;

typedef struct {
    vector_float3 emitterPos;
    float birthRate;             // particles / second
    vector_float3 gravity;
    float lifetime;
    float lifetimeJitter;
    float speed;
    float speedJitter;
    float spreadAngle;           // radians, cone around +Y
    float drag;
    float size0;
    float size1;
    vector_float4 color0;
    vector_float4 color1;
    float turbulence;
    float time;
    float dt;
    unsigned int seed;
    unsigned int particleCount;
    unsigned int shape;          // 0 point, 1 sphere, 2 box
    vector_float3 emitterSize;
    unsigned int floorBounce;
    float floorY;
    float restitution;
} EmitterParams;

typedef struct { matrix_float4x4 viewProj; unsigned int particleCount; vector_float2 viewport; float softness; } SplatParams;
typedef struct { vector_float2 blastCenter; float radius; float force; float rotation; float gravity; float time; float cellScale; unsigned int seed; } ShatterParams;
typedef struct { float speed; float damping; float dt; } WaveStepParams;
typedef struct { vector_float2 position; float radius; float strength; } ImpulseParams;       // position in pixels
typedef struct { float refraction; vector_float3 lightDir; float specular; float shine; } WaveRenderParams;
typedef struct { float scale; float bubbleSize; float refraction; vector_float3 lightDir; float time; float viscosity; unsigned int seed; } FoamParams;

#endif /* BPShaderTypes_h */
