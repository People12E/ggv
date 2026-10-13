# B.P — Architecture Spec

## 1. Frame pipeline (zero-copy, Apple Silicon)

```
Decode (VideoToolbox) ─► IOSurface CVPixelBuffer ─► MTLTexture (CVMetalTextureCache)
        │                                              │
        │                                              ▼
        │                                  Metal compositor (per-layer passes)
        │                                              │
        └──────► Core ML (ANE) ◄── same IOSurface ─────┤
                                                       ▼
                                      MTKView preview  /  AVAssetWriter export
```

- One `SharedFramePool` owns all frame buffers; textures and Core ML inputs are *views* of the same memory.
- Preview uses a triple-buffered in-flight ring (`DispatchSemaphore(3)`), export uses the same compositor driven by `ExportEngine.renderFrame`.
- Neural passes (Chronos, Proteus, Iris, Rotobrush, inpaint) run async on the ANE and are cached per frame; the preview plays cached results and renders at proxy resolution while they compute.

## 2. Layer evaluation order (per frame)

1. Evaluate keyframes (Bezier easing via `BezierEasing`) and expressions → property values.
2. Resolve parenting / 3D Nulls → world transforms.
3. For each layer bottom-to-top: source → masks → effect stack → track matte (alpha / luma / inverted) → transform (2D or 3D projection) → blend mode → accumulate.
4. 3D layers: perspective camera (FOV, focal distance, aperture) → depth buffer → optional depth-of-field pass and shadow-map pass from lights.

## 3. Effect library (51 effects, Metal 3 + Swift)

**Layout** (`Sources/Effects/`)

| File | Role |
|---|---|
| `FX_BlurSharpen / Color / Distort / Generate / Simulation / Stylize / Perspective / Keying .metal` | the kernels, one file per category; `FXCommon.h` holds shared helpers |
| `EffectDescriptor.swift` | schema: parameters, passes, aux inputs |
| `EffectCatalog.swift` | the 51 effects: passes + every parameter in slot order |
| `EffectRunner.swift` | `EffectValues` (parameter storage, proxy-scale handling) and `EffectRunner` (executes a pass graph) |
| `EffectDrivers.swift` | `ParticleRenderer` (instanced particles) and `Layer3DRenderer` (shadow map + colour/depth) |
| `EffectHelpers.swift` | Gaussian weights, Corner Pin homography, camera matrices, curve / .cube LUTs, FFT spectrum |

**How an effect runs.** The UI reads `EffectCatalog` to build its inspector. The runner expands the descriptor's passes
(repeating blur steps from a parameter), binds `src` / `dst` / `aux` plus one packed float buffer `P` (`P[0...4]` =
pass index, comp time, repeat index, per-pass tag, render scale; `P[5...]` = the effect's parameters), and dispatches.
Pixel-unit parameters are multiplied by the render scale on the CPU so a half-resolution proxy looks like full-res.
Frames are premultiplied RGBA in `.rgba16Float`.

**Keeping Swift and Metal in step.** The parameter slot is the ABI. `tools/check_catalog.py` cross-checks the `// @fx`
annotation above every kernel, the `PARAM()` reads in its body, and the catalog (overlaps, undeclared reads, unread
slots, name/slot/width mismatches, unmarked runner parameters, defaults outside range). `tools/check_msl.py`
syntax-checks every `.metal` file against a stub `<metal_stdlib>`. Run both after touching a kernel or the catalog.

| Category | Effects | Driver |
|---|---|---|
| Blur & Sharpen (7) | Camera Lens Blur, Radial, Smart, Directional, Fast Box, Gaussian, Unsharp Mask | compute; separable blurs are 2 passes |
| Color Correction (7) | Curves, Levels, Hue/Saturation, Lumetri (+ optional .cube look), Photo Filter, Color Balance, Tint | compute |
| Distort (7) | Displacement Map, Wave Warp, Polar Coordinates, Corner Pin, Mesh Warp, Offset, Spherize | compute, backward mapping |
| Generate (7) | Grid, 4-Color Gradient, Fractal Noise, Stroke, Audio Spectrum, Cell Pattern, Checkerboard | compute, composited over the layer |
| Simulation (6) | CC Particle World, Particle Playground, CC Particle Systems II, Shatter, Wave World, Foam | particles: render pipeline; others compute |
| Stylize (7) | Glow, Saber Edge Glow, Strobe Light, Mosaic, Find Edges, Threshold, Cartoon | compute; Glow/Saber are 3-pass (half-res Kawase) |
| Perspective (5) | 3D Camera, Bevel Alpha, Drop Shadow, Basic 3D, Bevel Edges | 3D Camera = `Layer3DRenderer` + depth-of-field via Lens Blur |
| Keying (5) | Keylight, Luma Key, Color Range, Difference Matte, Spill Suppressor | compute |

**Simulations are stateless**: every frame is a pure function of comp time and parameters, so scrubbing, looping and
export give identical results with no cached state. The cost is physical realism: no particle collisions, Wave World is
analytic ripples rather than a fluid solve, and Shatter pieces travel at most about two piece-sizes.

**Approximations to know about** (these are After Effects-*style*, not the originals): Keylight is a screen-difference
keyer, not Foundry's algorithm; Saber Edge Glow is driven by bright shapes in the layer and is not the Video Copilot
plug-in; Mesh Warp uses a Catmull-Rom offset field rather than AE's bezier patches; Lumetri covers basic correction,
three wheels and a .cube look, not the full panel.

Gen 1 of the library (typed parameter structs, `BPShaderTypes.h`) is kept under `_archive/gen1/` for reference and is not
part of the build.

## 4. Expressions

A small interpreter (tokenizer → AST → evaluate per frame) supporting `time`, `wiggle(freq, amp)` (seeded noise so renders are deterministic), arithmetic, `thisComp.layer("X").transform.position` property links, `linear/ease`, and `loopOut`. Do **not** embed JavaScriptCore with arbitrary scripts if you plan to share projects — sandboxing becomes a concern.

## 5. Lockdown-style mesh tracking

1. Detect features on the first frame (Vision / FAST corners), build a Delaunay mesh over the user's region.
2. Track points per frame with pyramidal Lucas–Kanade (or the optical-flow model already used by Chronos).
3. Pin tracked points as handles; run 10–30 ARAP local/global iterations per frame, warm-started from the previous frame (`arapLocal` → `arapGlobal`, ping-pong buffers).
4. Apply with `meshWarpApply`; attach layers to mesh vertices so graphics deform with skin/cloth.

Production notes: cotangent weights give better results than uniform weights; add a temporal smoothing term to reduce jitter; re-seed when error against the tracked image exceeds a threshold.

## 6. Neural modules — important caveat

`NeuralPipeline.swift` is the *runtime*. It expects compiled models named `BP_*.mlmodelc` with the input/output names used in the wrappers. You will need to supply or train them (e.g. RIFE → Core ML via coremltools for Chronos; a matting network for Rotobrush; LaMa-style for inpainting). Topaz's own models are proprietary and not available to bundle; use open-licensed models and check each licence.

## 7. Known gaps / next steps

- **Swift has not been compiled** (no Swift toolchain where this was written). Shaders pass `check_msl.py` against a stub
  header, which catches typos and type errors but not Metal-only rules; the Xcode build is the final judge.
- Not yet written: the 27 blend modes and track mattes (the compositor kernel), USDZ/.obj import with lighting inheritance,
  3D camera tracking (solve), the effect inspector UI (the catalog already holds everything it needs), the expression
  evaluator, and a Swift driver for the ARAP / Lockdown kernels in `Metal/Effects.metal`.
- `m.prediction(from:options:)` async requires iOS 17+.
