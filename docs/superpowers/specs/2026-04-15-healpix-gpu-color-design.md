# HEALPix GPU Color Computation — Design Spec

**Date:** 2026-04-15
**Branch:** `feature/gpu-corners`
**Status:** Approved

---

## Overview

Redesign the `HealpixCellsLayer` API to accept raw per-cell float values and compute colors entirely on the GPU. This replaces the current approach of pre-baking RGBA color frames on the CPU and uploading them as a texture array.

The new API supports multi-frame animation via a `frames` array and a `currentFrame` index. Root-level props serve as shared defaults for all frames. Color computation supports 1–4 dimensional values, a colorMap LUT, and per-frame overrides.

---

## API

### `HealpixFrameObject`

```ts
type HealpixFrameObject = {
  nside?: number               // overrides root nside
  scheme?: 'nest' | 'ring'    // overrides root scheme
  cellIds?: CellIdArray        // overrides root cellIds
  values: TypedArray           // interleaved floats, length = cellIds.length * dimensions
  min?: number                 // overrides root min
  max?: number                 // overrides root max
  dimensions?: 1 | 2 | 3 | 4  // overrides root dimensions
  colorMap?: Uint8Array        // overrides root colorMap (must be 256 × 4 = 1024 bytes)
}
```

### `HealpixCellsLayerProps`

```ts
type HealpixCellsLayerProps = {
  // Required (must be present at root or on the frame object)
  nside: number
  cellIds: CellIdArray

  // Geometry default
  scheme?: 'nest' | 'ring'    // default: 'nest'

  // Single-frame values (used when `frames` is absent)
  values?: TypedArray          // interleaved floats, length = cellIds.length * dimensions

  // Color defaults (apply to root single-frame mode and as fallbacks for frame objects)
  min?: number                 // default: 0
  max?: number                 // default: 1
  dimensions?: 1 | 2 | 3 | 4  // default: 1
  colorMap?: Uint8Array        // default: linear black→white gradient (256×4 RGBA)

  // Multi-frame
  frames?: HealpixFrameObject[]
  currentFrame?: number        // default: 0, clamped to [0, frames.length - 1]
} & CompositeLayerProps
```

### Single-frame usage

When `frames` is absent the layer renders a single frame from root props directly:

```tsx
<HealpixCellsLayer
  nside={512}
  cellIds={myIds}
  values={myValues}
/>
```

### Multi-frame usage

Root props serve as defaults; frame objects override selectively:

```tsx
<HealpixCellsLayer
  nside={512}
  colorMap={myLut}
  frames={[
    { cellIds: ids0, values: vals0 },
    { cellIds: ids1, values: vals1, min: -1 },
    { nside: 1024, cellIds: ids2, values: vals2 },
  ]}
  currentFrame={activeFrame}
/>
```

---

## Color Dimensions

The `dimensions` prop controls how per-cell `values` are interpreted. Values are interleaved: cell `i` occupies indices `i * dimensions` through `i * dimensions + dimensions - 1`.

### `dimensions = 1` — scalar → colorMap

Each cell has one float value. It is normalized through `[min, max]` to an index into the colorMap. The colorMap determines the final RGBA color.

```
value → normalize(min, max) → colorMap[0..255] → RGBA
opacity = colorMap.alpha × layer.opacity
```

### `dimensions = 2` — scalar + opacity → colorMap

Each cell has two float values:
- Value 0: normalized through `[min, max]` and looked up in the colorMap (same as dim 1)
- Value 1: opacity multiplier in range `[0, 1]`, applied to the colorMap alpha

```
value[0] → normalize(min, max) → colorMap[0..255] → RGB
value[1] → opacity multiplier
final alpha = colorMap.alpha × value[1] × layer.opacity
```

### `dimensions = 3` — direct RGB

Each cell has three float values in range `[0, 1]` interpreted directly as R, G, B. The colorMap and min/max are ignored. Alpha is always 1.0.

```
(value[0], value[1], value[2]) → RGB, alpha = 1.0
final alpha = layer.opacity
```

### `dimensions = 4` — direct RGBA

Each cell has four float values in range `[0, 1]` interpreted directly as R, G, B, A. The colorMap and min/max are ignored.

```
(value[0], value[1], value[2], value[3]) → RGBA
final alpha = value[3] × layer.opacity
```

### `dimensions > 4` — reserved

Not yet supported. The layer renders all such cells as transparent (`vec4(0.0)`) and emits a console warning. Future versions will use extra dimensions for band math operations.

---

## Frame Resolution

At render time the effective frame is computed by merging root props with the current frame object. Frame values take precedence:

```
effectiveFrame = {
  nside:      frame.nside      ?? props.nside,
  scheme:     frame.scheme     ?? props.scheme     ?? 'nest',
  cellIds:    frame.cellIds    ?? props.cellIds,
  values:     frame.values     ?? props.values,
  min:        frame.min        ?? props.min        ?? 0,
  max:        frame.max        ?? props.max        ?? 1,
  dimensions: frame.dimensions ?? props.dimensions ?? 1,
  colorMap:   frame.colorMap   ?? props.colorMap   ?? DEFAULT_COLORMAP,
}
```

When `frames` is absent, `frame` is treated as `{}` and root props fill all fields.

### Validation

- `nside` must be present (root or frame). Missing → throw with a clear error.
- `cellIds` must be present (root or frame). Missing → throw with a clear error.
- `colorMap` must be exactly 1024 bytes (256 × 4 RGBA). Invalid → throw with a clear error.
- `dimensions > 4` → transparent cells + console warning (no throw).

---

## Change Detection & Resource Rebuilding

Resources are only rebuilt when their inputs change, not on every frame switch. This means globally-defined props (e.g. a shared `nside` or `cellIds`) are never redundantly re-uploaded when only `currentFrame` advances.

| Resource | Rebuilt when effective value changes |
|---|---|
| Geometry (cellIdLo/Hi attributes) | `cellIds`, `nside`, or `scheme` |
| Values texture | `values`, `dimensions`, or `cellIds.length` |
| ColorMap texture | `colorMap` reference |
| Uniforms (`uMin`, `uMax`, `uDimensions`) | `min`, `max`, or `dimensions` |

---

## GPU Architecture

### Values texture

- Format: `RGBA32F` (four 32-bit floats per texel)
- Layout: same folded 2D layout as the current frames texture — cell `i` maps to texel `(i % width, i / width)`
- Channel assignment: channels 0 through `dimensions - 1` are filled; remaining channels are 0
- Width: `min(cellCount, maxTextureDimension2D)`
- Height: `ceil(cellCount / width)`

### ColorMap texture

- Format: `RGBA8`, 256 × 1, nearest-neighbor sampling
- Index 0 corresponds to `min`, index 255 to `max`
- Default: linear black (0,0,0,255) → white (255,255,255,255) gradient

### Extension: `HealpixColorExtension`

Replaces `HealpixColorFramesExtension`. Injects into the `DECKGL_FILTER_COLOR` hook:

```glsl
ivec2 coord = ivec2(cellIndex % uValuesWidth, cellIndex / uValuesWidth);
vec4 vals = texelFetch(valuesTexture, coord, 0);

vec4 color;
if (uDimensions == 1) {
  float t = clamp((vals.r - uMin) / (uMax - uMin), 0.0, 1.0);
  color = texelFetch(colorMapTexture, ivec2(int(t * 255.0), 0), 0);

} else if (uDimensions == 2) {
  float t = clamp((vals.r - uMin) / (uMax - uMin), 0.0, 1.0);
  color = texelFetch(colorMapTexture, ivec2(int(t * 255.0), 0), 0);
  color.a *= vals.g;

} else if (uDimensions == 3) {
  color = vec4(vals.rgb, 1.0);

} else if (uDimensions == 4) {
  color = vals;

} else {
  color = vec4(0.0); // dimensions > 4: reserved
}

color.a *= opacity;
```

Uniforms declared in `healpix-color-shader-module.ts`:
- `uMin: f32`
- `uMax: f32`
- `uDimensions: i32`
- `uValuesWidth: i32`
- `valuesTexture: sampler2D`
- `colorMapTexture: sampler2D`

---

## File Changes

### Deleted
- `src/extensions/healpix-color-frames-extension.ts`
- `src/extensions/healpix-color-frames-shader-module.ts`

### Modified
- `src/types/layer-props.ts` — add `HealpixFrameObject`, update `HealpixCellsLayerProps`
- `src/layers/healpix-cells-layer.ts` — frame resolution, change detection, values + colorMap texture management
- `src/layers/healpix-cells-primitive-layer.ts` — swap to `HealpixColorExtension`, update prop types
- `src/index.ts` — remove old extension export, add new type exports

### New
- `src/extensions/healpix-color-extension.ts` — `HealpixColorExtension`
- `src/extensions/healpix-color-shader-module.ts` — uniforms + sampler declarations
- `src/utils/color-map.ts` — `DEFAULT_COLORMAP` constant, `validateColorMap()`, `buildColorMapTexture()`
- `src/utils/values-texture.ts` — `buildValuesTexture()`
- `src/utils/resolve-frame.ts` — `resolveFrame()` merge, defaults, and validation

### Tests
- `src/utils/resolve-frame.test.ts` — frame merge logic, defaults, validation errors
- `src/utils/color-map.test.ts` — default colorMap shape, validateColorMap error cases
- `src/utils/values-texture.test.ts` — channel packing for each dimension value
- `src/utils/color-frame.test.ts` — remove tests for deleted functionality

---

## Backward Compatibility

This is a **clean break**. The following props are removed:
- `colorFrames` (replaced by `frames[].values` + GPU color computation)
- `currentFrame` is reused but now indexes into `frames[]` rather than `colorFrames[]`

Existing consumers must migrate to the new `frames` / root-level values API.
