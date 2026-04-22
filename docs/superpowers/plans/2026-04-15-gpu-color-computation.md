# HEALPix GPU Color Computation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CPU pre-baked `colorFrames` API with a `frames` / `values` API that computes per-cell colors entirely on the GPU using a float values texture and a colorMap LUT.

**Architecture:** A new `resolveFrame()` utility merges root props with the current frame object and validates inputs. `packValuesData()` packs interleaved float values into an `RGBA32F` texture layout. A new `HealpixColorExtension` injects GLSL into `DECKGL_FILTER_COLOR` to sample the values texture and compute color via a 256×1 colorMap texture. The composite layer does smart change detection — only rebuilding geometry, values texture, or colorMap texture when their specific inputs change.

**Tech Stack:** TypeScript, deck.gl composite layer / extension pattern, luma.gl v9 textures, GLSL shader injection via `vs:#decl` + `vs:DECKGL_FILTER_COLOR`.

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `src/types/layer-props.ts` | Add `HealpixFrameObject`, update `HealpixCellsLayerProps` |
| Create | `src/utils/color-map.ts` | `DEFAULT_COLORMAP`, `validateColorMap()` |
| Create | `src/utils/color-map.test.ts` | Tests for color-map utilities |
| Create | `src/utils/resolve-frame.ts` | `resolveFrame()`, `ResolvedFrame` type |
| Create | `src/utils/resolve-frame.test.ts` | Tests for frame resolution |
| Create | `src/utils/values-texture.ts` | `packValuesData()` — pure CPU texture packing |
| Create | `src/utils/values-texture.test.ts` | Tests for values packing |
| Create | `src/extensions/healpix-color-shader-module.ts` | luma.gl shader module: scalar uniforms |
| Create | `src/extensions/healpix-color-extension.ts` | `HealpixColorExtension` + GLSL injection |
| Modify | `src/layers/healpix-cells-layer.ts` | Frame resolution, smart change detection, texture management |
| Modify | `src/index.ts` | Export `HealpixFrameObject`, remove `makeColorFrameFromValues` |
| Delete | `src/extensions/healpix-color-frames-extension.ts` | Replaced by `healpix-color-extension.ts` |
| Delete | `src/extensions/healpix-color-frames-shader-module.ts` | Replaced by `healpix-color-shader-module.ts` |

---

## Task 1: Update Types

**Files:**
- Modify: `src/types/layer-props.ts`

- [ ] **Step 1: Replace the file contents**

```typescript
import type { CompositeLayerProps } from '@deck.gl/core';
import type { CellIdArray } from './cell-ids';

/** HEALPix pixel numbering scheme. */
export type HealpixScheme = 'nest' | 'ring';

export type { CellIdArray };

/**
 * One animation frame of HEALPix cell data.
 *
 * All fields are optional — any field not set here falls back to the
 * matching root-level prop on `HealpixCellsLayerProps`.
 *
 * `values` is the only field with no root-level equivalent: it must be
 * present either here or at the root (via `HealpixCellsLayerProps.values`).
 *
 * ## `values` layout
 *
 * `values` is an interleaved flat array. Cell `i` occupies indices
 * `i * dimensions` through `i * dimensions + dimensions - 1`.
 *
 * ## `dimensions` interpretation
 *
 * | `dimensions` | Interpretation |
 * |---|---|
 * | `1` | Scalar → normalized through `[min, max]` → colorMap LUT → RGBA |
 * | `2` | Scalar (→ colorMap) + opacity multiplier (0–1) |
 * | `3` | Direct RGB in range 0–1; colorMap/min/max ignored; alpha = 1 |
 * | `4` | Direct RGBA in range 0–1; colorMap/min/max ignored |
 *
 * Values beyond 4 dimensions are reserved for future band math. Cells
 * with `dimensions > 4` render as transparent with a console warning.
 */
export type HealpixFrameObject = {
  /** Overrides root `nside`. */
  nside?: number;
  /** Overrides root `scheme`. Default: `'nest'`. */
  scheme?: HealpixScheme;
  /** Overrides root `cellIds`. */
  cellIds?: CellIdArray;
  /**
   * Per-cell float values. Interleaved: cell `i` starts at index
   * `i * dimensions`. Length must equal `cellIds.length * dimensions`.
   */
  values?: ArrayLike<number>;
  /** Overrides root `min`. Default: `0`. */
  min?: number;
  /** Overrides root `max`. Default: `1`. */
  max?: number;
  /**
   * Number of values per cell. Controls color computation mode.
   * Default: `1`. See type-level docs for full interpretation table.
   */
  dimensions?: 1 | 2 | 3 | 4;
  /**
   * ColorMap LUT: exactly 256 × 4 = 1024 RGBA bytes.
   * Index 0 maps to `min`, index 255 maps to `max`.
   * Default: linear black → white gradient.
   */
  colorMap?: Uint8Array;
};

/**
 * Props for `HealpixCellsLayer`.
 *
 * ## Single-frame usage
 *
 * Omit `frames` and set `nside`, `cellIds`, and `values` directly:
 *
 * ```tsx
 * <HealpixCellsLayer nside={512} cellIds={ids} values={vals} />
 * ```
 *
 * ## Multi-frame usage
 *
 * Root-level props act as shared defaults. Each `frames` entry overrides
 * selectively. Switch frames by updating `currentFrame`.
 *
 * ```tsx
 * <HealpixCellsLayer
 *   nside={512}
 *   colorMap={myLut}
 *   frames={[
 *     { cellIds: ids0, values: vals0 },
 *     { cellIds: ids1, values: vals1, min: -1 },
 *     { nside: 1024, cellIds: ids2, values: vals2 },
 *   ]}
 *   currentFrame={activeFrame}
 * />
 * ```
 *
 * ## Color dimensions
 *
 * See `HealpixFrameObject` for the full `dimensions` interpretation table.
 */
export type HealpixCellsLayerProps = {
  /**
   * HEALPix resolution parameter (power of 2, up to 262144).
   * Required at render time: set here or on every frame object.
   */
  nside: number;
  /**
   * HEALPix cell indices.
   * Required at render time: set here or on every frame object.
   */
  cellIds: CellIdArray;
  /** Numbering scheme. Default: `'nest'`. */
  scheme?: HealpixScheme;
  /**
   * Per-cell values for single-frame mode (when `frames` is absent).
   * Interleaved: cell `i` starts at index `i * dimensions`.
   * Length must equal `cellIds.length * dimensions`.
   */
  values?: ArrayLike<number>;
  /** Value mapped to colorMap index 0. Default: `0`. */
  min?: number;
  /** Value mapped to colorMap index 255. Default: `1`. */
  max?: number;
  /**
   * Number of values per cell. Default: `1`.
   * See `HealpixFrameObject` for the full interpretation table.
   */
  dimensions?: 1 | 2 | 3 | 4;
  /**
   * ColorMap LUT: exactly 256 × 4 = 1024 RGBA bytes (default: black→white).
   * Used as a shared default when frames do not provide their own colorMap.
   */
  colorMap?: Uint8Array;
  /** Animation frames. When absent, the layer renders a single frame from root props. */
  frames?: HealpixFrameObject[];
  /** Active frame index into `frames`. Clamped to `[0, frames.length - 1]`. Default: `0`. */
  currentFrame?: number;
} & CompositeLayerProps;
```

- [ ] **Step 2: Run tests — all 26 must still pass**

```bash
cd .worktrees/gpu-color-geometry && npm test
```

Expected: `26 passed`.

- [ ] **Step 3: Commit**

```bash
git add src/types/layer-props.ts
git commit -m "feat: add HealpixFrameObject and update HealpixCellsLayerProps for GPU color API"
```

---

## Task 2: `color-map` Utility

**Files:**
- Create: `src/utils/color-map.ts`
- Create: `src/utils/color-map.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `src/utils/color-map.test.ts`:

```typescript
import { DEFAULT_COLORMAP, validateColorMap } from './color-map';

describe('DEFAULT_COLORMAP', () => {
  it('is exactly 1024 bytes (256 × 4)', () => {
    expect(DEFAULT_COLORMAP.length).toBe(1024);
  });

  it('starts with black (0,0,0,255)', () => {
    expect(DEFAULT_COLORMAP[0]).toBe(0);
    expect(DEFAULT_COLORMAP[1]).toBe(0);
    expect(DEFAULT_COLORMAP[2]).toBe(0);
    expect(DEFAULT_COLORMAP[3]).toBe(255);
  });

  it('ends with white (255,255,255,255)', () => {
    expect(DEFAULT_COLORMAP[1020]).toBe(255);
    expect(DEFAULT_COLORMAP[1021]).toBe(255);
    expect(DEFAULT_COLORMAP[1022]).toBe(255);
    expect(DEFAULT_COLORMAP[1023]).toBe(255);
  });

  it('has a linear gray gradient', () => {
    expect(DEFAULT_COLORMAP[128 * 4 + 0]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 1]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 2]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 3]).toBe(255);
  });
});

describe('validateColorMap', () => {
  it('does not throw for exactly 1024 bytes', () => {
    expect(() => validateColorMap(new Uint8Array(1024))).not.toThrow();
  });

  it('throws for wrong length with a message mentioning 1024', () => {
    expect(() => validateColorMap(new Uint8Array(100))).toThrow('1024');
    expect(() => validateColorMap(new Uint8Array(0))).toThrow('1024');
    expect(() => validateColorMap(new Uint8Array(1025))).toThrow('1024');
  });
});
```

- [ ] **Step 2: Run to verify they fail**

```bash
npx jest src/utils/color-map.test.ts
```

Expected: `Cannot find module './color-map'`.

- [ ] **Step 3: Implement `src/utils/color-map.ts`**

```typescript
/**
 * Default colorMap: linear black (0,0,0,255) → white (255,255,255,255) gradient.
 * 256 entries × 4 bytes (RGBA) = 1024 bytes.
 */
export const DEFAULT_COLORMAP: Uint8Array = (() => {
  const map = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    map[i * 4 + 0] = i;
    map[i * 4 + 1] = i;
    map[i * 4 + 2] = i;
    map[i * 4 + 3] = 255;
  }
  return map;
})();

/**
 * Validates that `colorMap` is exactly 256 × 4 = 1024 bytes.
 * Throws with a descriptive message if not.
 */
export function validateColorMap(colorMap: Uint8Array): void {
  if (colorMap.length !== 1024) {
    throw new Error(
      `HealpixCellsLayer: colorMap must be exactly 256 × 4 = 1024 bytes, got ${colorMap.length}`
    );
  }
}
```

- [ ] **Step 4: Run tests — must pass**

```bash
npx jest src/utils/color-map.test.ts
```

Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/utils/color-map.ts src/utils/color-map.test.ts
git commit -m "feat: add DEFAULT_COLORMAP and validateColorMap utilities"
```

---

## Task 3: `resolve-frame` Utility

**Files:**
- Create: `src/utils/resolve-frame.ts`
- Create: `src/utils/resolve-frame.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `src/utils/resolve-frame.test.ts`:

```typescript
import { resolveFrame } from './resolve-frame';
import { DEFAULT_COLORMAP } from './color-map';
import type { HealpixCellsLayerProps } from '../types/layer-props';

const validIds = new Uint32Array([1, 2, 3]);
const validValues = new Float32Array([0.1, 0.2, 0.3]); // dim=1, 3 cells

function makeProps(overrides: Partial<HealpixCellsLayerProps>): HealpixCellsLayerProps {
  return { nside: 64, cellIds: validIds, values: validValues, ...overrides } as HealpixCellsLayerProps;
}

describe('resolveFrame — single-frame mode (no frames array)', () => {
  it('uses root props and applies defaults', () => {
    const result = resolveFrame(makeProps({}));
    expect(result.nside).toBe(64);
    expect(result.cellIds).toBe(validIds);
    expect(result.values).toBe(validValues);
    expect(result.scheme).toBe('nest');
    expect(result.min).toBe(0);
    expect(result.max).toBe(1);
    expect(result.dimensions).toBe(1);
    expect(result.colorMap).toBe(DEFAULT_COLORMAP);
  });

  it('uses explicit root overrides', () => {
    const myColorMap = new Uint8Array(1024);
    const result = resolveFrame(makeProps({
      scheme: 'ring',
      min: -1,
      max: 10,
      dimensions: 3,
      colorMap: myColorMap,
      values: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]), // 3 cells × 3 dims
    }));
    expect(result.scheme).toBe('ring');
    expect(result.min).toBe(-1);
    expect(result.max).toBe(10);
    expect(result.dimensions).toBe(3);
    expect(result.colorMap).toBe(myColorMap);
  });
});

describe('resolveFrame — multi-frame mode', () => {
  it('uses the current frame at currentFrame index', () => {
    const vals0 = new Float32Array([0.1, 0.2, 0.3]);
    const vals1 = new Float32Array([0.4, 0.5, 0.6]);
    const result = resolveFrame(makeProps({
      frames: [{ values: vals0 }, { values: vals1 }],
      currentFrame: 1,
    }));
    expect(result.values).toBe(vals1);
  });

  it('frame fields override root props', () => {
    const frameIds = new Uint32Array([9, 8]);
    const frameVals = new Float32Array([0.5, 0.6]);
    const result = resolveFrame(makeProps({
      frames: [{ cellIds: frameIds, values: frameVals, min: -5, max: 5 }],
      currentFrame: 0,
    }));
    expect(result.cellIds).toBe(frameIds);
    expect(result.values).toBe(frameVals);
    expect(result.min).toBe(-5);
    expect(result.max).toBe(5);
  });

  it('root props fill gaps not set on frame', () => {
    const myColorMap = new Uint8Array(1024);
    const result = resolveFrame(makeProps({
      colorMap: myColorMap,
      frames: [{ values: validValues }],
      currentFrame: 0,
    }));
    expect(result.colorMap).toBe(myColorMap);
    expect(result.nside).toBe(64); // from root
  });

  it('clamps currentFrame to [0, frames.length - 1]', () => {
    const vals0 = new Float32Array([0.1, 0.2, 0.3]);
    const vals1 = new Float32Array([0.4, 0.5, 0.6]);
    const props = makeProps({
      frames: [{ values: vals0 }, { values: vals1 }],
    });

    expect(resolveFrame({ ...props, currentFrame: 99 }).values).toBe(vals1);
    expect(resolveFrame({ ...props, currentFrame: -5 }).values).toBe(vals0);
  });

  it('empty frames array falls back to root props', () => {
    const result = resolveFrame(makeProps({ frames: [] }));
    expect(result.values).toBe(validValues);
  });
});

describe('resolveFrame — validation', () => {
  it('throws if nside is missing', () => {
    expect(() =>
      resolveFrame({ cellIds: validIds, values: validValues } as HealpixCellsLayerProps)
    ).toThrow(/nside/);
  });

  it('throws if cellIds is missing', () => {
    expect(() =>
      resolveFrame({ nside: 64, values: validValues } as HealpixCellsLayerProps)
    ).toThrow(/cellIds/);
  });

  it('throws if values is missing', () => {
    expect(() =>
      resolveFrame({ nside: 64, cellIds: validIds } as HealpixCellsLayerProps)
    ).toThrow(/values/);
  });

  it('throws if colorMap has wrong length', () => {
    expect(() =>
      resolveFrame(makeProps({ colorMap: new Uint8Array(100) }))
    ).toThrow(/1024/);
  });

  it('throws if values.length !== cellIds.length * dimensions', () => {
    expect(() =>
      resolveFrame(makeProps({ values: new Float32Array([0.1, 0.2]) })) // 2 values, 3 cells × dim=1
    ).toThrow(/values\.length/);
  });
});
```

- [ ] **Step 2: Run to verify they fail**

```bash
npx jest src/utils/resolve-frame.test.ts
```

Expected: `Cannot find module './resolve-frame'`.

- [ ] **Step 3: Implement `src/utils/resolve-frame.ts`**

```typescript
import type { HealpixCellsLayerProps, HealpixFrameObject } from '../types/layer-props';
import type { CellIdArray } from '../types/cell-ids';
import { DEFAULT_COLORMAP, validateColorMap } from './color-map';

/** The fully resolved, validated frame ready for GPU upload. */
export type ResolvedFrame = {
  nside: number;
  scheme: 'nest' | 'ring';
  cellIds: CellIdArray;
  values: ArrayLike<number>;
  min: number;
  max: number;
  dimensions: 1 | 2 | 3 | 4;
  colorMap: Uint8Array;
};

/**
 * Resolve the effective frame from layer props.
 *
 * When `props.frames` is set and non-empty, the frame at `props.currentFrame`
 * (clamped to valid range) is merged on top of root props. When `frames` is
 * absent or empty, root props are used directly.
 *
 * Throws with a descriptive message if required fields are missing or invalid.
 */
export function resolveFrame(props: HealpixCellsLayerProps): ResolvedFrame {
  // Determine which frame object to use (may be empty object for single-frame mode)
  let frame: Partial<HealpixFrameObject> = {};
  if (props.frames && props.frames.length > 0) {
    const idx = Math.max(
      0,
      Math.min(props.frames.length - 1, Math.floor((props.currentFrame ?? 0)))
    );
    frame = props.frames[idx] ?? {};
  }

  // nside
  const nside = frame.nside ?? props.nside;
  if (!nside) {
    throw new Error(
      'HealpixCellsLayer: nside is required — set it on the layer or on each frame object'
    );
  }

  // cellIds
  const cellIds = frame.cellIds ?? props.cellIds;
  if (!cellIds) {
    throw new Error(
      'HealpixCellsLayer: cellIds is required — set it on the layer or on each frame object'
    );
  }

  // values
  const values = frame.values ?? props.values;
  if (!values) {
    throw new Error(
      'HealpixCellsLayer: values is required — set it on the layer or on each frame object'
    );
  }

  // colorMap
  const colorMap = frame.colorMap ?? props.colorMap ?? DEFAULT_COLORMAP;
  validateColorMap(colorMap); // throws if wrong length

  const dimensions = (frame.dimensions ?? props.dimensions ?? 1) as 1 | 2 | 3 | 4;

  // values length check
  const expectedLen = cellIds.length * dimensions;
  if (values.length !== expectedLen) {
    throw new Error(
      `HealpixCellsLayer: values.length (${values.length}) must equal cellIds.length × dimensions (${cellIds.length} × ${dimensions} = ${expectedLen})`
    );
  }

  return {
    nside,
    scheme: frame.scheme ?? props.scheme ?? 'nest',
    cellIds: cellIds as CellIdArray,
    values,
    min: frame.min ?? props.min ?? 0,
    max: frame.max ?? props.max ?? 1,
    dimensions,
    colorMap,
  };
}
```

- [ ] **Step 4: Run tests — must pass**

```bash
npx jest src/utils/resolve-frame.test.ts
```

Expected: `14 passed`.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: all 26 original + 14 new = `40 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/utils/resolve-frame.ts src/utils/resolve-frame.test.ts
git commit -m "feat: add resolveFrame utility with frame merging and validation"
```

---

## Task 4: `values-texture` Utility

**Files:**
- Create: `src/utils/values-texture.ts`
- Create: `src/utils/values-texture.test.ts`

- [ ] **Step 1: Write the failing tests**

Create `src/utils/values-texture.test.ts`:

```typescript
import { packValuesData } from './values-texture';

const MAX = 4096;

describe('packValuesData', () => {
  it('dim=1: fills only R channel, G/B/A are 0', () => {
    const { data, width, height } = packValuesData([0.5], 1, 1, MAX);
    expect(width).toBe(1);
    expect(height).toBe(1);
    expect(data.length).toBe(4); // 1 texel × 4 floats
    expect(data[0]).toBeCloseTo(0.5); // R
    expect(data[1]).toBe(0);           // G
    expect(data[2]).toBe(0);           // B
    expect(data[3]).toBe(0);           // A
  });

  it('dim=2: fills R and G, B/A are 0', () => {
    const { data } = packValuesData([0.3, 0.7], 2, 1, MAX);
    expect(data[0]).toBeCloseTo(0.3);
    expect(data[1]).toBeCloseTo(0.7);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(0);
  });

  it('dim=3: fills R, G, B; A is 0', () => {
    const { data } = packValuesData([0.1, 0.2, 0.3], 3, 1, MAX);
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    expect(data[2]).toBeCloseTo(0.3);
    expect(data[3]).toBe(0);
  });

  it('dim=4: fills all channels', () => {
    const { data } = packValuesData([0.1, 0.2, 0.3, 0.4], 4, 1, MAX);
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    expect(data[2]).toBeCloseTo(0.3);
    expect(data[3]).toBeCloseTo(0.4);
  });

  it('two cells, dim=1: each in a separate texel', () => {
    const { data, width } = packValuesData([0.2, 0.8], 1, 2, MAX);
    expect(width).toBe(2);
    // cell 0: texel at x=0 → data[0]
    expect(data[0]).toBeCloseTo(0.2);
    // cell 1: texel at x=1 → data[4]
    expect(data[4]).toBeCloseTo(0.8);
  });

  it('multiple cells, dim=2: each cell gets its own texel', () => {
    const values = [0.1, 0.2, 0.3, 0.4]; // 2 cells × 2 dims
    const { data } = packValuesData(values, 2, 2, MAX);
    // cell 0 at texel 0: data[0]=0.1, data[1]=0.2
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    // cell 1 at texel 1 (offset 4): data[4]=0.3, data[5]=0.4
    expect(data[4]).toBeCloseTo(0.3);
    expect(data[5]).toBeCloseTo(0.4);
  });

  it('folds into 2D when cellCount > maxTextureSize', () => {
    // 5 cells, maxTextureSize=3 → width=3, height=2
    const values = [0.1, 0.2, 0.3, 0.4, 0.5];
    const { data, width, height } = packValuesData(values, 1, 5, 3);
    expect(width).toBe(3);
    expect(height).toBe(2);
    // cell 3: x=3%3=0, y=floor(3/3)=1, dstBase=(1*3+0)*4=12
    expect(data[12]).toBeCloseTo(0.4);
    // cell 4: x=4%3=1, y=floor(4/3)=1, dstBase=(1*3+1)*4=16
    expect(data[16]).toBeCloseTo(0.5);
  });

  it('returns 1×1 zero texel for cellCount=0', () => {
    const { data, width, height } = packValuesData([], 1, 0, MAX);
    expect(width).toBe(1);
    expect(height).toBe(1);
    expect(data.length).toBe(4);
    expect(data[0]).toBe(0);
  });
});
```

- [ ] **Step 2: Run to verify they fail**

```bash
npx jest src/utils/values-texture.test.ts
```

Expected: `Cannot find module './values-texture'`.

- [ ] **Step 3: Implement `src/utils/values-texture.ts`**

```typescript
/**
 * Packs per-cell interleaved float values into an `RGBA32F` 2D texture layout.
 *
 * The texture is folded: cell `i` maps to texel `(i % width, floor(i / width))`.
 * Each texel has 4 floats (RGBA32F). Channels 0 through `dimensions-1` are
 * filled; the rest remain 0.
 *
 * @param values      Interleaved float values. Length = cellCount × dimensions.
 * @param dimensions  Number of values per cell (1–4).
 * @param cellCount   Total number of cells.
 * @param maxTextureSize  GPU max texture dimension (from device limits).
 */
export function packValuesData(
  values: ArrayLike<number>,
  dimensions: number,
  cellCount: number,
  maxTextureSize: number
): { data: Float32Array; width: number; height: number } {
  if (cellCount === 0) {
    return { data: new Float32Array(4), width: 1, height: 1 };
  }

  const channelCount = Math.min(dimensions, 4);
  const width = Math.min(cellCount, maxTextureSize);
  const height = Math.ceil(cellCount / width);
  const data = new Float32Array(width * height * 4);

  for (let i = 0; i < cellCount; i++) {
    const x = i % width;
    const y = Math.floor(i / width);
    const dstBase = (y * width + x) * 4;
    const srcBase = i * dimensions;
    for (let d = 0; d < channelCount; d++) {
      data[dstBase + d] = (values as number[])[srcBase + d] ?? 0;
    }
  }

  return { data, width, height };
}
```

- [ ] **Step 4: Run tests — must pass**

```bash
npx jest src/utils/values-texture.test.ts
```

Expected: `9 passed`.

- [ ] **Step 5: Run full suite**

```bash
npm test
```

Expected: all 40 original + 9 new = `49 passed`.

- [ ] **Step 6: Commit**

```bash
git add src/utils/values-texture.ts src/utils/values-texture.test.ts
git commit -m "feat: add packValuesData utility for RGBA32F texture packing"
```

---

## Task 5: `healpix-color-shader-module.ts`

**Files:**
- Create: `src/extensions/healpix-color-shader-module.ts`

- [ ] **Step 1: Create the shader module**

```typescript
import type { Texture } from '@luma.gl/core';
import type { ShaderModule } from '@luma.gl/shadertools';

/**
 * Props consumed by the HEALPix color shader module.
 *
 * Scalar uniforms go into the `healpixColorUniforms` block.
 * Texture bindings (`healpixValuesTexture`, `healpixColorMapTexture`) are
 * declared separately in the `vs:#decl` injection in the extension.
 */
export type HealpixColorProps = {
  uMin: number;
  uMax: number;
  uDimensions: number;
  uValuesWidth: number;
  healpixValuesTexture: Texture;
  healpixColorMapTexture: Texture;
};

/**
 * Shader module for GPU color computation.
 *
 * Declares the `healpixColor` uniform block (scalar uniforms).
 * Textures are bound alongside these props via `model.shaderInputs.setProps`.
 */
export const healpixColorShaderModule = {
  name: 'healpixColor',
  vs: `\
uniform healpixColorUniforms {
  float uMin;
  float uMax;
  int uDimensions;
  int uValuesWidth;
} healpixColor;
`,
  uniformTypes: {
    uMin: 'f32',
    uMax: 'f32',
    uDimensions: 'i32',
    uValuesWidth: 'i32',
  }
} as const satisfies ShaderModule<HealpixColorProps>;
```

- [ ] **Step 2: Run full suite — must still pass**

```bash
npm test
```

Expected: `49 passed`.

- [ ] **Step 3: Commit**

```bash
git add src/extensions/healpix-color-shader-module.ts
git commit -m "feat: add healpixColor shader module for GPU color uniforms"
```

---

## Task 6: `HealpixColorExtension`

**Files:**
- Create: `src/extensions/healpix-color-extension.ts`

- [ ] **Step 1: Create the extension**

```typescript
import { Layer, LayerExtension, LayerProps } from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { healpixColorShaderModule } from './healpix-color-shader-module';

/** Extra props this extension reads from the host primitive layer. */
export type HealpixColorExtensionProps = LayerProps & {
  valuesTexture: Texture;
  colorMapTexture: Texture;
  uMin: number;
  uMax: number;
  uDimensions: number;
  uValuesWidth: number;
};

/**
 * GLSL declaration injected into the vertex shader.
 * Declares the two texture samplers used for color computation.
 */
const VERTEX_DECLARATION_INJECT = `
uniform mediump sampler2D healpixValuesTexture;
uniform mediump sampler2D healpixColorMapTexture;
`;

/**
 * GLSL injected into deck.gl's DECKGL_FILTER_COLOR hook.
 *
 * Samples per-cell float values from the values texture, then computes
 * the output color based on the dimensions mode:
 *
 *   dimensions=1  scalar → normalized through [min,max] → colorMap LUT
 *   dimensions=2  scalar (→ colorMap) + opacity multiplier (second value)
 *   dimensions=3  direct RGB in 0–1; alpha=1
 *   dimensions=4  direct RGBA in 0–1
 *   else          transparent (reserved for future band math)
 */
const VERTEX_COLOR_FILTER_INJECT = `
int healpixCell = gl_InstanceID;
int healpixX = healpixCell % healpixColor.uValuesWidth;
int healpixY = healpixCell / healpixColor.uValuesWidth;
vec4 healpixVals = texelFetch(healpixValuesTexture, ivec2(healpixX, healpixY), 0);

vec4 healpixOut;
if (healpixColor.uDimensions == 1) {
  float t = clamp(
    (healpixVals.r - healpixColor.uMin) / (healpixColor.uMax - healpixColor.uMin),
    0.0, 1.0
  );
  healpixOut = texelFetch(healpixColorMapTexture, ivec2(int(t * 255.0), 0), 0);
} else if (healpixColor.uDimensions == 2) {
  float t = clamp(
    (healpixVals.r - healpixColor.uMin) / (healpixColor.uMax - healpixColor.uMin),
    0.0, 1.0
  );
  healpixOut = texelFetch(healpixColorMapTexture, ivec2(int(t * 255.0), 0), 0);
  healpixOut.a *= healpixVals.g;
} else if (healpixColor.uDimensions == 3) {
  healpixOut = vec4(healpixVals.rgb, 1.0);
} else if (healpixColor.uDimensions == 4) {
  healpixOut = healpixVals;
} else {
  healpixOut = vec4(0.0);
}
color = vec4(healpixOut.rgb, healpixOut.a * layer.opacity);
`;

/**
 * Layer extension that computes HEALPix cell colors on the GPU.
 *
 * Reads per-cell float values from an RGBA32F texture and converts them
 * to RGBA using a 256×1 colorMap LUT texture, driven by the `dimensions` mode.
 * Replaces `HealpixColorFramesExtension`.
 */
class HealpixColorExtension extends LayerExtension {
  static extensionName = 'HealpixColorExtension';

  getShaders(): unknown {
    return {
      modules: [healpixColorShaderModule],
      inject: {
        'vs:#decl': VERTEX_DECLARATION_INJECT,
        'vs:DECKGL_FILTER_COLOR': VERTEX_COLOR_FILTER_INJECT
      }
    };
  }

  draw(
    this: Layer<HealpixColorExtensionProps>,
    _opts: { uniforms: unknown }
  ): void {
    const {
      valuesTexture,
      colorMapTexture,
      uMin,
      uMax,
      uDimensions,
      uValuesWidth
    } = this.props as HealpixColorExtensionProps;

    for (const model of this.getModels()) {
      model.shaderInputs.setProps({
        healpixColor: {
          uMin,
          uMax,
          uDimensions,
          uValuesWidth,
          healpixValuesTexture: valuesTexture,
          healpixColorMapTexture: colorMapTexture
        }
      });
    }
  }
}

/** Shared singleton used by all HEALPix sublayers. */
export const HEALPIX_COLOR_EXTENSION = new HealpixColorExtension();
```

- [ ] **Step 2: Run full suite — must still pass**

```bash
npm test
```

Expected: `49 passed`.

- [ ] **Step 3: Commit**

```bash
git add src/extensions/healpix-color-extension.ts
git commit -m "feat: add HealpixColorExtension for GPU value-based color computation"
```

---

## Task 7: Rewrite `HealpixCellsLayer`

**Files:**
- Modify: `src/layers/healpix-cells-layer.ts`

- [ ] **Step 1: Replace the file contents**

```typescript
/**
 * HealpixCellsLayer — render arbitrary HEALPix cells by ID with GPU color computation.
 *
 * This composite layer is responsible for:
 * - Resolving the effective frame from `frames[currentFrame]` merged with root props.
 * - Splitting cell IDs into GPU-friendly 32-bit halves.
 * - Building an RGBA32F values texture and an RGBA8 colorMap texture.
 * - Smart change detection: each resource is only rebuilt when its inputs change.
 * - Rendering a `HealpixCellsPrimitiveLayer` sublayer that computes colors on the GPU.
 */
import {
  CompositeLayer,
  DefaultProps,
  Layer,
  LayerExtension,
  UpdateParameters
} from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { splitCellIds } from '../utils/cell-id-split';
import { HealpixCellsPrimitiveLayer } from './healpix-cells-primitive-layer';
import { HEALPIX_COLOR_EXTENSION } from '../extensions/healpix-color-extension';
import { resolveFrame, type ResolvedFrame } from '../utils/resolve-frame';
import { packValuesData } from '../utils/values-texture';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixCellsLayerProps, HealpixFrameObject } from '../types/layer-props';

type _HealpixCellsLayerProps = {
  nside: number;
  cellIds: CellIdArray;
  scheme: 'nest' | 'ring';
  values: ArrayLike<number> | null;
  min: number;
  max: number;
  dimensions: 1 | 2 | 3 | 4;
  colorMap: Uint8Array | null;
  frames: HealpixFrameObject[] | null;
  currentFrame: number;
};

type HealpixCellsLayerState = {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
  valuesTexture: Texture | null;
  colorMapTexture: Texture | null;
  valuesTextureWidth: number;
  /** Last resolved frame — used for change detection in updateState. */
  prevResolved: ResolvedFrame | null;
};

const defaultProps: DefaultProps<_HealpixCellsLayerProps> = {
  nside: { type: 'number', value: 0 },
  cellIds: { type: 'object', value: new Uint32Array(0), compare: true },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  values: { type: 'object', value: null, compare: true },
  min: { type: 'number', value: 0 },
  max: { type: 'number', value: 1 },
  dimensions: { type: 'number', value: 1 },
  colorMap: { type: 'object', value: null, compare: true },
  frames: { type: 'object', value: null, compare: true },
  currentFrame: { type: 'number', value: 0 }
};

export class HealpixCellsLayer extends CompositeLayer<HealpixCellsLayerProps> {
  static layerName = 'HealpixCellsLayer';
  static defaultProps = defaultProps;

  declare state: HealpixCellsLayerState;

  initializeState(): void {
    this.setState({
      cellIdLo: new Uint32Array(0),
      cellIdHi: new Uint32Array(0),
      valuesTexture: null,
      colorMapTexture: null,
      valuesTextureWidth: 1,
      prevResolved: null
    });
    this._rebuildAll();
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return !!changeFlags.propsOrDataChanged;
  }

  updateState({ props }: UpdateParameters<this>): void {
    let resolved: ResolvedFrame;
    try {
      resolved = resolveFrame(props);
    } catch (e) {
      this.raiseError(e as Error, 'HealpixCellsLayer frame resolution failed');
      return;
    }

    const prev = this.state.prevResolved;

    const geometryChanged =
      !prev ||
      resolved.cellIds !== prev.cellIds ||
      resolved.nside !== prev.nside ||
      resolved.scheme !== prev.scheme;

    const valuesChanged =
      !prev ||
      resolved.values !== prev.values ||
      resolved.dimensions !== prev.dimensions ||
      resolved.cellIds.length !== prev.cellIds.length;

    const colorMapChanged = !prev || resolved.colorMap !== prev.colorMap;

    if (geometryChanged) this._splitCellIds(resolved.cellIds);
    if (valuesChanged || geometryChanged) this._updateValuesTexture(resolved);
    if (colorMapChanged) this._updateColorMapTexture(resolved);

    this.setState({ prevResolved: resolved });
  }

  finalizeState(): void {
    this.state.valuesTexture?.destroy();
    this.state.colorMapTexture?.destroy();
  }

  renderLayers(): Layer[] {
    const {
      cellIdLo,
      cellIdHi,
      valuesTexture,
      colorMapTexture,
      valuesTextureWidth,
      prevResolved
    } = this.state;

    if (!prevResolved || !valuesTexture || !colorMapTexture) return [];

    const { nside, scheme, cellIds, min, max, dimensions } = prevResolved;
    const count = cellIds.length;
    if (count === 0) return [];

    return [
      new HealpixCellsPrimitiveLayer(
        this.getSubLayerProps({
          id: 'cells',
          nside,
          scheme,
          instanceCount: count,
          data: {
            length: count,
            attributes: {
              cellIdLo: { value: cellIdLo, size: 1 },
              cellIdHi: { value: cellIdHi, size: 1 }
            }
          },
          valuesTexture,
          colorMapTexture,
          uMin: min,
          uMax: max,
          uDimensions: dimensions,
          uValuesWidth: valuesTextureWidth,
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_EXTENSION
          ]
        })
      )
    ];
  }

  /** Rebuild all resources from scratch (called on first init). */
  private _rebuildAll(): void {
    let resolved: ResolvedFrame;
    try {
      resolved = resolveFrame(this.props);
    } catch (e) {
      this.raiseError(e as Error, 'HealpixCellsLayer frame resolution failed');
      return;
    }
    this._splitCellIds(resolved.cellIds);
    this._updateValuesTexture(resolved);
    this._updateColorMapTexture(resolved);
    this.setState({ prevResolved: resolved });
  }

  private _splitCellIds(cellIds: CellIdArray): void {
    if (!cellIds?.length) {
      this.setState({
        cellIdLo: new Uint32Array(0),
        cellIdHi: new Uint32Array(0)
      });
      return;
    }
    const { lo, hi } = splitCellIds(cellIds);
    this.setState({ cellIdLo: lo, cellIdHi: hi });
  }

  /**
   * Build and upload an RGBA32F values texture.
   *
   * Each texel stores the float values for one cell in channels 0–(dimensions-1).
   * The texture is folded: cell i → texel (i % width, floor(i / width)).
   */
  private _updateValuesTexture(resolved: ResolvedFrame): void {
    const { values, dimensions, cellIds } = resolved;
    const cellCount = cellIds.length;
    const oldTexture = this.state.valuesTexture;

    const maxTextureSize = this.context.device.limits.maxTextureDimension2D;
    const { data, width, height } = packValuesData(
      values,
      dimensions,
      cellCount,
      maxTextureSize
    );

    const texture = this.context.device.createTexture({
      id: `${this.id}-values`,
      width,
      height,
      dimension: '2d',
      format: 'rgba32float',
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        mipmapFilter: 'none',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    texture.copyImageData({ data });

    this.setState({ valuesTexture: texture, valuesTextureWidth: width });
    oldTexture?.destroy();
  }

  /**
   * Build and upload an RGBA8 256×1 colorMap texture.
   *
   * Index 0 maps to min, index 255 maps to max.
   */
  private _updateColorMapTexture(resolved: ResolvedFrame): void {
    const { colorMap } = resolved;
    const oldTexture = this.state.colorMapTexture;

    const texture = this.context.device.createTexture({
      id: `${this.id}-colormap`,
      width: 256,
      height: 1,
      dimension: '2d',
      format: 'rgba8unorm',
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        mipmapFilter: 'none',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    texture.copyImageData({ data: colorMap });

    this.setState({ colorMapTexture: texture });
    oldTexture?.destroy();
  }
}
```

- [ ] **Step 2: Run full suite — must still pass**

```bash
npm test
```

Expected: `49 passed`. TypeScript must compile without errors (`npx tsc --noEmit`).

- [ ] **Step 3: Commit**

```bash
git add src/layers/healpix-cells-layer.ts
git commit -m "feat: rewrite HealpixCellsLayer with GPU color computation and smart change detection"
```

---

## Task 8: Cleanup — Exports and Delete Old Files

**Files:**
- Modify: `src/index.ts`
- Delete: `src/extensions/healpix-color-frames-extension.ts`
- Delete: `src/extensions/healpix-color-frames-shader-module.ts`

- [ ] **Step 1: Update `src/index.ts`**

```typescript
export { HealpixCellsLayer } from './layers/healpix-cells-layer';
export type { CellIdArray } from './types/cell-ids';
export type {
  HealpixCellsLayerProps,
  HealpixFrameObject,
  HealpixScheme
} from './types/layer-props';
```

Note: `makeColorFrameFromValues` is intentionally removed (clean break API). The `color-frame.ts` utility file and its tests remain for internal use.

- [ ] **Step 2: Delete old extension files**

```bash
rm src/extensions/healpix-color-frames-extension.ts
rm src/extensions/healpix-color-frames-shader-module.ts
```

- [ ] **Step 3: Run full suite — must still pass**

```bash
npm test
```

Expected: `49 passed`.

- [ ] **Step 4: TypeScript check**

```bash
npx tsc --noEmit
```

Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add src/index.ts
git rm src/extensions/healpix-color-frames-extension.ts
git rm src/extensions/healpix-color-frames-shader-module.ts
git commit -m "feat: update exports and remove old color frames extension (clean break)"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** Types ✓, resolveFrame ✓, DEFAULT_COLORMAP ✓, validateColorMap ✓, packValuesData ✓, shader module ✓, extension GLSL ✓, layer change detection ✓, geometry/values/colorMap textures ✓, opacity ✓, dimensions 1-4 ✓, dimensions>4 transparent+warning (handled in extension GLSL else branch) ✓, multi-frame frame switching ✓, single-frame mode ✓, smart resource rebuild ✓, clean break exports ✓
- [x] **No placeholders**
- [x] **Type consistency:** `ResolvedFrame` defined in Task 3, used in Tasks 7 and 8. `packValuesData` signature matches test calls. `HEALPIX_COLOR_EXTENSION` matches import in layer.
