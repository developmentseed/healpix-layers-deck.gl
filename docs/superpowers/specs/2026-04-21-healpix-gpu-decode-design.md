# GPU-side HEALPix NEST/RING decode + shader modularization

**Date:** 2026-04-21
**Status:** Design — awaiting implementation
**Goal:** Move `nest2fxy` and `ring2fxy` from CPU to GPU so `HealpixCellsLayer` uploads raw cell IDs instead of pre-decomposed `(face, ix, iy)`. Reorganize the shader into focused modules so each file has one responsibility.

---

## Scope

### In scope

- Replace `decomposeCellIds` (CPU, calls `healpix-ts`) with a trivial CPU ID-split and a GPU-side NEST/RING decoder.
- Split `healpix-corners.glsl.ts` (currently 264 lines, mixing fp64 primitives, attribute declarations, and the vertex `main`) into focused GLSL modules assembled at the shader-build site.
- Update the two existing GPU readback pages and add a third for RING; move them from `tmp/` to `test/gpu/`.
- Document how to run the GPU readback tests.

### Out of scope

- Texture packing / `_buildTextureData` remains on CPU. The `texture.copyImageData` redundant-copy cleanup is deferred to a separate change.
- Corner-expansion math (`fxy2tu → tu2za`) and the fp64 Dekker primitives are **unchanged semantically**; only their file placement moves.
- Support for `nside > 2^24`. The current maximum (`face << 24 | ix`) continues to be the ceiling.

---

## Problem

After the GPU-corners refactor, the `HealpixCellsLayer` pipeline looks like:

```
cellIds → decomposeCellIds (CPU, healpix-ts) → faceIx, iy
         → vertex shader (corner math, fp64) → lon/lat → project
```

The CPU decompose step is the only remaining per-cell-update CPU work besides color-texture packing. It costs a JS loop at O(N) over `cellIds` on every `cellIds`/`nside`/`scheme` change, plus a runtime dependency on `healpix-ts`'s `nest2fxy` / `ring2fxy`. The shader file itself has grown to 264 lines and mixes three unrelated concerns (fp64 primitives, HEALPix math, vertex glue) in a single string literal.

---

## Solution overview

```
cellIds → splitCellIds (CPU, 6 lines of pure JS) → cellIdLo, cellIdHi
        → vertex shader:
            decode (NEST or RING) → (face, ix, iy)
            → corner math (fxy2tu, tu2za, Dekker) → lon/lat
            → project
```

The CPU step drops from "call `healpix-ts` per cell" to "split a 53-bit integer into two u32 halves per cell". For `Uint32Array` / `Int32Array` inputs (the common case at `nside ≤ 2^14`, where max cell id `12·nside²` fits in 32 bits) the split degenerates to a zero-copy buffer alias plus a shared zero buffer.

The shader is split across six files under `src/shaders/`, each with a single responsibility and explicit dependencies.

---

## Architecture

### Attribute layout

Two GLSL `uint` attributes, composed in `main()` into a `uvec2`:

```glsl
in uint cellIdLo;
in uint cellIdHi;
// main():
uvec2 cellId = uvec2(cellIdLo, cellIdHi);
```

The layer supplies each as a standard deck.gl instanced attribute (size 1, `uint32`). Two attributes rather than one because deck.gl's attribute manager doesn't expose a clean path to a single-attribute `uvec2`.

**Zero-copy path.** When the user's `cellIds` is a `Uint32Array` or `Int32Array`, the layer constructs `cellIdLo` as a `Uint32Array` view over the same `ArrayBuffer` (no byte copy; `Int32Array`'s bit pattern is identical under u32 reinterpretation for non-negative values, and cell ids are always non-negative). `cellIdHi` uses a shared zero-filled `Uint32Array` sized to the maximum instance count seen on the layer. All instances read identical `cellIdHi` values, but deck.gl still expects a buffer — the shared buffer satisfies that.

**Split path.** When `cellIds` is `Float64Array` or `Float32Array`, the layer runs:

```ts
export function splitCellIds(cellIds: CellIdArray): {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
} {
  const n = cellIds.length;
  const lo = new Uint32Array(n);
  const hi = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const id = cellIds[i];
    lo[i] = id >>> 0;
    hi[i] = Math.floor(id / 4294967296);
  }
  return { cellIdLo: lo, cellIdHi: hi };
}
```

This replaces `decomposeCellIds` entirely. No `healpix-ts` import in the composite layer.

### `healpixCells` shader module — new uniforms

```glsl
uniform healpixCellsUniforms {
  uint nside;
  uint log2nside;
  uint scheme;       // 0 = nest, 1 = ring
  uvec2 polarLim;    // 2·nside·(nside - 1)  — only used by RING
  uvec2 eqLim;       // polarLim + 8·nside²  — only used by RING
  uvec2 npix;        // 12·nside²            — only used by RING
} healpixCells;
```

All values are computed once per draw call on the CPU side of the shader module (`getUniforms()` in the module definition). `polarLim`/`eqLim`/`npix` are uvec2 to avoid 64-bit integer arithmetic in the shader for per-draw constants; each is built by a small `splitU53(x: number) => [lo, hi]` helper that mirrors `splitCellIds` for a single scalar. (luma.gl's shader-module `uniformTypes` supports `uvec2` directly; if a future luma version disallows it we fall back to two `uint` uniforms.)

### Shader file layout

Under `src/shaders/`:

```
fp64.glsl.ts                  # Dekker primitives, constants (PI64, PI2_64, PI4_64)
int64.glsl.ts                 # uvec2 helpers (add, sub, mul32, shr, shl, and, lt, div32, isqrt)
healpix-decompose.glsl.ts     # decodeNest, decodeRing, compact1By1
healpix-corners.glsl.ts       # fxyCorner(face, cx, cy, nside, out lon_fp, out lat_fp)
healpix-cells.vs.glsl.ts      # vertex shader header + main()
healpix-cells.fs.glsl.ts      # fragment shader
healpix-cells-shader-module.ts  # existing file, gains the new uniforms
index.ts                      # assembles & exports HEALPIX_VERTEX_SHADER / HEALPIX_FRAGMENT_SHADER
```

**Dependency rules:**

- `int64.glsl.ts` — no deps.
- `fp64.glsl.ts` — no deps.
- `healpix-decompose.glsl.ts` — depends on `int64` only. Pure integer; no fp64.
- `healpix-corners.glsl.ts` — depends on `fp64` only. `fxy2tu` / `tu2za` in fp64; no uvec2 work.
- `healpix-cells.vs.glsl.ts` — composes all of the above. Contains `main()` plus attribute declarations.

**Assembly.** `shaders/index.ts` concatenates in dependency order:

```ts
const VS_HEADER = `#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;
in uint cellIdLo;
in uint cellIdHi;
in vec3 positions;
out vec4 vColor;
`;

export const HEALPIX_VERTEX_SHADER: string = [
  VS_HEADER,
  INT64_GLSL,
  FP64_GLSL,
  HEALPIX_DECOMPOSE_GLSL,
  HEALPIX_CORNERS_GLSL,
  HEALPIX_CELLS_VS_MAIN
].join('\n');

export const HEALPIX_FRAGMENT_SHADER: string = HEALPIX_CELLS_FS;
```

Each leaf file exports exactly one string constant with a `/* glsl */` comment marker for editor syntax highlighting, and contains only GLSL — no TypeScript logic, no conditional includes.

---

## GPU decode algorithms

### `int64.glsl.ts` — uvec2 helpers

Nine functions, each 5–15 lines of pure integer GLSL:

```glsl
uvec2 u64_add(uvec2 a, uvec2 b);        // carry = lessThan(sum.x, a.x)
uvec2 u64_sub(uvec2 a, uvec2 b);        // borrow = lessThan(a.x, b.x)
uvec2 u64_mul32(uint a, uint b);        // 32×32 → 64, via 16-bit Knuth split
uvec2 u64_mul32_u64(uvec2 a, uint b);   // low 64 bits of (uvec2)·u32
uvec2 u64_shr(uvec2 v, uint s);         // 0 ≤ s < 64
uvec2 u64_shl(uvec2 v, uint s);
uvec2 u64_and(uvec2 a, uvec2 b);
bool  u64_lt(uvec2 a, uvec2 b);
uint  u64_div32(uvec2 a, uint d, out uint rem);
uint  u64_isqrt(uvec2 a);
```

`u64_shr` / `u64_shl` keep shift amounts within per-half `[0, 32)` ranges (shifts by ≥ 32 are undefined in GLSL on some drivers) by splitting into a cross-half move and a sub-32 shift, selected branchlessly with `mix`.

`u64_div32` computes the quotient as `floor(float(a.hi) * 2^32 + float(a.lo)) / float(d))`, reifies as `uint`, and runs a correction loop using `u64_mul32(q, d)` and `u64_lt` to verify `q·d ≤ a < (q+1)·d`. For `d ≤ 2^26` and `a ≤ 2^52` the fp32 seed is off by at most ±2, so the loop terminates in ≤ 2 iterations. The output is bit-exact.

`u64_isqrt` seeds with `floor(sqrt(float_from_uvec2(a)))`, then corrects until `i² ≤ a < (i+1)²` using `u64_mul32(i, i)`. For `a ≤ 2^52` the fp32 sqrt is off by at most ±4; ≤ 4 iterations. Bit-exact.

### `healpix-decompose.glsl.ts` — NEST and RING decode

`decodeNest(uvec2 cellId, uint log2n) → uvec3(face, ix, iy)`

Mirrors `nest2fxy`:

```
k          = 2u * log2n
mask       = u64_sub(u64_shl(uvec2(1u, 0u), k), uvec2(1u, 0u))
nestInFace = u64_and(cellId, mask)
face       = u64_shr(cellId, k).x       // face < 12, fits in u32
ix         = compact1By1(nestInFace.x) | (compact1By1(nestInFace.y) << 16u)
iy         = compact1By1(nestInFace.x >> 1u) | (compact1By1(nestInFace.y >> 1u) << 16u)
```

`compact1By1(uint w)` is the 6-line parallel-bit-compact from Morton code literature:

```glsl
uint compact1By1(uint w) {
  w &= 0x55555555u;
  w = (w | (w >>  1u)) & 0x33333333u;
  w = (w | (w >>  2u)) & 0x0F0F0F0Fu;
  w = (w | (w >>  4u)) & 0x00FF00FFu;
  w = (w | (w >>  8u)) & 0x0000FFFFu;
  return w;
}
```

**Precision:** bit-exact for every valid `(nside ∈ [1, 2^24], cellId)`. Pure integer.

`decodeRing(uvec2 cellId, uint nside, uvec2 polarLim, uvec2 eqLim, uvec2 npix) → uvec3(face, ix, iy)`

Direct port of `ring2fxy` with three branches.

**North polar cap** (`u64_lt(cellId, polarLim)`):

```
i     = (u64_isqrt(u64_add(u64_shl(cellId, 1u), uvec2(1u, 0u))) + 1u) / 2u
jFull = u64_sub(cellId, u64_mul32(2u, i * (i - 1u)))
j     = jFull.x                        // j < 4·i ≤ 4·nside, fits in u32
f     = j / i
k     = j - f * i
x     = nside - i + k
y     = nside - 1u - k
```

`i ≤ nside ≤ 2^24`, so `i·(i-1)` needs uvec2 (via `u64_mul32`).

**Equatorial belt** (`u64_lt(cellId, eqLim)`):

```
kFull = u64_sub(cellId, polarLim)       // ≤ 8·nside² ≤ 2^51
ring  = 4u * nside
q     = u64_div32(kFull, ring, kmod)    // q ≤ 2·nside, fp32 seed
i     = nside - q
s     = 1u - (i & 1u)
j     = 2u * kmod + s                   // ≤ 8·nside
// remainder of body is pure u32, mirrors healpix-ts ring2fxy equatorial branch:
int jj = int(j) - 4 * int(nside);
int ii = int(i) + 5 * int(nside) - 1;
uint pp = uint((ii + jj) >> 1);
uint qq = uint((ii - jj) >> 1);
uint PP = pp / nside;
uint QQ = qq / nside;
uint V  = 5u - (PP + QQ);
uint H  = PP - QQ + 4u;
face    = 4u * V + ((H >> 1u) & 3u);
ix      = pp - PP * nside;
iy      = qq - QQ * nside;
```

**South polar cap** (else):

```
p   = u64_sub(u64_sub(npix, cellId), uvec2(1u, 0u))
// mirror of north; f = 11 - (j / i), x = i - k - 1, y = k
```

**Precision:** bit-exact, matching `ring2fxy` output byte-for-byte for every valid `(nside, cellId)`. The fp32-seeded `u64_div32` and `u64_isqrt` are bit-exact by construction of their correction loops.

### `healpix-corners.glsl.ts` — `fxyCorner` function

The current `main()` body from the corner-selection step onward is extracted into a named function:

```glsl
void fxyCorner(int face, int cx, int cy, int nside,
               out vec2 lon_rad_fp, out vec2 lat_rad_fp) {
  // existing f_row / f1 / f2 computation
  // existing k_ring wrap
  // existing equatorial / polar / pole branches
  // writes outputs
}
```

No algorithmic change. The `_seal` barriers stay inside the function (their job was to defeat the GLSL optimizer, and function-boundary inlining preserves the barriers).

### `healpix-cells.vs.glsl.ts` — `main()`

```glsl
void main() {
  uvec2 cellId = uvec2(cellIdLo, cellIdHi);

  uvec3 fxy;
  if (healpixCells.scheme == 0u) {
    fxy = decodeNest(cellId, healpixCells.log2nside);
  } else {
    fxy = decodeRing(cellId, healpixCells.nside,
                     healpixCells.polarLim, healpixCells.eqLim, healpixCells.npix);
  }
  int face = int(fxy.x), ix = int(fxy.y), iy = int(fxy.z);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  vec2 lon_rad_fp, lat_rad_fp;
  fxyCorner(face, cx, cy, int(healpixCells.nside), lon_rad_fp, lat_rad_fp);

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);
  gl_Position = project_position_to_clipspace(
    vec3(lon_deg_fp.x, lat_deg_fp.x, 0.0),
    vec3(lon_deg_fp.y, lat_deg_fp.y, 0.0),
    vec3(0.0), geometry.position
  );

  vColor = vec4(1.0);
  DECKGL_FILTER_COLOR(vColor, geometry);
}
```

The `scheme` branch is uniform-controlled; all vertices in a draw take the same path. No warp divergence cost.

---

## Layer-side changes

### `HealpixCellsLayer` (composite)

- Delete `_decomposeCellIds` + `state.faceIx` / `state.iy`.
- Add `_splitCellIds`: computes `cellIdLo` / `cellIdHi` from `props.cellIds`. For `Uint32Array` inputs, aliases the source into `cellIdLo` and uses a shared zero buffer for `cellIdHi`.
- Sublayer data attributes change from `{ faceIx, instIy }` to `{ cellIdLo, cellIdHi }`.
- Forward new shader module props (`log2nside`, `scheme`, `polarLim`, `eqLim`, `npix`) from the composite layer to the primitive layer.

### `HealpixCellsPrimitiveLayer`

- Attribute declarations: `faceIx` / `instIy` → `cellIdLo` / `cellIdHi` (both `size: 1`, `type: 'uint32'`).
- Shader imports from `./shaders` (barrel) instead of the specific `healpix-corners.glsl` file.
- No behavior change beyond the attribute rename.

### `healpix-cells-shader-module.ts`

- Extend `HealpixCellsProps` and `uniformTypes` with `log2nside`, `scheme`, `polarLim`, `eqLim`, `npix`.
- Add `getUniforms({ nside, scheme }): { ... }` that computes the uvec2 constants using the `splitU53` helper (same split formula as `splitCellIds`).

### Files deleted

- `src/utils/decompose-cell-ids.ts`
- `src/utils/decompose-cell-ids.test.ts`

### Files added

- `src/utils/split-cell-ids.ts`
- `src/utils/split-cell-ids.test.ts`
- `src/shaders/int64.glsl.ts`
- `src/shaders/fp64.glsl.ts`
- `src/shaders/healpix-decompose.glsl.ts`
- `src/shaders/healpix-cells.vs.glsl.ts`
- `src/shaders/healpix-cells.fs.glsl.ts`
- `src/shaders/index.ts`
- `src/shaders/__tests__/gpu-decode-reference.ts`
- `src/shaders/__tests__/gpu-decode-reference.test.ts`

### Files modified

- `src/shaders/healpix-corners.glsl.ts` — stripped down to `fxyCorner` function only
- `src/shaders/healpix-cells-shader-module.ts` — new uniforms
- `src/layers/healpix-cells-layer.ts` — composite layer pipeline update
- `src/layers/healpix-cells-primitive-layer.ts` — attribute rename

### Files moved

- `tmp/gpu-readback.html` → `test/gpu/gpu-readback-nest-equatorial.html`
- `tmp/gpu-readback-polar.html` → `test/gpu/gpu-readback-nest-polar.html`
- `tmp/inspect-cell.mjs` → `test/gpu/inspect-cell.mjs`
- `tmp/compute-truth.mjs` → `test/gpu/compute-truth.mjs`

### Files added under `test/gpu/`

- `gpu-readback-ring.html` — RING cell readback page, covers the three RING decode branches.
- `README.md` — how-to-run documentation.

---

## Testing

### (a) Jest unit tests

- `split-cell-ids.test.ts` — round-trip for `Uint32Array`, `Int32Array`, `Float32Array`, `Float64Array` at `nside ∈ {1, 2^15, 2^15 + 1, 2^20, 2^24}`, including boundary ids (0, max cell id at each nside, ring-boundary neighborhoods).

### (b) Jest reference-port tests

`src/shaders/__tests__/gpu-decode-reference.ts` is a JS translation of `decodeNest` and `decodeRing` that uses two-element `Uint32Array` views to emulate uvec2 and mirrors the shader's integer correction loops. Tested against `healpix-ts` `nest2fxy` / `ring2fxy` on:

- All 12 faces × 4 corners at `nside ∈ {1, 2, 4, 256, 2^12, 2^15, 2^16, 2^20, 2^24}`
- Ring-boundary cells: `polar_lim - 1`, `polar_lim`, `polar_lim + 1`, `eq_lim - 1`, `eq_lim`, `npix - 1` at several nside values
- 10 000 random `(nside, id)` pairs with a fixed RNG seed

This layer catches any algorithmic bug in the port before any GPU is involved. When it passes, the GLSL is a faithful transcription of a known-correct reference.

### (c) GPU readback tests under `test/gpu/`

Three browser pages, each loads a tiny shader that includes the exact production GLSL (`int64` + `fp64` + `healpix-decompose` + `healpix-corners`), runs one or more transform-feedback draws, and diffs the output against CPU truth.

- `gpu-readback-nest-equatorial.html` — one NEST cell in the equatorial band at `nside = 262144`. Emits `(face, ix, iy, lon_hi, lon_lo, lat_hi, lat_lo)` — so **one page covers both decode and corner precision**.
- `gpu-readback-nest-polar.html` — same, polar cell.
- `gpu-readback-ring.html` — three RING cells (north cap, equatorial, south cap) at `nside = 262144`.
- `compute-truth.mjs` and `inspect-cell.mjs` — updated to print `(cellIdLo, cellIdHi)` alongside `(f, x, y)` and to generate TRUTH arrays suitable for pasting into the HTML pages.

### (d) Visual comparison

Manual: run the demo page on `main` and on the feature branch, side-by-side at `nside ∈ {64, 1024, 262144}` for both `nest` and `ring`. No diff in rendered output is the acceptance criterion.

### `test/gpu/README.md`

Content outline:

```
# HEALPix GPU pipeline tests

The `test/gpu/` folder contains manual validation scripts that the
Jest suite can't replicate (GPU transform feedback, ULP diffs against
fp64 truth).

## Node truth scripts

    node test/gpu/compute-truth.mjs
    node test/gpu/inspect-cell.mjs <cellId>

Output: CPU truth `(face, ix, iy, lon, lat)` for the given cell,
formatted to paste into a readback page's TRUTH constant.

## Browser readback

    npx http-server .
    open http://localhost:8080/test/gpu/gpu-readback-nest-equatorial.html
    open http://localhost:8080/test/gpu/gpu-readback-nest-polar.html
    open http://localhost:8080/test/gpu/gpu-readback-ring.html

Each page runs transform feedback on a production-equivalent shader,
reads back 32-bit results, and diffs against CPU truth. Legend:

    green  ≤ 0.5 ULP   (bit-exact)
    amber  ≤ 4 ULP     (GLSL spec-bound for asin/sin/cos)
    red    > 4 ULP     (pipeline bug)

## Updating truth

When changing the shader in a way that alters output:

    node test/gpu/compute-truth.mjs <cellId>

Paste the printed TRUTH array into the HTML page's TRUTH constant.
```

---

## Risks

| Risk | Mitigation |
|---|---|
| `u64_div32` fp32 seed off by > 2 | Unbounded integer correction loop; Jest reference port detects any off-by-N before GPU. |
| `u64_isqrt` fp32 seed off by > 4 near `nside = 2^24` polar caps | Unbounded integer correction loop with monotonic convergence. |
| WebGL2 driver oddities on `shl`/`shr` by 32+ | `u64_shl` / `u64_shr` never shift by ≥ 32 per half; cross-half movement is branchless via `mix`. Readback suite covers `nside = 2^24`. |
| Scheme branch divergence | Scheme is a uniform (layer prop). All vertices in a draw take the same path. Zero cost. |
| Attribute aliasing mutation | For `Uint32Array` inputs, `cellIdLo` aliases the source. deck.gl's `compare: true` catches mutation via identity; document that mutating the input buffer in place is unsupported. |
| Corner math regression from wrapping as function | `_seal` barriers stay inside `fxyCorner`. Readback tests catch any numerical drift directly. |

---

## Implementation order (single PR)

The work is small enough to ship as one change, but internally ordered so each step compiles and tests pass:

1. Add `int64.glsl.ts` + the uvec2 helpers, standalone.
2. Add the Jest reference-port test file and implement `decodeNest` reference; get it passing against `healpix-ts`.
3. Implement `decodeRing` reference; get it passing.
4. Write `healpix-decompose.glsl.ts` (shader version mirroring the reference).
5. Split `healpix-corners.glsl.ts` into `fp64.glsl.ts` + `healpix-corners.glsl.ts` (fxyCorner function); keep existing readback tests green.
6. Create `shaders/index.ts`, `healpix-cells.vs.glsl.ts`, `healpix-cells.fs.glsl.ts`.
7. Update `healpix-cells-shader-module.ts` with new uniforms + `getUniforms`.
8. Add `splitCellIds` util + tests; delete `decomposeCellIds`.
9. Update `HealpixCellsLayer` and `HealpixCellsPrimitiveLayer` to the new attribute names and sublayer props.
10. Update the existing two readback pages to the new shader composition and move them to `test/gpu/`; add the RING readback page and README.

Each step is a self-contained commit.
