# HEALPix GPU NEST/RING decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `nest2fxy` and `ring2fxy` from CPU to GPU; reorganize the monolithic 264-line corner shader into focused GLSL modules.

**Architecture:** A tiny CPU ID-split replaces `decomposeCellIds`. The vertex shader receives `uvec2` cell IDs and a `scheme` uniform, runs the decoder on-GPU, then feeds `(face, ix, iy)` into the existing `fxyCorner` corner-expansion math. Shader is assembled from six focused files (`int64`, `fp64`, `healpix-decompose`, `healpix-corners`, `healpix-cells.vs`, `healpix-cells.fs`) via string concatenation in `shaders/index.ts`. A Jest-only JS reference mirrors the GLSL integer ops and is tested against `healpix-ts` before the shader is touched; GPU transform-feedback pages under `test/gpu/` catch regressions in the live pipeline.

**Tech Stack:** TypeScript, Jest, deck.gl, luma.gl, WebGL2, `healpix-ts` (CPU reference only). Tests: Jest unit + browser transform-feedback HTML pages.

---

## Spec reference

See `docs/superpowers/specs/2026-04-21-healpix-gpu-decode-design.md`. Section numbers cited below refer to that file.

## File map

**Created:**
- `src/shaders/__tests__/gpu-decode-reference.ts` — JS port of GLSL integer ops + decoders (test-only utility)
- `src/shaders/__tests__/gpu-decode-reference.test.ts` — tests it against `healpix-ts`
- `src/shaders/int64.glsl.ts`, `src/shaders/fp64.glsl.ts`, `src/shaders/healpix-decompose.glsl.ts`
- `src/shaders/healpix-cells.vs.glsl.ts`, `src/shaders/healpix-cells.fs.glsl.ts`, `src/shaders/index.ts`
- `src/utils/split-cell-ids.ts`, `src/utils/split-cell-ids.test.ts`
- `test/gpu/gpu-readback-ring.html`, `test/gpu/README.md`

**Modified:**
- `src/shaders/healpix-corners.glsl.ts` — slimmed to `fxyCorner(...)` function only
- `src/shaders/healpix-cells-shader-module.ts` — new uniforms + `getUniforms`
- `src/layers/healpix-cells-layer.ts` — swaps decompose for split, forwards new uniforms
- `src/layers/healpix-cells-primitive-layer.ts` — attribute rename + shader import path

**Deleted:**
- `src/utils/decompose-cell-ids.ts`
- `src/utils/decompose-cell-ids.test.ts`

**Moved:**
- `tmp/gpu-readback.html` → `test/gpu/gpu-readback-nest-equatorial.html`
- `tmp/gpu-readback-polar.html` → `test/gpu/gpu-readback-nest-polar.html`
- `tmp/inspect-cell.mjs` → `test/gpu/inspect-cell.mjs`
- `tmp/compute-truth.mjs` → `test/gpu/compute-truth.mjs`

---

## Task 1: JS `u64_*` helpers + tests (Jest)

**Files:**
- Create: `src/shaders/__tests__/gpu-decode-reference.ts`
- Create: `src/shaders/__tests__/gpu-decode-reference.test.ts`

These helpers mirror the GLSL `int64.glsl.ts` functions byte-for-byte. They represent a `uvec2` as a tuple `[lo, hi]` of JS numbers (both `>>> 0`-canonicalised u32s). All multi-word ops follow the exact structure the GLSL version will have, so the shader is a mechanical transcription.

- [ ] **Step 1: Write the failing test file**

```ts
// src/shaders/__tests__/gpu-decode-reference.test.ts
import {
  u64_add,
  u64_sub,
  u64_mul32,
  u64_shr,
  u64_shl,
  u64_and,
  u64_lt,
  u64_div32,
  u64_isqrt,
  toBig,
  fromBig
} from './gpu-decode-reference';

describe('u64_* helpers', () => {
  describe('roundtrip', () => {
    it('toBig / fromBig roundtrip on boundaries', () => {
      const cases = [0n, 1n, TWO32 - 1n, TWO32, TWO32 + 1n, (1n << 52n) - 1n];
      for (const x of cases) {
        expect(toBig(fromBig(x))).toBe(x);
      }
    });
  });

  describe('u64_add', () => {
    it('no carry', () => {
      expect(u64_add([1, 0], [2, 0])).toEqual([3, 0]);
    });
    it('carry into hi', () => {
      expect(u64_add([0xffffffff, 0], [1, 0])).toEqual([0, 1]);
    });
    it('full 64-bit add, no overflow', () => {
      const a = fromBig(0xdeadbeefcafef00dn);
      const b = fromBig(0x0000000112345678n);
      expect(toBig(u64_add(a, b))).toBe(0xdeadbef0df237685n);
    });
  });

  describe('u64_sub', () => {
    it('no borrow', () => {
      expect(u64_sub([5, 0], [2, 0])).toEqual([3, 0]);
    });
    it('borrow from hi', () => {
      expect(u64_sub([0, 1], [1, 0])).toEqual([0xffffffff, 0]);
    });
    it('full 64-bit sub', () => {
      const a = fromBig(0x100000000n);
      const b = fromBig(1n);
      expect(toBig(u64_sub(a, b))).toBe(0xffffffffn);
    });
  });

  describe('u64_mul32', () => {
    it('small * small', () => {
      expect(u64_mul32(3, 7)).toEqual([21, 0]);
    });
    it('big * big overflows into hi', () => {
      expect(toBig(u64_mul32(0xffffffff, 0xffffffff))).toBe(
        0xfffffffe00000001n
      );
    });
    it('2^16 * 2^16 = 2^32', () => {
      expect(toBig(u64_mul32(0x10000, 0x10000))).toBe(0x100000000n);
    });
  });

  describe('u64_shr', () => {
    it('shift 0 is identity', () => {
      expect(u64_shr([0xdeadbeef, 0xcafef00d], 0)).toEqual([
        0xdeadbeef, 0xcafef00d
      ]);
    });
    it('shift 4 within lo', () => {
      expect(toBig(u64_shr(fromBig(0x123456789abcdef0n), 4))).toBe(
        0x0123456789abcdefn
      );
    });
    it('shift 32 moves hi to lo', () => {
      expect(u64_shr([0xdeadbeef, 0xcafef00d], 32)).toEqual([0xcafef00d, 0]);
    });
    it('shift 48 ', () => {
      expect(toBig(u64_shr(fromBig(0x0001234567890000n), 48))).toBe(0x123n);
    });
    it('shift 63', () => {
      expect(u64_shr([0, 0x80000000], 63)).toEqual([1, 0]);
    });
  });

  describe('u64_shl', () => {
    it('shift 32 moves lo to hi', () => {
      expect(u64_shl([0xdeadbeef, 0], 32)).toEqual([0, 0xdeadbeef]);
    });
    it('shift 1 with cross-half carry', () => {
      expect(u64_shl([0x80000000, 0], 1)).toEqual([0, 1]);
    });
    it('shift 48', () => {
      expect(toBig(u64_shl(fromBig(0x123n), 48))).toBe(0x0123000000000000n);
    });
  });

  describe('u64_and', () => {
    it('mask lower 24 bits', () => {
      const v = fromBig(0xdeadbeefcafef00dn);
      const mask = fromBig(0xffffffn);
      expect(toBig(u64_and(v, mask))).toBe(0xfef00dn);
    });
  });

  describe('u64_lt', () => {
    it('hi dominates', () => {
      expect(u64_lt([100, 1], [1, 2])).toBe(true);
    });
    it('tie on hi, lo decides', () => {
      expect(u64_lt([1, 5], [2, 5])).toBe(true);
      expect(u64_lt([5, 5], [2, 5])).toBe(false);
    });
    it('equal → false', () => {
      expect(u64_lt([5, 5], [5, 5])).toBe(false);
    });
  });

  describe('u64_div32', () => {
    it('exact division', () => {
      const out = u64_div32(fromBig(1000n), 10);
      expect(out.q).toBe(100);
      expect(out.r).toBe(0);
    });
    it('with remainder', () => {
      const out = u64_div32(fromBig(1003n), 10);
      expect(out.q).toBe(100);
      expect(out.r).toBe(3);
    });
    it('large dividend', () => {
      const n = (1n << 50n) + 12345n;
      const d = 4_000_003;
      const out = u64_div32(fromBig(n), d);
      const q = n / BigInt(d);
      const r = n - q * BigInt(d);
      expect(BigInt(out.q)).toBe(q);
      expect(BigInt(out.r)).toBe(r);
    });
    it('k=2^51, d=2^26 (equatorial worst case)', () => {
      const n = 1n << 51n;
      const d = 1 << 26;
      const out = u64_div32(fromBig(n), d);
      expect(BigInt(out.q)).toBe(n / BigInt(d));
      expect(out.r).toBe(0);
    });
  });

  describe('u64_isqrt', () => {
    it('square numbers', () => {
      expect(u64_isqrt(fromBig(0n))).toBe(0);
      expect(u64_isqrt(fromBig(1n))).toBe(1);
      expect(u64_isqrt(fromBig(4n))).toBe(2);
      expect(u64_isqrt(fromBig(9n))).toBe(3);
      expect(u64_isqrt(fromBig(100000000n))).toBe(10000);
    });
    it('non-square rounds down', () => {
      expect(u64_isqrt(fromBig(2n))).toBe(1);
      expect(u64_isqrt(fromBig(3n))).toBe(1);
      expect(u64_isqrt(fromBig(8n))).toBe(2);
      expect(u64_isqrt(fromBig(99n))).toBe(9);
    });
    it('large value near 2^49 (polar cap worst case)', () => {
      const n = (1n << 49n) - 7n;
      const expected = 23726566n; // floor(sqrt(2^49 - 7))
      expect(BigInt(u64_isqrt(fromBig(n)))).toBe(expected);
    });
    it('2^52 - 1 (upper end of safe-integer domain)', () => {
      const n = (1n << 52n) - 1n;
      const q = BigInt(u64_isqrt(fromBig(n)));
      expect(q * q <= n).toBe(true);
      expect((q + 1n) * (q + 1n) > n).toBe(true);
    });
  });
});
```

- [ ] **Step 2: Run the test and confirm failure**

Run: `npm test -- gpu-decode-reference`
Expected: FAIL — the reference module doesn't exist yet.

- [ ] **Step 3: Implement the JS reference helpers**

```ts
// src/shaders/__tests__/gpu-decode-reference.ts
/**
 * Pure-JS mirror of src/shaders/int64.glsl.ts + src/shaders/healpix-decompose.glsl.ts.
 *
 * Every function is structured to match its GLSL counterpart line-for-line,
 * so the shader is a mechanical transcription. Uvec2 is a [lo, hi] tuple of
 * u32-canonical JS numbers. Hot path avoids BigInt to keep Jest fast.
 */

export type U64 = readonly [number, number];

const TWO32 = 4294967296;

export function fromBig(x: bigint): U64 {
  return [Number(x & 0xffffffffn), Number((x >> 32n) & 0xffffffffn)];
}

export function toBig(v: U64): bigint {
  return (BigInt(v[1]) << 32n) | BigInt(v[0]);
}

export function u64_add(a: U64, b: U64): U64 {
  const loSum = a[0] + b[0];
  const lo = loSum >>> 0;
  const carry = loSum >= TWO32 ? 1 : 0;
  const hi = (a[1] + b[1] + carry) >>> 0;
  return [lo, hi];
}

export function u64_sub(a: U64, b: U64): U64 {
  const lo = (a[0] - b[0]) >>> 0;
  const borrow = a[0] < b[0] ? 1 : 0;
  const hi = (a[1] - b[1] - borrow) >>> 0;
  return [lo, hi];
}

export function u64_mul32(a: number, b: number): U64 {
  const aLo = a & 0xffff;
  const aHi = (a >>> 16) & 0xffff;
  const bLo = b & 0xffff;
  const bHi = (b >>> 16) & 0xffff;
  const p0 = aLo * bLo;
  const p1 = aLo * bHi;
  const p2 = aHi * bLo;
  const p3 = aHi * bHi;
  const mid = p1 + p2;
  const midCarry = mid > 0xffffffff ? 0x10000 : 0;
  const loFull = p0 + ((mid & 0xffff) << 16);
  const lo = loFull >>> 0;
  const loCarry = loFull >= TWO32 ? 1 : 0;
  const hi = (p3 + ((mid >>> 16) & 0xffff) + midCarry + loCarry) >>> 0;
  return [lo, hi];
}

export function u64_shr(v: U64, s: number): U64 {
  if (s === 0) return v;
  if (s >= 32) {
    const s2 = s - 32;
    const lo = s2 === 0 ? v[1] : (v[1] >>> s2);
    return [lo >>> 0, 0];
  }
  const lo = ((v[0] >>> s) | (v[1] << (32 - s))) >>> 0;
  const hi = (v[1] >>> s) >>> 0;
  return [lo, hi];
}

export function u64_shl(v: U64, s: number): U64 {
  if (s === 0) return v;
  if (s >= 32) {
    const s2 = s - 32;
    const hi = s2 === 0 ? v[0] : (v[0] << s2);
    return [0, hi >>> 0];
  }
  const lo = (v[0] << s) >>> 0;
  const hi = ((v[1] << s) | (v[0] >>> (32 - s))) >>> 0;
  return [lo, hi];
}

export function u64_and(a: U64, b: U64): U64 {
  return [(a[0] & b[0]) >>> 0, (a[1] & b[1]) >>> 0];
}

export function u64_lt(a: U64, b: U64): boolean {
  if (a[1] !== b[1]) return a[1] < b[1];
  return a[0] < b[0];
}

/**
 * 64 / 32 -> (quotient, remainder). fp32 seed + integer correction loop.
 * Quotient must fit in u32; caller guarantees a < d * 2^32.
 */
export function u64_div32(a: U64, d: number): { q: number; r: number } {
  const fa = a[1] * TWO32 + a[0];
  let q = Math.floor(fa / d) >>> 0;
  let qd = u64_mul32(q, d);
  while (u64_lt(a, qd)) {
    q = (q - 1) >>> 0;
    qd = u64_sub(qd, [d, 0]);
  }
  let dp1 = u64_add(qd, [d, 0]);
  while (!u64_lt(a, dp1)) {
    q = (q + 1) >>> 0;
    qd = dp1;
    dp1 = u64_add(qd, [d, 0]);
  }
  const rFull = u64_sub(a, qd);
  return { q, r: rFull[0] >>> 0 };
}

/** floor(sqrt(a)) for a < 2^52. fp32-ish seed + integer correction loop. */
export function u64_isqrt(a: U64): number {
  if (a[0] === 0 && a[1] === 0) return 0;
  const fa = a[1] * TWO32 + a[0];
  let i = Math.floor(Math.sqrt(fa)) >>> 0;
  let ii = u64_mul32(i, i);
  while (u64_lt(a, ii)) {
    i = (i - 1) >>> 0;
    ii = u64_mul32(i, i);
  }
  let next = u64_mul32(i + 1, i + 1);
  while (!u64_lt(a, next)) {
    i = (i + 1) >>> 0;
    ii = next;
    next = u64_mul32(i + 1, i + 1);
  }
  return i;
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `npm test -- gpu-decode-reference`
Expected: PASS, all `u64_*` describe blocks green.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/__tests__/gpu-decode-reference.ts \
        src/shaders/__tests__/gpu-decode-reference.test.ts
git commit -m "test: add JS uvec2 helpers mirroring future GLSL int64.glsl.ts"
```

---

## Task 2: JS `decodeNest` reference + tests

**Files:**
- Modify: `src/shaders/__tests__/gpu-decode-reference.ts`
- Modify: `src/shaders/__tests__/gpu-decode-reference.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/shaders/__tests__/gpu-decode-reference.test.ts`:

```ts
import { nest2fxy } from 'healpix-ts';
import { decodeNest, compact1By1, fromBig } from './gpu-decode-reference';

describe('compact1By1', () => {
  it('extracts even bits from a 32-bit word', () => {
    expect(compact1By1(0b10101010)).toBe(0b1111);
    expect(compact1By1(0x55555555)).toBe(0xffff);
    expect(compact1By1(0xaaaaaaaa)).toBe(0);
  });
});

describe('decodeNest', () => {
  const NSIDES = [1, 2, 4, 8, 256, 1 << 12, 1 << 15, 1 << 16, 1 << 20, 1 << 24];

  for (const nside of NSIDES) {
    it(`matches nest2fxy across 12 faces at nside=${nside}`, () => {
      const log2n = Math.log2(nside);
      const nside2 = BigInt(nside) * BigInt(nside);
      for (let face = 0; face < 12; face++) {
        const ids = [
          0n,
          nside2 - 1n,
          nside2 / 2n,
          (nside2 * 3n) / 7n
        ].map((k) => BigInt(face) * nside2 + k);
        for (const id of ids) {
          const truth = nest2fxy(nside, Number(id));
          const cellId = fromBig(id);
          const got = decodeNest(cellId, log2n);
          expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
        }
      }
    });
  }

  it('random NEST ids at nside=2^24', () => {
    const nside = 1 << 24;
    const log2n = 24;
    const nside2 = BigInt(nside) * BigInt(nside);
    let state = 0x9e3779b97f4a7c15n;
    const rand = () => {
      state = (state * 6364136223846793005n + 1442695040888963407n) &
              0xffffffffffffffffn;
      return state;
    };
    for (let trial = 0; trial < 200; trial++) {
      const f = Number(rand() % 12n);
      const k = rand() % nside2;
      const id = BigInt(f) * nside2 + k;
      const truth = nest2fxy(nside, Number(id));
      const got = decodeNest(fromBig(id), log2n);
      expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
    }
  });
});
```

- [ ] **Step 2: Run test to verify failure**

Run: `npm test -- gpu-decode-reference`
Expected: FAIL — `decodeNest` / `compact1By1` don't exist.

- [ ] **Step 3: Implement `compact1By1` and `decodeNest`**

Append to `src/shaders/__tests__/gpu-decode-reference.ts`:

```ts
/** Extract even-positioned bits from a 32-bit word and pack them into bits 0..15. */
export function compact1By1(w: number): number {
  w = w & 0x55555555;
  w = (w | (w >>> 1)) & 0x33333333;
  w = (w | (w >>> 2)) & 0x0f0f0f0f;
  w = (w | (w >>> 4)) & 0x00ff00ff;
  w = (w | (w >>> 8)) & 0x0000ffff;
  return w >>> 0;
}

export type DecodeResult = { face: number; ix: number; iy: number };

export function decodeNest(cellId: U64, log2n: number): DecodeResult {
  const k = 2 * log2n;
  const one: U64 = [1, 0];
  const mask = u64_sub(u64_shl(one, k), one);
  const nestInFace = u64_and(cellId, mask);
  const face = u64_shr(cellId, k)[0];
  const ix =
    (compact1By1(nestInFace[0]) | (compact1By1(nestInFace[1]) << 16)) >>> 0;
  const iy =
    (compact1By1(nestInFace[0] >>> 1) |
      (compact1By1(nestInFace[1] >>> 1) << 16)) >>>
    0;
  return { face, ix, iy };
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `npm test -- gpu-decode-reference`
Expected: PASS, all NEST tests green across nside ∈ [1, 2^24].

- [ ] **Step 5: Commit**

```bash
git add src/shaders/__tests__/gpu-decode-reference.ts \
        src/shaders/__tests__/gpu-decode-reference.test.ts
git commit -m "test: add JS decodeNest reference matching healpix-ts nest2fxy"
```

---

## Task 3: JS `decodeRing` reference + tests

**Files:**
- Modify: `src/shaders/__tests__/gpu-decode-reference.ts`
- Modify: `src/shaders/__tests__/gpu-decode-reference.test.ts`

- [ ] **Step 1: Write the failing test**

Append to `src/shaders/__tests__/gpu-decode-reference.test.ts`:

```ts
import { ring2fxy } from 'healpix-ts';
import { decodeRing, ringUniforms, fromBig } from './gpu-decode-reference';

describe('decodeRing', () => {
  const NSIDES_SMALL = [1, 2, 4, 8, 256, 1 << 12];
  const NSIDES_LARGE = [1 << 15, 1 << 16, 1 << 20, 1 << 24];

  function checkRange(nside: number, ids: bigint[]): void {
    const u = ringUniforms(nside);
    for (const id of ids) {
      const truth = ring2fxy(nside, Number(id));
      const got = decodeRing(fromBig(id), nside, u.polarLim, u.eqLim, u.npix);
      expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
    }
  }

  for (const nside of NSIDES_SMALL) {
    it(`matches ring2fxy exhaustively at nside=${nside}`, () => {
      const npix = 12 * nside * nside;
      const ids: bigint[] = [];
      for (let i = 0; i < npix; i++) ids.push(BigInt(i));
      checkRange(nside, ids);
    });
  }

  for (const nside of NSIDES_LARGE) {
    it(`matches ring2fxy at boundaries, nside=${nside}`, () => {
      const n = BigInt(nside);
      const polar = 2n * n * (n - 1n);
      const eq = polar + 8n * n * n;
      const npix = 12n * n * n;
      const ids = [
        0n,
        1n,
        polar - 1n,
        polar,
        polar + 1n,
        polar + 4n * n,
        eq - 1n,
        eq,
        eq + 1n,
        npix - 1n
      ];
      checkRange(nside, ids);
    });

    it(`matches ring2fxy on 200 random ids, nside=${nside}`, () => {
      const n = BigInt(nside);
      const npix = 12n * n * n;
      let state = (BigInt(nside) * 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
      const rand = () => {
        state = (state * 6364136223846793005n + 1442695040888963407n) &
                0xffffffffffffffffn;
        return state;
      };
      const ids: bigint[] = [];
      for (let i = 0; i < 200; i++) ids.push(rand() % npix);
      checkRange(nside, ids);
    });
  }
});
```

- [ ] **Step 2: Run test to verify failure**

Run: `npm test -- gpu-decode-reference`
Expected: FAIL — `decodeRing` / `ringUniforms` don't exist.

- [ ] **Step 3: Implement `ringUniforms` and `decodeRing`**

Append to `src/shaders/__tests__/gpu-decode-reference.ts`:

```ts
export function ringUniforms(nside: number): {
  polarLim: U64;
  eqLim: U64;
  npix: U64;
} {
  const n = BigInt(nside);
  const polar = 2n * n * (n - 1n);
  const eq = polar + 8n * n * n;
  const npix = 12n * n * n;
  return {
    polarLim: fromBig(polar),
    eqLim: fromBig(eq),
    npix: fromBig(npix)
  };
}

export function decodeRing(
  cellId: U64,
  nside: number,
  polarLim: U64,
  eqLim: U64,
  npix: U64
): DecodeResult {
  if (u64_lt(cellId, polarLim)) {
    return decodeRingNorth(cellId, nside);
  }
  if (u64_lt(cellId, eqLim)) {
    return decodeRingEquatorial(cellId, nside, polarLim);
  }
  return decodeRingSouth(cellId, nside, npix);
}

function decodeRingNorth(cellId: U64, nside: number): DecodeResult {
  // i = (isqrt(1 + 2p) + 1) / 2
  const onePlus2p = u64_add(u64_shl(cellId, 1), [1, 0]);
  const root = u64_isqrt(onePlus2p);
  const i = ((root + 1) >>> 1) >>> 0;
  // j = p - 2*i*(i-1)       (< 4i, fits u32)
  const i2 = u64_mul32(2, i * (i - 1) >>> 0);
  const jFull = u64_sub(cellId, i2);
  const j = jFull[0] >>> 0;
  const f = Math.floor(j / i);
  const k = j - f * i;
  const ix = (nside - i + k) >>> 0;
  const iy = (nside - 1 - k) >>> 0;
  return { face: f, ix, iy };
}

function decodeRingEquatorial(
  cellId: U64,
  nside: number,
  polarLim: U64
): DecodeResult {
  const kFull = u64_sub(cellId, polarLim);
  const ring = (4 * nside) >>> 0;
  const { q, r: kmod } = u64_div32(kFull, ring);
  const i = (nside - q) >>> 0;
  const s = 1 - (i & 1);
  const j = (2 * kmod + s) >>> 0;
  // All u32 from here, mirrors healpix-ts ring2fxy equatorial branch
  const jj = j - 4 * nside;
  const ii = i + 5 * nside - 1;
  const pp = (ii + jj) >> 1;
  const qq = (ii - jj) >> 1;
  const PP = Math.floor(pp / nside);
  const QQ = Math.floor(qq / nside);
  const V = 5 - (PP + QQ);
  const H = PP - QQ + 4;
  const face = (4 * V + ((H >> 1) & 3)) >>> 0;
  const ix = (pp - PP * nside) >>> 0;
  const iy = (qq - QQ * nside) >>> 0;
  return { face, ix, iy };
}

function decodeRingSouth(cellId: U64, nside: number, npix: U64): DecodeResult {
  // p = npix - cellId - 1
  const p = u64_sub(u64_sub(npix, cellId), [1, 0]);
  const onePlus2p = u64_add(u64_shl(p, 1), [1, 0]);
  const root = u64_isqrt(onePlus2p);
  const i = ((root + 1) >>> 1) >>> 0;
  const i2 = u64_mul32(2, i * (i - 1) >>> 0);
  const jFull = u64_sub(p, i2);
  const j = jFull[0] >>> 0;
  const f = (11 - Math.floor(j / i)) >>> 0;
  const k = j - Math.floor(j / i) * i;
  const ix = (i - k - 1) >>> 0;
  const iy = k >>> 0;
  return { face: f, ix, iy };
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `npm test -- gpu-decode-reference`
Expected: PASS. All RING tests green including exhaustive sweep at nside ≤ 2^12 (3,145,728 ids at nside=2^12) and boundary + random sweeps at nside up to 2^24.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/__tests__/gpu-decode-reference.ts \
        src/shaders/__tests__/gpu-decode-reference.test.ts
git commit -m "test: add JS decodeRing reference matching healpix-ts ring2fxy"
```

---

## Task 4: Extract `fp64.glsl.ts` from the corner shader

**Files:**
- Create: `src/shaders/fp64.glsl.ts`
- Modify: `src/shaders/healpix-corners.glsl.ts:1-94` (remove fp64 primitives)

This is a pure refactor — no behavior change. The existing `tmp/gpu-readback.html` and `tmp/gpu-readback-polar.html` continue to pass bit-for-bit.

- [ ] **Step 1: Create `src/shaders/fp64.glsl.ts`**

```ts
// src/shaders/fp64.glsl.ts
/**
 * Dekker / double-single arithmetic for GLSL.
 *
 * Each "fp64" value is a vec2 (hi, lo) with hi = rounded fp32, lo = residual.
 *
 * CRITICAL: Dekker primitives depend on strict fp32 round-to-nearest semantics.
 * The GLSL optimizer can (and WILL) algebraically simplify these to return
 * lo = 0. Round-tripping load-bearing intermediates through
 * uintBitsToFloat ∘ floatBitsToUint gives the compiler an opaque identity
 * that breaks algebraic simplification chains.
 */
export const FP64_GLSL: string = /* glsl */ `
float _seal(float x) { return uintBitsToFloat(floatBitsToUint(x)); }

vec2 _split(float a) {
  const float SPLIT = 4097.0;
  float t = _seal(a * SPLIT);
  float hi = t - (t - a);
  float lo = a - hi;
  return vec2(hi, lo);
}

vec2 _twoSum(float a, float b) {
  float s  = _seal(a + b);
  float bb = _seal(s - a);
  float err = (a - (s - bb)) + (b - bb);
  return vec2(s, err);
}

vec2 _qts(float a, float b) {
  float s = _seal(a + b);
  float err = b - (s - a);
  return vec2(s, err);
}

vec2 _twoProd(float a, float b) {
  float p = _seal(a * b);
  vec2 ap = _split(a);
  vec2 bp = _split(b);
  float err = ((ap.x * bp.x - p) + ap.x * bp.y + ap.y * bp.x) + ap.y * bp.y;
  return vec2(p, err);
}

vec2 _add64(vec2 a, vec2 b) {
  vec2 s = _twoSum(a.x, b.x);
  vec2 t = _twoSum(a.y, b.y);
  s.y += t.x;
  s = _qts(s.x, s.y);
  s.y += t.y;
  return _qts(s.x, s.y);
}

vec2 _sub64(vec2 a, vec2 b) { return _add64(a, vec2(-b.x, -b.y)); }

vec2 _mul64(vec2 a, vec2 b) {
  vec2 p = _twoProd(a.x, b.x);
  p.y += a.x * b.y + a.y * b.x;
  return _qts(p.x, p.y);
}

vec2 _div64(vec2 a, vec2 b) {
  float xn = 1.0 / b.x;
  vec2 yn  = a * xn;
  float diff = _sub64(a, _mul64(b, yn)).x;
  vec2 corr  = _twoProd(xn, diff);
  return _add64(yn, corr);
}

vec2 _mul64f(vec2 a, float b) { return _mul64(a, vec2(b, 0.0)); }

const vec2 PI64   = vec2( 3.1415927,  -8.742278e-8 );
const vec2 PI2_64 = vec2( 1.5707964,  -4.371139e-8 );
const vec2 PI4_64 = vec2( 0.78539819, -2.1855695e-8);
`;
```

- [ ] **Step 2: Modify `src/shaders/healpix-corners.glsl.ts` — delete fp64 primitives, import `FP64_GLSL`**

Replace lines 1–94 of the existing file with:

```ts
import { FP64_GLSL } from './fp64.glsl';

export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

in uint faceIx;
in uint instIy;
in vec3 positions;

out vec4 vColor;

${FP64_GLSL}

// ---------------------------------------------------------------------------
void main() {
` + /* existing main() body starting from `int face = int(faceIx >> 24u);` */ `
`;
```

Concretely: keep the existing `main()` body verbatim from line 96 onward. The only change is that the fp64 primitives (old lines 27–94) are replaced by the `${FP64_GLSL}` interpolation, and the leading `\` on the template is removed so the interpolation works.

Full replacement file:

```ts
import { FP64_GLSL } from './fp64.glsl';

export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

in uint faceIx;
in uint instIy;
in vec3 positions;

out vec4 vColor;

${FP64_GLSL}

// ---------------------------------------------------------------------------
void main() {
  int face = int(faceIx >> 24u);
  int ix   = int(faceIx & 0xFFFFFFu);
  int iy   = int(instIy);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  int f_row = face / 4;
  int f1 = f_row + 2;
  int f2 = 2 * (face - 4 * f_row) - (f_row & 1) + 1;
  int nside = int(healpixCells.nside);

  int i_ring = f1 * nside - cx - cy;
  int k_ring = f2 * nside + (cx - cy) + 8 * nside;

  int period = 8 * nside;
  k_ring = k_ring - (k_ring / period) * period;
  if (k_ring > 4 * nside) k_ring -= period;

  int k_int = k_ring / nside;
  int k_rem = k_ring - k_int * nside;
  int i_int = i_ring / nside;
  int i_rem = i_ring - i_int * nside;
  float k_frac = float(k_rem) / float(nside);
  float i_frac = float(i_rem) / float(nside);

  vec2 t_fp = _mul64f(PI4_64, float(k_int));
  t_fp      = _add64(t_fp, _mul64f(PI4_64, k_frac));

  vec2 i_ang = _mul64f(PI4_64, float(i_int));
  i_ang      = _add64(i_ang, _mul64f(PI4_64, i_frac));
  vec2 u_fp  = _sub64(PI2_64, i_ang);

  float u_hi  = u_fp.x;
  float abs_u = abs(u_hi);

  vec2 lat_rad_fp, lon_rad_fp;
  if (abs_u >= PI2_64.x) {
    float sgn = sign(u_hi);
    lat_rad_fp = vec2(sgn * PI2_64.x, sgn * PI2_64.y);
    lon_rad_fp = vec2(0.0);
  } else if (abs_u <= PI4_64.x) {
    vec2 three_pi  = _mul64f(PI64, 3.0);
    vec2 k_eq      = _div64(vec2(8.0, 0.0), three_pi);
    vec2 z_fp      = _mul64(k_eq, u_fp);
    lon_rad_fp     = t_fp;

    float z_hi     = clamp(z_fp.x, -1.0, 1.0);
    float lat_hi_0 = _seal(asin(z_hi));
    float cos_lat0 = cos(lat_hi_0);
    float sin_lat0 = sin(lat_hi_0);
    float r        = (sin_lat0 - z_hi) - z_fp.y;
    float lat_hi   = _seal(lat_hi_0 - r / cos_lat0);
    float sin_lat  = sin(lat_hi);
    float cos_lat  = cos(lat_hi);
    float r2       = (sin_lat - z_hi) - z_fp.y;
    float lat_lo   = -r2 / cos_lat;
    lat_rad_fp     = vec2(lat_hi, lat_lo);
  } else {
    float sgn = sign(u_hi);
    float s   = 2.0 - 4.0 * abs_u / PI64.x;
    const float INV_SQRT_6 = 0.40824829046386;
    float w    = abs(s) * INV_SQRT_6;
    float a0   = _seal(asin(w));
    float a_hi = _seal(a0 - (sin(a0) - w) / cos(a0));
    float a_lo = -(sin(a_hi) - w) / cos(a_hi);

    vec2 delta_fp   = vec2(2.0 * a_hi, 2.0 * a_lo);
    vec2 lat_mag_fp = _sub64(PI2_64, delta_fp);
    lat_rad_fp      = vec2(sgn * lat_mag_fp.x, sgn * lat_mag_fp.y);

    float t_hi = t_fp.x;
    float t_t  = mod(t_hi, PI2_64.x);
    float a_f  = t_hi - ((abs_u - PI4_64.x) / (abs_u - PI2_64.x))
                         * (t_t - PI4_64.x);
    lon_rad_fp = vec2(a_f, 0.0);
  }

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);

  vec3 pos_hi = vec3(lon_deg_fp.x, lat_deg_fp.x, 0.0);
  vec3 pos_lo = vec3(lon_deg_fp.y, lat_deg_fp.y, 0.0);
  gl_Position = project_position_to_clipspace(pos_hi, pos_lo, vec3(0.0), geometry.position);

  vColor = vec4(1.0);
  DECKGL_FILTER_COLOR(vColor, geometry);
}
`;

export const HEALPIX_FRAGMENT_SHADER: string = /* glsl */ `\
#version 300 es
precision highp float;

in  vec4 vColor;
out vec4 fragColor;

void main() {
  fragColor = vColor;
  DECKGL_FILTER_COLOR(fragColor, geometry);
}
`;
```

- [ ] **Step 3: Verify existing Jest tests still pass**

Run: `npm test`
Expected: PASS (no test uses the corner shader directly; this confirms the module still compiles under tsc/jest).

- [ ] **Step 4: Verify existing readback tests still pass**

Serve the worktree and open the equatorial + polar readback pages:

```bash
npx http-server . -p 8080 &
open http://localhost:8080/tmp/gpu-readback.html
open http://localhost:8080/tmp/gpu-readback-polar.html
```

Expected: all four corners on both pages continue to show ≤ 0.5 ULP (green) on `lon_hi`, `lat_hi`, `lon_lo`, `lat_lo`. The readback pages embed their own shader so they don't read our new `fp64.glsl.ts`; they're a regression check that our refactor didn't change the production shader's semantics. No pixel diff on the demo page either.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/fp64.glsl.ts src/shaders/healpix-corners.glsl.ts
git commit -m "refactor: extract fp64 Dekker primitives to fp64.glsl.ts"
```

---

## Task 5: Wrap corner math as a reusable `fxyCorner` function

**Files:**
- Modify: `src/shaders/healpix-corners.glsl.ts` (slim to function-only export)

After this task, `healpix-corners.glsl.ts` exports a single `/* glsl */` string constant containing just the `fxyCorner(face, cx, cy, nside, out lon, out lat)` function. The `main()` body moves to `healpix-cells.vs.glsl.ts` in Task 7.

- [ ] **Step 1: Replace `src/shaders/healpix-corners.glsl.ts` contents**

```ts
// src/shaders/healpix-corners.glsl.ts
/**
 * Corner-expansion math: (face, cx, cy, nside) → (lon_rad_fp, lat_rad_fp).
 *
 * Takes integer-lattice corner coordinates `(cx, cy)` = `(ix + {0,1}, iy + {0,1})`
 * and emits the spherical lon/lat for that corner in fp64 (vec2 = (hi, lo)).
 *
 * Uses fxy2tu → tu2za composition. Inner branches:
 *   - polar cap   (|u| > π/4): half-angle asin identity + Newton refinement
 *   - equatorial  (|u| ≤ π/4): direct z = (8/3π)·u, asin + Newton refinement
 *   - exact pole  (|u| ≥ π/2): lat = ±π/2, lon = 0
 *
 * Depends on fp64.glsl.ts for the Dekker primitives and π constants.
 */
export const HEALPIX_CORNERS_GLSL: string = /* glsl */ `
void fxyCorner(
  int face, int cx, int cy, int nside,
  out vec2 lon_rad_fp, out vec2 lat_rad_fp
) {
  int f_row = face / 4;
  int f1 = f_row + 2;
  int f2 = 2 * (face - 4 * f_row) - (f_row & 1) + 1;

  int i_ring = f1 * nside - cx - cy;
  int k_ring = f2 * nside + (cx - cy) + 8 * nside;

  int period = 8 * nside;
  k_ring = k_ring - (k_ring / period) * period;
  if (k_ring > 4 * nside) k_ring -= period;

  int k_int = k_ring / nside;
  int k_rem = k_ring - k_int * nside;
  int i_int = i_ring / nside;
  int i_rem = i_ring - i_int * nside;
  float k_frac = float(k_rem) / float(nside);
  float i_frac = float(i_rem) / float(nside);

  vec2 t_fp = _mul64f(PI4_64, float(k_int));
  t_fp      = _add64(t_fp, _mul64f(PI4_64, k_frac));

  vec2 i_ang = _mul64f(PI4_64, float(i_int));
  i_ang      = _add64(i_ang, _mul64f(PI4_64, i_frac));
  vec2 u_fp  = _sub64(PI2_64, i_ang);

  float u_hi  = u_fp.x;
  float abs_u = abs(u_hi);

  if (abs_u >= PI2_64.x) {
    float sgn = sign(u_hi);
    lat_rad_fp = vec2(sgn * PI2_64.x, sgn * PI2_64.y);
    lon_rad_fp = vec2(0.0);
  } else if (abs_u <= PI4_64.x) {
    vec2 three_pi  = _mul64f(PI64, 3.0);
    vec2 k_eq      = _div64(vec2(8.0, 0.0), three_pi);
    vec2 z_fp      = _mul64(k_eq, u_fp);
    lon_rad_fp     = t_fp;

    float z_hi     = clamp(z_fp.x, -1.0, 1.0);
    float lat_hi_0 = _seal(asin(z_hi));
    float cos_lat0 = cos(lat_hi_0);
    float sin_lat0 = sin(lat_hi_0);
    float r        = (sin_lat0 - z_hi) - z_fp.y;
    float lat_hi   = _seal(lat_hi_0 - r / cos_lat0);
    float sin_lat  = sin(lat_hi);
    float cos_lat  = cos(lat_hi);
    float r2       = (sin_lat - z_hi) - z_fp.y;
    float lat_lo   = -r2 / cos_lat;
    lat_rad_fp     = vec2(lat_hi, lat_lo);
  } else {
    float sgn = sign(u_hi);
    float s   = 2.0 - 4.0 * abs_u / PI64.x;
    const float INV_SQRT_6 = 0.40824829046386;
    float w    = abs(s) * INV_SQRT_6;
    float a0   = _seal(asin(w));
    float a_hi = _seal(a0 - (sin(a0) - w) / cos(a0));
    float a_lo = -(sin(a_hi) - w) / cos(a_hi);

    vec2 delta_fp   = vec2(2.0 * a_hi, 2.0 * a_lo);
    vec2 lat_mag_fp = _sub64(PI2_64, delta_fp);
    lat_rad_fp      = vec2(sgn * lat_mag_fp.x, sgn * lat_mag_fp.y);

    float t_hi = t_fp.x;
    float t_t  = mod(t_hi, PI2_64.x);
    float a_f  = t_hi - ((abs_u - PI4_64.x) / (abs_u - PI2_64.x))
                         * (t_t - PI4_64.x);
    lon_rad_fp = vec2(a_f, 0.0);
  }
}
`;
```

The file no longer exports `HEALPIX_VERTEX_SHADER` or `HEALPIX_FRAGMENT_SHADER`. Tasks 6–7 create the replacements; Task 9 wires them into the primitive layer.

- [ ] **Step 2: Temporarily maintain the old exports during transition**

The primitive layer still imports `HEALPIX_VERTEX_SHADER` and `HEALPIX_FRAGMENT_SHADER` from this file. To avoid breaking the build between now and Task 9, add a legacy shim at the bottom of `src/shaders/healpix-corners.glsl.ts`:

```ts
import { FP64_GLSL } from './fp64.glsl';

/** @deprecated Transitional re-export; removed in Task 9. */
export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

in uint faceIx;
in uint instIy;
in vec3 positions;
out vec4 vColor;

${FP64_GLSL}
${HEALPIX_CORNERS_GLSL}

void main() {
  int face = int(faceIx >> 24u);
  int ix   = int(faceIx & 0xFFFFFFu);
  int iy   = int(instIy);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  vec2 lon_rad_fp, lat_rad_fp;
  fxyCorner(face, cx, cy, int(healpixCells.nside), lon_rad_fp, lat_rad_fp);

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);
  vec3 pos_hi = vec3(lon_deg_fp.x, lat_deg_fp.x, 0.0);
  vec3 pos_lo = vec3(lon_deg_fp.y, lat_deg_fp.y, 0.0);
  gl_Position = project_position_to_clipspace(pos_hi, pos_lo, vec3(0.0), geometry.position);

  vColor = vec4(1.0);
  DECKGL_FILTER_COLOR(vColor, geometry);
}
`;

/** @deprecated Transitional re-export; removed in Task 9. */
export const HEALPIX_FRAGMENT_SHADER: string = /* glsl */ `\
#version 300 es
precision highp float;

in  vec4 vColor;
out vec4 fragColor;

void main() {
  fragColor = vColor;
  DECKGL_FILTER_COLOR(fragColor, geometry);
}
`;
```

This shim calls `fxyCorner` instead of inlining the math; semantically identical to Task 4's output.

- [ ] **Step 3: Verify existing tests still pass**

Run: `npm test`
Expected: PASS.

- [ ] **Step 4: Verify existing readback tests still pass**

Open `http://localhost:8080/tmp/gpu-readback.html` and `/tmp/gpu-readback-polar.html`.
Expected: all corners ≤ 0.5 ULP (green). Visually confirm the demo page renders identically.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/healpix-corners.glsl.ts
git commit -m "refactor: wrap corner math as fxyCorner function"
```

---

## Task 6: Create `int64.glsl.ts` and `healpix-decompose.glsl.ts`

**Files:**
- Create: `src/shaders/int64.glsl.ts`
- Create: `src/shaders/healpix-decompose.glsl.ts`

These files are not wired into the pipeline yet. They compile as TypeScript string constants only. Task 9 wires them in.

- [ ] **Step 1: Create `src/shaders/int64.glsl.ts`**

```ts
// src/shaders/int64.glsl.ts
/**
 * uvec2-based unsigned 64-bit integer ops for GLSL.
 *
 * All ops mirror src/shaders/__tests__/gpu-decode-reference.ts line-for-line.
 * `uvec2(lo, hi)` is the wire format; `u64_*` functions operate on it.
 *
 * Shift ops never shift by ≥ 32 within a single 32-bit half (GLSL undefined
 * behavior on some drivers); cross-half movement is explicit.
 */
export const INT64_GLSL: string = /* glsl */ `
uvec2 u64_add(uvec2 a, uvec2 b) {
  uint lo = a.x + b.x;
  uint carry = lo < a.x ? 1u : 0u;
  uint hi = a.y + b.y + carry;
  return uvec2(lo, hi);
}

uvec2 u64_sub(uvec2 a, uvec2 b) {
  uint lo = a.x - b.x;
  uint borrow = a.x < b.x ? 1u : 0u;
  uint hi = a.y - b.y - borrow;
  return uvec2(lo, hi);
}

uvec2 u64_mul32(uint a, uint b) {
  uint aLo = a & 0xffffu;
  uint aHi = a >> 16u;
  uint bLo = b & 0xffffu;
  uint bHi = b >> 16u;
  uint p0 = aLo * bLo;
  uint p1 = aLo * bHi;
  uint p2 = aHi * bLo;
  uint p3 = aHi * bHi;
  uint mid = p1 + p2;
  uint midCarry = mid < p1 ? 0x10000u : 0u;
  uint lo0 = p0 + ((mid & 0xffffu) << 16u);
  uint loCarry = lo0 < p0 ? 1u : 0u;
  uint hi = p3 + (mid >> 16u) + midCarry + loCarry;
  return uvec2(lo0, hi);
}

uvec2 u64_shr(uvec2 v, uint s) {
  if (s == 0u) return v;
  if (s >= 32u) {
    uint s2 = s - 32u;
    uint lo = s2 == 0u ? v.y : (v.y >> s2);
    return uvec2(lo, 0u);
  }
  uint lo = (v.x >> s) | (v.y << (32u - s));
  uint hi = v.y >> s;
  return uvec2(lo, hi);
}

uvec2 u64_shl(uvec2 v, uint s) {
  if (s == 0u) return v;
  if (s >= 32u) {
    uint s2 = s - 32u;
    uint hi = s2 == 0u ? v.x : (v.x << s2);
    return uvec2(0u, hi);
  }
  uint lo = v.x << s;
  uint hi = (v.y << s) | (v.x >> (32u - s));
  return uvec2(lo, hi);
}

uvec2 u64_and(uvec2 a, uvec2 b) {
  return uvec2(a.x & b.x, a.y & b.y);
}

bool u64_lt(uvec2 a, uvec2 b) {
  if (a.y != b.y) return a.y < b.y;
  return a.x < b.x;
}

// Returns quotient; writes remainder to 'rem'. Caller guarantees q fits in u32.
uint u64_div32(uvec2 a, uint d, out uint rem) {
  float fa = float(a.y) * 4294967296.0 + float(a.x);
  uint q = uint(floor(fa / float(d)));
  uvec2 qd = u64_mul32(q, d);
  // Correct downward
  while (u64_lt(a, qd)) {
    q = q - 1u;
    qd = u64_sub(qd, uvec2(d, 0u));
  }
  // Correct upward
  uvec2 dp1 = u64_add(qd, uvec2(d, 0u));
  while (!u64_lt(a, dp1)) {
    q = q + 1u;
    qd = dp1;
    dp1 = u64_add(qd, uvec2(d, 0u));
  }
  uvec2 r64 = u64_sub(a, qd);
  rem = r64.x;
  return q;
}

// floor(sqrt(a)) for a < 2^52. fp32 seed + integer correction loop.
uint u64_isqrt(uvec2 a) {
  if (a.x == 0u && a.y == 0u) return 0u;
  float fa = float(a.y) * 4294967296.0 + float(a.x);
  uint i = uint(floor(sqrt(fa)));
  uvec2 ii = u64_mul32(i, i);
  while (u64_lt(a, ii)) {
    i = i - 1u;
    ii = u64_mul32(i, i);
  }
  uvec2 next = u64_mul32(i + 1u, i + 1u);
  while (!u64_lt(a, next)) {
    i = i + 1u;
    ii = next;
    next = u64_mul32(i + 1u, i + 1u);
  }
  return i;
}
`;
```

- [ ] **Step 2: Create `src/shaders/healpix-decompose.glsl.ts`**

```ts
// src/shaders/healpix-decompose.glsl.ts
/**
 * GPU decoders for NEST and RING HEALPix cell IDs.
 *
 *   decodeNest(cellId, log2n)       → uvec3(face, ix, iy)    bit-exact
 *   decodeRing(cellId, nside, ...)  → uvec3(face, ix, iy)    bit-exact
 *
 * Mirrors src/shaders/__tests__/gpu-decode-reference.ts. Depends on
 * int64.glsl.ts for uvec2 ops.
 */
export const HEALPIX_DECOMPOSE_GLSL: string = /* glsl */ `
uint compact1By1(uint w) {
  w = w & 0x55555555u;
  w = (w | (w >>  1u)) & 0x33333333u;
  w = (w | (w >>  2u)) & 0x0f0f0f0fu;
  w = (w | (w >>  4u)) & 0x00ff00ffu;
  w = (w | (w >>  8u)) & 0x0000ffffu;
  return w;
}

uvec3 decodeNest(uvec2 cellId, uint log2n) {
  uint k = 2u * log2n;
  uvec2 one = uvec2(1u, 0u);
  uvec2 mask = u64_sub(u64_shl(one, k), one);
  uvec2 nestInFace = u64_and(cellId, mask);
  uint face = u64_shr(cellId, k).x;
  uint ix = compact1By1(nestInFace.x) | (compact1By1(nestInFace.y) << 16u);
  uint iy = compact1By1(nestInFace.x >> 1u) |
            (compact1By1(nestInFace.y >> 1u) << 16u);
  return uvec3(face, ix, iy);
}

uvec3 decodeRingNorth(uvec2 cellId, uint nside) {
  uvec2 onePlus2p = u64_add(u64_shl(cellId, 1u), uvec2(1u, 0u));
  uint root = u64_isqrt(onePlus2p);
  uint i = (root + 1u) / 2u;
  uvec2 i2 = u64_mul32(2u, i * (i - 1u));
  uvec2 jFull = u64_sub(cellId, i2);
  uint j = jFull.x;
  uint f = j / i;
  uint k = j - f * i;
  uint ix = nside - i + k;
  uint iy = nside - 1u - k;
  return uvec3(f, ix, iy);
}

uvec3 decodeRingEquatorial(uvec2 cellId, uint nside, uvec2 polarLim) {
  uvec2 kFull = u64_sub(cellId, polarLim);
  uint ring = 4u * nside;
  uint kmod;
  uint q = u64_div32(kFull, ring, kmod);
  uint i = nside - q;
  uint s = 1u - (i & 1u);
  uint j = 2u * kmod + s;
  int jj = int(j) - 4 * int(nside);
  int ii = int(i) + 5 * int(nside) - 1;
  uint pp = uint((ii + jj) >> 1);
  uint qq = uint((ii - jj) >> 1);
  uint PP = pp / nside;
  uint QQ = qq / nside;
  uint V = 5u - (PP + QQ);
  uint H = PP - QQ + 4u;
  uint face = 4u * V + ((H >> 1u) & 3u);
  uint ix = pp - PP * nside;
  uint iy = qq - QQ * nside;
  return uvec3(face, ix, iy);
}

uvec3 decodeRingSouth(uvec2 cellId, uint nside, uvec2 npix) {
  uvec2 p = u64_sub(u64_sub(npix, cellId), uvec2(1u, 0u));
  uvec2 onePlus2p = u64_add(u64_shl(p, 1u), uvec2(1u, 0u));
  uint root = u64_isqrt(onePlus2p);
  uint i = (root + 1u) / 2u;
  uvec2 i2 = u64_mul32(2u, i * (i - 1u));
  uvec2 jFull = u64_sub(p, i2);
  uint j = jFull.x;
  uint f = 11u - j / i;
  uint k = j - (j / i) * i;
  uint ix = i - k - 1u;
  uint iy = k;
  return uvec3(f, ix, iy);
}

uvec3 decodeRing(
  uvec2 cellId, uint nside, uvec2 polarLim, uvec2 eqLim, uvec2 npix
) {
  if (u64_lt(cellId, polarLim)) {
    return decodeRingNorth(cellId, nside);
  }
  if (u64_lt(cellId, eqLim)) {
    return decodeRingEquatorial(cellId, nside, polarLim);
  }
  return decodeRingSouth(cellId, nside, npix);
}
`;
```

- [ ] **Step 3: Verify build and tests**

Run: `npm test && npm run build`
Expected: PASS; rollup finishes without warnings on the new files.

- [ ] **Step 4: Commit**

```bash
git add src/shaders/int64.glsl.ts src/shaders/healpix-decompose.glsl.ts
git commit -m "feat: add GLSL uvec2 helpers and NEST/RING decoders (not wired yet)"
```

---

## Task 7: Create `healpix-cells.vs.glsl.ts`, `healpix-cells.fs.glsl.ts`, `shaders/index.ts`

**Files:**
- Create: `src/shaders/healpix-cells.vs.glsl.ts`
- Create: `src/shaders/healpix-cells.fs.glsl.ts`
- Create: `src/shaders/index.ts`

Still not wired in — the primitive layer keeps its old import until Task 9.

- [ ] **Step 1: Create `src/shaders/healpix-cells.vs.glsl.ts`**

```ts
// src/shaders/healpix-cells.vs.glsl.ts
/**
 * Main vertex-shader glue: attributes + main().
 *
 * Assembled with fp64, int64, healpix-decompose, healpix-corners by
 * shaders/index.ts. The final shader string prepends those dependencies
 * before this body.
 */
export const HEALPIX_CELLS_VS_MAIN: string = /* glsl */ `
in uint cellIdLo;
in uint cellIdHi;
in vec3 positions;

out vec4 vColor;

void main() {
  uvec2 cellId = uvec2(cellIdLo, cellIdHi);

  uvec3 fxy;
  if (healpixCells.scheme == 0u) {
    fxy = decodeNest(cellId, healpixCells.log2nside);
  } else {
    fxy = decodeRing(
      cellId,
      healpixCells.nside,
      healpixCells.polarLim,
      healpixCells.eqLim,
      healpixCells.npix
    );
  }
  int face = int(fxy.x);
  int ix = int(fxy.y);
  int iy = int(fxy.z);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  vec2 lon_rad_fp, lat_rad_fp;
  fxyCorner(face, cx, cy, int(healpixCells.nside), lon_rad_fp, lat_rad_fp);

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);

  vec3 pos_hi = vec3(lon_deg_fp.x, lat_deg_fp.x, 0.0);
  vec3 pos_lo = vec3(lon_deg_fp.y, lat_deg_fp.y, 0.0);
  gl_Position = project_position_to_clipspace(
    pos_hi, pos_lo, vec3(0.0), geometry.position
  );

  vColor = vec4(1.0);
  DECKGL_FILTER_COLOR(vColor, geometry);
}
`;
```

- [ ] **Step 2: Create `src/shaders/healpix-cells.fs.glsl.ts`**

```ts
// src/shaders/healpix-cells.fs.glsl.ts
/** Fragment shader — trivial; actual color work happens in DECKGL_FILTER_COLOR. */
export const HEALPIX_CELLS_FS: string = /* glsl */ `\
#version 300 es
precision highp float;

in  vec4 vColor;
out vec4 fragColor;

void main() {
  fragColor = vColor;
  DECKGL_FILTER_COLOR(fragColor, geometry);
}
`;
```

- [ ] **Step 3: Create `src/shaders/index.ts`**

```ts
// src/shaders/index.ts
/**
 * Assembles the final HEALPix cells vertex and fragment shaders from
 * focused modules. Leaf files contain only GLSL string literals; this
 * file owns dependency ordering.
 */
import { INT64_GLSL } from './int64.glsl';
import { FP64_GLSL } from './fp64.glsl';
import { HEALPIX_DECOMPOSE_GLSL } from './healpix-decompose.glsl';
import { HEALPIX_CORNERS_GLSL } from './healpix-corners.glsl';
import { HEALPIX_CELLS_VS_MAIN } from './healpix-cells.vs.glsl';
import { HEALPIX_CELLS_FS } from './healpix-cells.fs.glsl';

const VS_HEADER = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;
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

- [ ] **Step 4: Verify build**

Run: `npm test && npm run build`
Expected: PASS; build output contains the new modules. The primitive layer still imports the old transitional shim from `healpix-corners.glsl.ts` — not broken.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/healpix-cells.vs.glsl.ts \
        src/shaders/healpix-cells.fs.glsl.ts \
        src/shaders/index.ts
git commit -m "feat: add shaders/index.ts assembler and vs/fs glue modules"
```

---

## Task 8: Extend `healpix-cells-shader-module.ts` with new uniforms

**Files:**
- Modify: `src/shaders/healpix-cells-shader-module.ts`

- [ ] **Step 1: Replace file contents**

```ts
// src/shaders/healpix-cells-shader-module.ts
import type { ShaderModule } from '@luma.gl/shadertools';

export type HealpixSchemeCode = 0 | 1; // 0 = nest, 1 = ring

export type HealpixCellsProps = {
  nside: number;
  log2nside: number;
  scheme: HealpixSchemeCode;
  polarLim: [number, number];
  eqLim: [number, number];
  npix: [number, number];
};

/** Split a non-negative JS number (≤ 2^53 - 1) into [lo, hi] u32 halves. */
export function splitU53(x: number): [number, number] {
  return [x >>> 0, Math.floor(x / 4294967296)];
}

/**
 * Compute the per-draw uniforms the GPU decoders need from `nside` + `scheme`.
 * Cheap: a handful of multiplies and two u64 splits per draw.
 */
export function computeHealpixCellsUniforms(
  nside: number,
  scheme: 'nest' | 'ring'
): HealpixCellsProps {
  const polarLimN = 2 * nside * (nside - 1);
  const npixN = 12 * nside * nside;
  const eqLimN = polarLimN + 8 * nside * nside;
  return {
    nside,
    log2nside: Math.round(Math.log2(nside)),
    scheme: scheme === 'nest' ? 0 : 1,
    polarLim: splitU53(polarLimN),
    eqLim: splitU53(eqLimN),
    npix: splitU53(npixN)
  };
}

export const healpixCellsShaderModule = {
  name: 'healpixCells',
  vs: `\
uniform healpixCellsUniforms {
  uint nside;
  uint log2nside;
  uint scheme;
  uvec2 polarLim;
  uvec2 eqLim;
  uvec2 npix;
} healpixCells;
`,
  uniformTypes: {
    nside: 'u32',
    log2nside: 'u32',
    scheme: 'u32',
    polarLim: 'vec2<u32>',
    eqLim: 'vec2<u32>',
    npix: 'vec2<u32>'
  }
} as const satisfies ShaderModule<HealpixCellsProps>;
```

Note on `uniformTypes`: luma.gl's uniform type string for a uvec2 is `'vec2<u32>'`. If a future luma version rejects this at runtime, split each uvec2 into two `u32` uniforms (`polarLimLo` / `polarLimHi`, etc.) and update the GLSL uniform block accordingly.

- [ ] **Step 2: Verify build**

Run: `npm run build`
Expected: PASS. No test exercises this module directly; correctness confirmed in Task 9 when the layer wires it up.

- [ ] **Step 3: Commit**

```bash
git add src/shaders/healpix-cells-shader-module.ts
git commit -m "feat: extend healpixCells shader module with log2nside/scheme/ring uniforms"
```

---

## Task 9: Add `splitCellIds` + wire new attributes through both layers

**Files:**
- Create: `src/utils/split-cell-ids.ts`
- Create: `src/utils/split-cell-ids.test.ts`
- Modify: `src/layers/healpix-cells-layer.ts`
- Modify: `src/layers/healpix-cells-primitive-layer.ts`
- Modify: `src/shaders/healpix-corners.glsl.ts` (remove transitional shim)
- Delete: `src/utils/decompose-cell-ids.ts`
- Delete: `src/utils/decompose-cell-ids.test.ts`

This task is larger because the pieces must ship together for the app to run. Order the steps so each intermediate state compiles.

- [ ] **Step 1: Write the failing test for `splitCellIds`**

```ts
// src/utils/split-cell-ids.test.ts
import { splitCellIds, getSharedZeroU32 } from './split-cell-ids';

describe('splitCellIds', () => {
  it('Uint32Array input: lo aliases, hi is shared zero buffer', () => {
    const ids = new Uint32Array([0, 1, 42, 0xffffffff]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(cellIdLo.buffer).toBe(ids.buffer);
    expect(cellIdLo.length).toBe(4);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 42, 0xffffffff]);
    expect(cellIdHi.length).toBeGreaterThanOrEqual(4);
    expect(Array.from(cellIdHi.slice(0, 4))).toEqual([0, 0, 0, 0]);
    // hi is the shared buffer
    expect(cellIdHi).toBe(getSharedZeroU32(4));
  });

  it('Int32Array input with non-negative values: lo aliases bytes', () => {
    const ids = new Int32Array([0, 1, 2147483647]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(cellIdLo.buffer).toBe(ids.buffer);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 2147483647]);
    expect(Array.from(cellIdHi.slice(0, 3))).toEqual([0, 0, 0]);
  });

  it('Float64Array input: runs split loop', () => {
    const TWO32 = 4294967296;
    const ids = new Float64Array([0, 1, TWO32, TWO32 + 7, 12 * 2 ** 48 - 1]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(Array.from(cellIdLo)).toEqual([
      0,
      1,
      0,
      7,
      ((12 * 2 ** 48 - 1) >>> 0)
    ]);
    expect(Array.from(cellIdHi)).toEqual([
      0,
      0,
      1,
      1,
      Math.floor((12 * 2 ** 48 - 1) / TWO32)
    ]);
    // Does not alias source
    expect(cellIdLo.buffer).not.toBe(ids.buffer);
  });

  it('Float32Array input: runs split loop', () => {
    const ids = new Float32Array([0, 1, 42]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 42]);
    expect(Array.from(cellIdHi)).toEqual([0, 0, 0]);
    expect(cellIdLo.buffer).not.toBe(ids.buffer);
  });

  it('empty input produces empty buffers', () => {
    const { cellIdLo, cellIdHi } = splitCellIds(new Uint32Array(0));
    expect(cellIdLo.length).toBe(0);
    expect(cellIdHi.length).toBeGreaterThanOrEqual(0);
  });

  it('shared zero buffer grows on demand', () => {
    splitCellIds(new Uint32Array(4));
    const small = getSharedZeroU32(4);
    splitCellIds(new Uint32Array(10));
    const large = getSharedZeroU32(10);
    expect(large.length).toBeGreaterThanOrEqual(10);
    // The earlier reference may or may not still point at the same buffer
    // depending on growth strategy; large must always zero-fill.
    expect(Array.from(large.slice(0, 10))).toEqual([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    // Silence unused
    void small;
  });
});
```

- [ ] **Step 2: Run test to verify failure**

Run: `npm test -- split-cell-ids`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `splitCellIds`**

```ts
// src/utils/split-cell-ids.ts
import type { CellIdArray } from '../types/cell-ids';

const TWO32 = 4294967296;

let sharedZero: Uint32Array = new Uint32Array(0);

export function getSharedZeroU32(minLength: number): Uint32Array {
  if (sharedZero.length < minLength) {
    sharedZero = new Uint32Array(Math.max(minLength, 1024));
  }
  return sharedZero;
}

export type SplitCellIds = {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
};

/**
 * Split cell IDs into low/high u32 halves suitable for upload as two
 * deck.gl instance attributes composed to uvec2 in the vertex shader.
 *
 * - Uint32Array / Int32Array inputs: cellIdLo aliases the input's bytes
 *   (zero copy) and cellIdHi is a shared zero buffer sized ≥ n.
 * - Float64Array / Float32Array: runs a JS split loop.
 */
export function splitCellIds(cellIds: CellIdArray): SplitCellIds {
  const n = cellIds.length;
  if (cellIds instanceof Uint32Array) {
    return {
      cellIdLo: cellIds,
      cellIdHi: getSharedZeroU32(n)
    };
  }
  if (cellIds instanceof Int32Array) {
    return {
      cellIdLo: new Uint32Array(cellIds.buffer, cellIds.byteOffset, n),
      cellIdHi: getSharedZeroU32(n)
    };
  }
  const lo = new Uint32Array(n);
  const hi = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const id = cellIds[i];
    lo[i] = id >>> 0;
    hi[i] = Math.floor(id / TWO32);
  }
  return { cellIdLo: lo, cellIdHi: hi };
}
```

- [ ] **Step 4: Run test to verify pass**

Run: `npm test -- split-cell-ids`
Expected: PASS.

- [ ] **Step 5: Commit the util**

```bash
git add src/utils/split-cell-ids.ts src/utils/split-cell-ids.test.ts
git commit -m "feat: add splitCellIds utility"
```

- [ ] **Step 6: Update `HealpixCellsPrimitiveLayer`**

Replace `src/layers/healpix-cells-primitive-layer.ts` with:

```ts
import {
  DefaultProps,
  Layer,
  LayerContext,
  picking,
  project32,
  UpdateParameters
} from '@deck.gl/core';
import { Geometry, Model } from '@luma.gl/engine';
import type { RenderPass } from '@luma.gl/core';
import {
  HEALPIX_FRAGMENT_SHADER,
  HEALPIX_VERTEX_SHADER
} from '../shaders';
import {
  healpixCellsShaderModule,
  computeHealpixCellsUniforms
} from '../shaders/healpix-cells-shader-module';

export type HealpixCellsPrimitiveLayerProps = {
  nside: number;
  scheme: 'nest' | 'ring';
  instanceCount: number;
};

type _HealpixCellsPrimitiveLayerProps = HealpixCellsPrimitiveLayerProps;
type HealpixCellsPrimitiveLayerMergedProps = _HealpixCellsPrimitiveLayerProps &
  import('@deck.gl/core').LayerProps;

const defaultProps: DefaultProps<_HealpixCellsPrimitiveLayerProps> = {
  nside: { type: 'number', value: 1 },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  instanceCount: { type: 'number', value: 0 }
};

const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);
const QUAD_POSITIONS = new Float32Array(12);

export class HealpixCellsPrimitiveLayer extends Layer<HealpixCellsPrimitiveLayerMergedProps> {
  static layerName = 'HealpixCellsPrimitiveLayer';
  static defaultProps = defaultProps;

  declare state: { model: Model | null };

  getNumInstances(): number {
    return this.props.instanceCount;
  }

  getShaders(): ReturnType<Layer['getShaders']> {
    return super.getShaders({
      vs: HEALPIX_VERTEX_SHADER,
      fs: HEALPIX_FRAGMENT_SHADER,
      modules: [project32, picking, healpixCellsShaderModule]
    });
  }

  initializeState(_context: LayerContext): void {
    this.getAttributeManager()!.addInstanced({
      cellIdLo: { size: 1, type: 'uint32', noAlloc: true },
      cellIdHi: { size: 1, type: 'uint32', noAlloc: true }
    });
  }

  updateState(params: UpdateParameters<this>): void {
    super.updateState(params);
    if (params.changeFlags.extensionsChanged || !this.state.model) {
      this.state.model?.destroy();
      this.state.model = this._getModel();
      this.getAttributeManager()!.invalidateAll();
    }
  }

  finalizeState(context: LayerContext): void {
    super.finalizeState(context);
    this.state.model?.destroy();
  }

  draw({ renderPass }: { renderPass: RenderPass }): void {
    const { model } = this.state;
    if (!model || this.props.instanceCount === 0) return;

    model.shaderInputs.setProps({
      healpixCells: computeHealpixCellsUniforms(
        this.props.nside,
        this.props.scheme
      )
    });
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  private _getModel(): Model {
    const parameters =
      this.context.device.type === 'webgpu'
        ? {
            depthWriteEnabled: true,
            depthCompare: 'less-equal' as const
          }
        : undefined;

    return new Model(this.context.device, {
      ...this.getShaders(),
      id: `${this.props.id}-model`,
      bufferLayout: this.getAttributeManager()!.getBufferLayouts(),
      geometry: new Geometry({
        topology: 'triangle-list',
        attributes: {
          indices: QUAD_INDICES,
          positions: { size: 3, value: QUAD_POSITIONS }
        }
      }),
      isInstanced: true,
      parameters
    });
  }
}
```

- [ ] **Step 7: Update `HealpixCellsLayer` (composite)**

Replace the imports and the `_decomposeCellIds` / state-shape / `renderLayers` portions of `src/layers/healpix-cells-layer.ts`:

```ts
import {
  CompositeLayer,
  DefaultProps,
  Layer,
  LayerExtension,
  UpdateParameters
} from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { splitCellIds } from '../utils/split-cell-ids';
import { HealpixCellsPrimitiveLayer } from './healpix-cells-primitive-layer';
import { HEALPIX_COLOR_FRAMES_EXTENSION } from '../extensions/healpix-color-frames-extension';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixCellsLayerProps } from '../types/layer-props';

type _HealpixCellsLayerProps = {
  nside: number;
  cellIds: CellIdArray;
  scheme: 'nest' | 'ring';
  colorFrames: Uint8Array[];
  currentFrame: number;
};

type HealpixCellsLayerState = {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
  frameTexture: Texture | null;
  cellTextureWidth: number;
  frameCount: number;
};

type TextureData = {
  colors: Uint8Array;
  width: number;
  height: number;
  depth: number;
  frameCount: number;
};

const EMPTY_RGBA_TEXEL = new Uint8Array([0, 0, 0, 0]);

const defaultProps: DefaultProps<_HealpixCellsLayerProps> = {
  nside: { type: 'number', value: 0 },
  cellIds: { type: 'object', value: new Uint32Array(0), compare: true },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  colorFrames: { type: 'object', value: [], compare: true },
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
      frameTexture: null,
      cellTextureWidth: 1,
      frameCount: 0
    });
    this._splitCellIds();
    this._updateColorTexture();
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return !!changeFlags.propsOrDataChanged;
  }

  updateState({ props, oldProps }: UpdateParameters<this>): void {
    if (props.cellIds !== oldProps.cellIds) {
      this._splitCellIds();
    }
    if (
      props.cellIds !== oldProps.cellIds ||
      props.colorFrames !== oldProps.colorFrames
    ) {
      this._updateColorTexture();
    }
  }

  finalizeState(): void {
    this.state.frameTexture?.destroy();
  }

  renderLayers(): Layer[] {
    const { cellIdLo, cellIdHi, frameTexture, cellTextureWidth, frameCount } =
      this.state;
    const { cellIds, nside, scheme, currentFrame } = this.props;
    const count = cellIds.length;
    if (count === 0 || !frameTexture) return [];

    const frameIndex = Math.max(
      0,
      Math.min(frameCount - 1, Math.floor(currentFrame))
    );

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
          frameTexture,
          frameIndex,
          cellTextureWidth,
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_FRAMES_EXTENSION
          ]
        })
      )
    ];
  }

  private _splitCellIds(): void {
    const { cellIds } = this.props;
    if (!cellIds?.length) {
      this.setState({
        cellIdLo: new Uint32Array(0),
        cellIdHi: new Uint32Array(0)
      });
      return;
    }
    const { cellIdLo, cellIdHi } = splitCellIds(cellIds);
    this.setState({ cellIdLo, cellIdHi });
  }

  // _updateColorTexture and _buildTextureData are unchanged from the existing
  // file — keep them verbatim below this line.
}
```

Keep `_updateColorTexture` and `_buildTextureData` exactly as they are in the existing file (lines 167 onward — the color-texture path is out of scope for this change).

- [ ] **Step 8: Remove the transitional shim from `healpix-corners.glsl.ts`**

Open `src/shaders/healpix-corners.glsl.ts` and delete the `/** @deprecated … */` `HEALPIX_VERTEX_SHADER` and `HEALPIX_FRAGMENT_SHADER` exports added in Task 5. The file's only export is now `HEALPIX_CORNERS_GLSL`.

- [ ] **Step 9: Delete `decomposeCellIds`**

```bash
git rm src/utils/decompose-cell-ids.ts src/utils/decompose-cell-ids.test.ts
```

- [ ] **Step 10: Run all tests**

Run: `npm test`
Expected: PASS. No references to `decomposeCellIds` anywhere.

- [ ] **Step 11: Build**

Run: `npm run build`
Expected: PASS. `dist/` output has no reference to `decompose-cell-ids` and imports `shaders/index.ts` transitively through the primitive layer.

- [ ] **Step 12: Visual regression check**

Open the demo page (whatever loads `HealpixCellsLayer` in dev). At `nside ∈ {64, 1024, 262144}` with both `scheme: 'nest'` and `scheme: 'ring'`, compare against the last known-good build of main. Rendered cells must be pixel-identical.

- [ ] **Step 13: Commit**

```bash
git add -A
git commit -m "feat: move NEST/RING decode to GPU; drop decomposeCellIds"
```

---

## Task 10: Move and update NEST readback tests to `test/gpu/`

**Files:**
- Move: `tmp/gpu-readback.html` → `test/gpu/gpu-readback-nest-equatorial.html`
- Move: `tmp/gpu-readback-polar.html` → `test/gpu/gpu-readback-nest-polar.html`
- Move: `tmp/inspect-cell.mjs` → `test/gpu/inspect-cell.mjs`
- Move: `tmp/compute-truth.mjs` → `test/gpu/compute-truth.mjs`

Each readback page embeds the production shader inline. Update the inline shader to mirror the new assembly (int64 + fp64 + decompose + corners + vs-main) and the attributes to `cellIdLo` / `cellIdHi`. Also emit decode output (face/ix/iy) in the transform-feedback varyings.

- [ ] **Step 1: Move the files**

```bash
mkdir -p test/gpu
git mv tmp/gpu-readback.html       test/gpu/gpu-readback-nest-equatorial.html
git mv tmp/gpu-readback-polar.html test/gpu/gpu-readback-nest-polar.html
git mv tmp/inspect-cell.mjs        test/gpu/inspect-cell.mjs
git mv tmp/compute-truth.mjs       test/gpu/compute-truth.mjs
rmdir tmp 2>/dev/null || true
```

- [ ] **Step 2: Update `test/gpu/inspect-cell.mjs` header and packing info**

Open the file and replace the "faceIx (u32 hex)" / "instIy (u32 hex)" print block (lines near the top that show the attribute packing) with:

```js
const cellIdLo = cellId >>> 0;
const cellIdHi = Math.floor(cellId / 4294967296);
console.log(`  cellIdLo (u32 hex): 0x${cellIdLo.toString(16).padStart(8, '0')}`);
console.log(`  cellIdHi (u32 hex): 0x${cellIdHi.toString(16).padStart(8, '0')}`);
```

No other logic changes — the `fxyCorners` + `fxy2tu` + `tu2za` sections still produce the correct truth.

- [ ] **Step 3: Update `test/gpu/compute-truth.mjs` packing block**

Find the block that prints `faceIx (u32 hex)` / `instIy (u32 hex)` (around line 19 of the current file) and replace with:

```js
const cellIdLo = cellId >>> 0;
const cellIdHi = Math.floor(cellId / 4294967296);
console.log('cellIdLo (u32 hex):', cellIdLo.toString(16).padStart(8, '0'));
console.log('cellIdHi (u32 hex):', cellIdHi.toString(16).padStart(8, '0'));
```

Also add, after the initial `nest2fxy` print:

```js
console.log('\nFor GPU decode tests, expected (face, ix, iy) from decodeNest:');
console.log(`  face=${f}  ix=${x}  iy=${y}`);
```

- [ ] **Step 4: Rewrite `test/gpu/gpu-readback-nest-equatorial.html`**

The shader inside the page needs to become a full decode+corner composition. Replace the entire `<script type="module">` block's `VS` constant with the content below and adjust the attribute plumbing. Full new VS:

```js
const VS = `#version 300 es
precision highp float;
precision highp int;

uniform int u_nside;
uniform uint u_log2nside;
uniform uint u_scheme;
uniform uvec2 u_polarLim;
uniform uvec2 u_eqLim;
uniform uvec2 u_npix;

in uint cellIdLo;
in uint cellIdHi;

// Two varyings: decoded (face, ix, iy) and (lon_hi, lon_lo, lat_hi, lat_lo).
flat out uvec3 vDecode;
flat out vec4 vCorner;

${/* paste INT64_GLSL source here */ ''}
${/* paste FP64_GLSL source here */ ''}
${/* paste HEALPIX_DECOMPOSE_GLSL source here */ ''}
${/* paste HEALPIX_CORNERS_GLSL source here */ ''}

void main() {
  uvec2 cellId = uvec2(cellIdLo, cellIdHi);
  uvec3 fxy;
  if (u_scheme == 0u) {
    fxy = decodeNest(cellId, u_log2nside);
  } else {
    fxy = decodeRing(cellId, uint(u_nside), u_polarLim, u_eqLim, u_npix);
  }
  int face = int(fxy.x);
  int ix = int(fxy.y);
  int iy = int(fxy.z);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  vec2 lon_rad_fp, lat_rad_fp;
  fxyCorner(face, cx, cy, u_nside, lon_rad_fp, lat_rad_fp);

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);

  vDecode = fxy;
  vCorner = vec4(lon_deg_fp.x, lon_deg_fp.y, lat_deg_fp.x, lat_deg_fp.y);
  gl_Position = vec4(0.0);
  gl_PointSize = 1.0;
}`;
```

Because the page doesn't have a build system, populate the `${…}` placeholders by literally copy-pasting the GLSL from `src/shaders/int64.glsl.ts`, `src/shaders/fp64.glsl.ts`, `src/shaders/healpix-decompose.glsl.ts`, and `src/shaders/healpix-corners.glsl.ts` between the backticks.

Update the attribute plumbing: rename `locFaceIx`/`locInstIy` → `locCellIdLo`/`locCellIdHi`; replace the `faceBuf` / `iyBuf` allocations with:

```js
const cellId = 228532280344;
const cellIdLo = (cellId >>> 0);
const cellIdHi = Math.floor(cellId / 4294967296);

const loBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, loBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array([cellIdLo]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(locCellIdLo);
gl.vertexAttribIPointer(locCellIdLo, 1, gl.UNSIGNED_INT, 0, 0);
gl.vertexAttribDivisor(locCellIdLo, 1);

const hiBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, hiBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array([cellIdHi]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(locCellIdHi);
gl.vertexAttribIPointer(locCellIdHi, 1, gl.UNSIGNED_INT, 0, 0);
gl.vertexAttribDivisor(locCellIdHi, 1);
```

Set the new uniforms:

```js
const NSIDE = 262144;
const LOG2N = Math.round(Math.log2(NSIDE));
const SCHEME = 0; // nest
const polarLim = 2 * NSIDE * (NSIDE - 1);
const eqLim = polarLim + 8 * NSIDE * NSIDE;
const npix = 12 * NSIDE * NSIDE;
const splitU53 = (x) => [x >>> 0, Math.floor(x / 4294967296)];
gl.uniform1i(locNside, NSIDE);
gl.uniform1ui(gl.getUniformLocation(prog, 'u_log2nside'), LOG2N);
gl.uniform1ui(gl.getUniformLocation(prog, 'u_scheme'), SCHEME);
gl.uniform2ui(gl.getUniformLocation(prog, 'u_polarLim'), ...splitU53(polarLim));
gl.uniform2ui(gl.getUniformLocation(prog, 'u_eqLim'),    ...splitU53(eqLim));
gl.uniform2ui(gl.getUniformLocation(prog, 'u_npix'),     ...splitU53(npix));
```

Change transform-feedback varyings to `['vDecode', 'vCorner']` and expand the readback buffer to hold 4 vertices × (uvec3 + vec4) = 4 × 28 bytes = 112 bytes. Since GLSL ES 300 transform feedback can't interleave uint and float, pass them separately with `SEPARATE_ATTRIBS`:

```js
gl.transformFeedbackVaryings(prog, ['vDecode', 'vCorner'], gl.SEPARATE_ATTRIBS);
// …
const tfDecode = gl.createBuffer();
gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, tfDecode);
gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, 4 * 3 * 4, gl.STREAM_READ);
const tfCorner = gl.createBuffer();
gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, tfCorner);
gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, 4 * 4 * 4, gl.STREAM_READ);

gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, tfDecode);
gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, tfCorner);
```

Read back both buffers:

```js
const decode = new Uint32Array(12);
gl.bindBuffer(gl.COPY_READ_BUFFER, tfDecode);
gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, decode);

const corner = new Float32Array(16);
gl.bindBuffer(gl.COPY_READ_BUFFER, tfCorner);
gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, corner);
```

Prepend the diff section with a decode check. For the Lisbon cell (`cellId=228532280344`, nside=262144), `nest2fxy` → `(f=3, x=227300, y=18594)`. All four `gl_VertexID` runs produce the same decoded `(face, ix, iy)` (decode is per-instance), so check them all equal to expected:

```js
html += '\nDecode check (face, ix, iy) — must match for all 4 vertices:\n';
const EXPECTED_FACE = 3, EXPECTED_IX = 227300, EXPECTED_IY = 18594;
let decodeOk = true;
for (let i = 0; i < 4; i++) {
  const face = decode[i * 3 + 0];
  const ix   = decode[i * 3 + 1];
  const iy   = decode[i * 3 + 2];
  const ok = face === EXPECTED_FACE && ix === EXPECTED_IX && iy === EXPECTED_IY;
  decodeOk = decodeOk && ok;
  html += `  v${i}  face=${face}  ix=${ix}  iy=${iy}  ${
    ok ? '<span class="ok">OK</span>' : '<span class="bad">MISMATCH</span>'
  }\n`;
}
```

Then run the existing ULP-diff block, reading from `corner` instead of `result`.

The polar page gets the same treatment; pick its expected `(face, ix, iy)` by running `node test/gpu/inspect-cell.mjs 274626645138`.

- [ ] **Step 5: Apply the same edits to `test/gpu/gpu-readback-nest-polar.html`**

Same changes; the only differences are the `cellId` constant and the TRUTH array (unchanged from the pre-move version).

- [ ] **Step 6: Manual verify**

```bash
npx http-server . -p 8080 &
open http://localhost:8080/test/gpu/gpu-readback-nest-equatorial.html
open http://localhost:8080/test/gpu/gpu-readback-nest-polar.html
```

Expected on both pages:
- Decode check: all 4 vertices `OK`.
- Corner diff: all four corners green on `lon_hi`, `lat_hi`, `lon_lo`, `lat_lo` (≤ 0.5 ULP).

- [ ] **Step 7: Commit**

```bash
git add test/gpu/
git commit -m "test: move GPU readback tests to test/gpu/ and update for GPU decode"
```

---

## Task 11: Add `test/gpu/gpu-readback-ring.html` for RING scheme

**Files:**
- Create: `test/gpu/gpu-readback-ring.html`

Covers the three RING decode branches (north polar cap, equatorial belt, south polar cap) in one page. All three cells at nside=262144 and `SCHEME=1`.

Truth values were pre-computed using `healpix-ts` v1.0.0 (`ring2fxy` + `cornersRingLonLat`):

| Cell | cellId | cellIdLo | cellIdHi | face | ix | iy |
|------|--------:|---------:|---------:|-----:|---:|---:|
| North cap | 68719214592 | 4294705152 | 15 | 3 | 222472 | 116452 |
| Equatorial | 320690379491 | 2862799587 | 74 | 7 | 99726 | 249799 |
| South cap | 755913981952 | 4294705152 | 175 | 9 | 71476 | 113887 |

- [ ] **Step 1: Write `test/gpu/gpu-readback-ring.html`**

Clone `gpu-readback-nest-equatorial.html` and apply these changes:

- `<h1>` text: `"HEALPix GPU readback — RING cells, nside=262144"`.
- In the JS block, set `SCHEME = 1` and define the cases up front:

```js
const NSIDE = 262144;
const SCHEME = 1;
const CASES = [
  {
    label: 'RING north cap',
    cellId: 68719214592,
    cellIdLo: 4294705152,
    cellIdHi: 15,
    expected: { face: 3, ix: 222472, iy: 116452 },
    truth: [
      { lon: -19.261714914599395, lat: 56.443025755521965 },
      { lon: -19.26209653490713,  lat: 56.44283936188415  },
      { lon: -19.26199261992616,  lat: 56.4426529681549   },
      { lon: -19.261611001116762, lat: 56.44283936188415  }
    ]
  },
  {
    label: 'RING equatorial',
    cellId: 320690379491,
    cellIdLo: 2862799587,
    cellIdHi: 74,
    expected: { face: 7, ix: 99726, iy: 249799 },
    truth: [
      { lon: -115.76173782348641, lat: 12.839837486161988 },
      { lon: -115.76190948486337, lat: 12.839688038577663 },
      { lon: -115.76173782348641, lat: 12.83953859108219  },
      { lon: -115.76156616210946, lat: 12.839688038577663 }
    ]
  },
  {
    label: 'RING south cap',
    cellId: 755913981952,
    cellIdLo: 4294705152,
    cellIdHi: 175,
    expected: { face: 9, ix: 71476, iy: 113887 },
    truth: [
      { lon: 124.70412429530926, lat: -56.44246657433427   },
      { lon: 124.70382598562826, lat: -56.442652968154896  },
      { lon: 124.70401320651911, lat: -56.442839361884154  },
      { lon: 124.70431151679938, lat: -56.442652968154896  }
    ]
  }
];
```

- Compile the shader program once (same VS/FS composition as the NEST page, copy-pasted from `src/shaders/*.glsl.ts`).
- Set the fixed uniforms once:

```js
const LOG2N = Math.round(Math.log2(NSIDE));
const polarLim = 2 * NSIDE * (NSIDE - 1);
const eqLim = polarLim + 8 * NSIDE * NSIDE;
const npix = 12 * NSIDE * NSIDE;
const splitU53 = (x) => [x >>> 0, Math.floor(x / 4294967296)];
gl.useProgram(prog);
gl.uniform1i(locNside, NSIDE);
gl.uniform1ui(gl.getUniformLocation(prog, 'u_log2nside'), LOG2N);
gl.uniform1ui(gl.getUniformLocation(prog, 'u_scheme'), SCHEME);
gl.uniform2ui(gl.getUniformLocation(prog, 'u_polarLim'), ...splitU53(polarLim));
gl.uniform2ui(gl.getUniformLocation(prog, 'u_eqLim'),    ...splitU53(eqLim));
gl.uniform2ui(gl.getUniformLocation(prog, 'u_npix'),     ...splitU53(npix));
```

- For each case, upload its `cellIdLo` / `cellIdHi` (1 instance), run transform feedback for 4 vertices, read back `vDecode` and `vCorner`, accumulate the per-case HTML output, and append it to the page.

- [ ] **Step 2: Manual verify**

```bash
open http://localhost:8080/test/gpu/gpu-readback-ring.html
```

Expected: each of the three cases shows decode `OK` for all 4 vertices, and all four corners ≤ 0.5 ULP (green).

- [ ] **Step 3: Commit**

```bash
git add test/gpu/gpu-readback-ring.html
git commit -m "test: add GPU readback page for RING scheme (3 branches)"
```

---

## Task 12: Add `test/gpu/README.md`

**Files:**
- Create: `test/gpu/README.md`

- [ ] **Step 1: Write the README**

```markdown
# HEALPix GPU pipeline tests

The `test/gpu/` folder contains manual validation scripts that the Jest
suite can't replicate: GPU transform-feedback readback and ULP diffs
against fp64 CPU truth.

## Node truth scripts

    node test/gpu/compute-truth.mjs [cellId]
    node test/gpu/inspect-cell.mjs <cellId>

Each prints:
- The CPU truth `(face, ix, iy, lon, lat)` for the given cell.
- The `(cellIdLo, cellIdHi)` u32 pair the shader expects.
- A ready-to-paste `TRUTH` array of `{lon, lat}` objects.

## Browser readback

Serve the worktree with any static file server and open the pages:

    npx http-server . -p 8080
    open http://localhost:8080/test/gpu/gpu-readback-nest-equatorial.html
    open http://localhost:8080/test/gpu/gpu-readback-nest-polar.html
    open http://localhost:8080/test/gpu/gpu-readback-ring.html

Each page runs a transform-feedback draw that mirrors the production
vertex shader (int64 + fp64 + decompose + corners), reads back
`(face, ix, iy)` and `(lon, lat)` as fp64 pairs, and diffs against CPU
truth. Legend:

    green  ≤ 0.5 ULP   (bit-exact)
    amber  ≤ 4 ULP     (GLSL spec-bound for asin/sin/cos)
    red    > 4 ULP     (pipeline bug)

Decode output (face/ix/iy) must be green (bit-exact) — any mismatch
indicates a bug in `int64.glsl.ts` or `healpix-decompose.glsl.ts`.

## Updating truth

When changing the shader in a way that alters output:

    node test/gpu/compute-truth.mjs <cellId>

Paste the printed TRUTH array into the HTML page's TRUTH (or CASES)
constant. If the shader source files change, also re-copy them into
the page's inline shader (pages embed their own shader for
independence from the build).

## Keeping pages in sync with the build

Each HTML page embeds the production shader source inline as a JS
template string. When you change any of:

- `src/shaders/int64.glsl.ts`
- `src/shaders/fp64.glsl.ts`
- `src/shaders/healpix-decompose.glsl.ts`
- `src/shaders/healpix-corners.glsl.ts`

copy the updated GLSL into the corresponding section of each readback
page. The pages are deliberately self-contained so they can be loaded
with `file://` or any static server.
```

- [ ] **Step 2: Commit**

```bash
git add test/gpu/README.md
git commit -m "docs: add test/gpu/README.md with run instructions"
```

---

## Final verification

After all tasks:

- [ ] `npm test` — all Jest suites pass (uvec2 helpers, decodeNest, decodeRing, splitCellIds, plus any pre-existing tests).
- [ ] `npm run build` — rollup completes without warnings; dist bundle no longer references `healpix-ts` in the cell-id decomposition path (should only appear under `devDependencies` consumers like tests).
- [ ] `npm run lint` — clean.
- [ ] Manual: demo page renders identically to `main` at `nside ∈ {64, 1024, 262144}`, both `scheme ∈ {nest, ring}`.
- [ ] Manual: all three readback pages show green decode + green corners across every case.
