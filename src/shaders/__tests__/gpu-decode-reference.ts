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
  const lo = (a[0] + b[0]) >>> 0;
  const carry = lo < a[0] ? 1 : 0;
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
  // GLSL `<< 16` on uint is unsigned; in JS, `(x << 16)` wraps to a negative
  // int32 when bit 15 of x is set, which breaks the `loFull >= TWO32` overflow
  // check. Force unsigned with `>>> 0` to mirror GLSL semantics.
  const loFull = p0 + (((mid & 0xffff) << 16) >>> 0);
  const lo = loFull >>> 0;
  const loCarry = loFull >= TWO32 ? 1 : 0;
  const hi = (p3 + ((mid >>> 16) & 0xffff) + midCarry + loCarry) >>> 0;
  return [lo, hi];
}

export function u64_shr(v: U64, s: number): U64 {
  if (s === 0) return v;
  if (s >= 32) {
    const s2 = s - 32;
    const lo = s2 === 0 ? v[1] : v[1] >>> s2;
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
    const hi = s2 === 0 ? v[0] : v[0] << s2;
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
 *
 * Note: During correction, `q` is a signed JS number — never use `(q - 1) >>> 0`,
 * which wraps 0 → 0xffffffff and corrupts the quotient.
 */
export function u64_div32(a: U64, d: number): { q: number; r: number } {
  const fa = a[1] * TWO32 + a[0];
  let q = Math.floor(fa / d);
  if (!Number.isFinite(q) || q < 0) q = 0;
  let qd = u64_mul32(q >>> 0, d);
  while (u64_lt(a, qd)) {
    q -= 1;
    qd = u64_sub(qd, [d, 0]);
  }
  let dp1 = u64_add(qd, [d, 0]);
  while (!u64_lt(a, dp1)) {
    q += 1;
    qd = dp1;
    dp1 = u64_add(qd, [d, 0]);
  }
  const rFull = u64_sub(a, qd);
  return { q: q >>> 0, r: rFull[0] >>> 0 };
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

/**
 * Decode a NEST HEALPix cell ID into (face, ix, iy).
 * `log2n = log2(nside)`; the low `2·log2n` bits are the Morton-interleaved
 * (ix, iy) within-face coords, remaining bits are the face id (0..11).
 */
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

/**
 * Compute the three RING-scheme cutpoints as uvec2 tuples:
 *   polarLim = 2·nside·(nside−1)   (end of north cap)
 *   eqLim    = polarLim + 8·nside²  (end of equatorial belt)
 *   npix     = 12·nside²            (total cell count)
 */
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

/**
 * Decode a RING HEALPix cell ID into (face, ix, iy). Dispatches to the
 * north-polar, equatorial, or south-polar branch based on which range
 * `cellId` lands in.
 */
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
  const onePlus2p = u64_add(u64_shl(cellId, 1), [1, 0]);
  const root = u64_isqrt(onePlus2p);
  const i = ((root + 1) >>> 1) >>> 0;
  // i*(i-1) can exceed u32 at nside ≥ 2^17, so compute as u64 then ×2.
  const i2 = u64_shl(u64_mul32(i, (i - 1) >>> 0), 1);
  const jFull = u64_sub(cellId, i2);
  // j < 4·i ≤ 4·nside ≤ 2^26 at nside=2^24, so the low word holds j exactly.
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
  const ring = 4 * nside;
  const { q, r: kmod } = u64_div32(kFull, ring);
  const i = nside - q;
  const s = i % 2 === 0 ? 1 : 0;
  const j = 2 * kmod + s;
  const jj = j - 4 * nside;
  const ii = i + 5 * nside - 1;
  const pp = (ii + jj) / 2;
  const qq = (ii - jj) / 2;
  const PP = Math.floor(pp / nside);
  const QQ = Math.floor(qq / nside);
  const V = 5 - (PP + QQ);
  const H = PP - QQ + 4;
  const face = 4 * V + ((H >> 1) & 3);
  const ix = pp - PP * nside;
  const iy = qq - QQ * nside;
  return { face, ix, iy };
}

function decodeRingSouth(cellId: U64, nside: number, npix: U64): DecodeResult {
  const p = u64_sub(u64_sub(npix, cellId), [1, 0]);
  const onePlus2p = u64_add(u64_shl(p, 1), [1, 0]);
  const root = u64_isqrt(onePlus2p);
  const i = ((root + 1) >>> 1) >>> 0;
  // Same u32-overflow guard as the north branch.
  const i2 = u64_shl(u64_mul32(i, (i - 1) >>> 0), 1);
  const jFull = u64_sub(p, i2);
  const j = jFull[0] >>> 0;
  const fDiv = Math.floor(j / i);
  const f = (11 - fDiv) >>> 0;
  const k = j - fDiv * i;
  const ix = (i - k - 1) >>> 0;
  const iy = k >>> 0;
  return { face: f, ix, iy };
}
