/**
 * uvec2-based unsigned 64-bit integer ops for GLSL.
 *
 * All ops mirror src/shaders/__tests__/gpu-decode-reference.ts line-for-line.
 * `uvec2(lo, hi)` is the wire format; `u64_*` functions operate on it.
 *
 * Shift ops never shift by ≥ 32 within a single 32-bit half (GLSL undefined
 * behavior on some drivers); cross-half movement is explicit.
 *
 * `u64_div32` uses a signed int quotient during correction so `q - 1` never
 * wraps from 0 to 0xffffffff (same pitfall as the JS reference).
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

uint u64_div32(uvec2 a, uint d, out uint rem) {
  float fa = float(a.y) * 4294967296.0 + float(a.x);
  int q = int(floor(fa / float(d)));
  if (q < 0) q = 0;
  uvec2 qd = u64_mul32(uint(q), d);
  while (u64_lt(a, qd)) {
    q -= 1;
    qd = u64_sub(qd, uvec2(d, 0u));
  }
  uvec2 dp1 = u64_add(qd, uvec2(d, 0u));
  while (!u64_lt(a, dp1)) {
    q += 1;
    qd = dp1;
    dp1 = u64_add(qd, uvec2(d, 0u));
  }
  uvec2 r64 = u64_sub(a, qd);
  rem = r64.x;
  return uint(q);
}

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
