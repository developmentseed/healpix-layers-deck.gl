/**
 * GPU decoders for NEST and RING HEALPix cell IDs.
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
  uvec2 i2 = u64_shl(u64_mul32(i, i - 1u), 1u);
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
  uint qu = u64_div32(kFull, ring, kmod);
  int q = int(qu);
  int i = int(nside) - q;
  uint s = (i & 1) == 0 ? 1u : 0u;
  uint j = 2u * kmod + s;
  int jj = int(j) - 4 * int(nside);
  int ii = i + 5 * int(nside) - 1;
  float pp = float(ii + jj) * 0.5;
  float qq = float(ii - jj) * 0.5;
  float fn = float(nside);
  uint PP = uint(floor(pp / fn));
  uint QQ = uint(floor(qq / fn));
  uint V = 5u - (PP + QQ);
  int H = int(PP) - int(QQ) + 4;
  uint face = 4u * V + (uint(H >> 1) & 3u);
  uint ix = uint(pp - float(PP) * fn);
  uint iy = uint(qq - float(QQ) * fn);
  return uvec3(face, ix, iy);
}

uvec3 decodeRingSouth(uvec2 cellId, uint nside, uvec2 npix) {
  uvec2 p = u64_sub(u64_sub(npix, cellId), uvec2(1u, 0u));
  uvec2 onePlus2p = u64_add(u64_shl(p, 1u), uvec2(1u, 0u));
  uint root = u64_isqrt(onePlus2p);
  uint i = (root + 1u) / 2u;
  uvec2 i2 = u64_shl(u64_mul32(i, i - 1u), 1u);
  uvec2 jFull = u64_sub(p, i2);
  uint j = jFull.x;
  uint fDiv = j / i;
  uint f = 11u - fDiv;
  uint k = j - fDiv * i;
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
