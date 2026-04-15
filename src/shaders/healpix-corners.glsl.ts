// src/shaders/healpix-corners.glsl.ts
//
// GLSL vertex and fragment shaders for the HealpixCellsPrimitiveLayer.
//
// The vertex shader computes HEALPix cell corner positions entirely on the GPU,
// using the (t, u) projection-space approach validated in healpix-reference.ts.
// fp64 emulation (hi+lo float32 pairs) is used throughout the projection math
// for correctness at nside up to 262144.

export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

// ---------------------------------------------------------------------------
// Per-instance attributes — the 64-bit cell ID split into two uint32 words.
// ---------------------------------------------------------------------------
in uint cellIdLo;
in uint cellIdHi;

// Per-vertex quad template (static Geometry); not used in HEALPix math.
in vec3 positions;

// ---------------------------------------------------------------------------
// Varying
// ---------------------------------------------------------------------------
out vec4 vColor;

// ===========================================================================
// fp64 emulation — all arithmetic carried out as (hi, lo) float32 pairs.
// Convention: value ≈ hi + lo, with |lo| << |hi|.
// ===========================================================================

// Lift a float32 scalar to fp64.
vec2 f64_from(float a) {
  return vec2(a, 0.0);
}

// Two-sum: exact split of a + b into (hi, lo).
vec2 f64_add(vec2 a, vec2 b) {
  float s = a.x + b.x;
  float v = s - a.x;
  float e = (a.x - (s - v)) + (b.x - v) + a.y + b.y;
  return vec2(s, e);
}

// Subtraction via negation.
vec2 f64_sub(vec2 a, vec2 b) {
  return f64_add(a, vec2(-b.x, -b.y));
}

// Veltkamp-Dekker split then exact multiply.
vec2 f64_mul(vec2 a, vec2 b) {
  // Split each operand into two 26-bit halves.
  float c = (float(1 << 12) + 1.0) * a.x;
  float ahi = c - (c - a.x);
  float alo = a.x - ahi;

  float d = (float(1 << 12) + 1.0) * b.x;
  float bhi = d - (d - b.x);
  float blo = b.x - bhi;

  float p = a.x * b.x;
  float e = ((ahi * bhi - p) + ahi * blo + alo * bhi + alo * blo) +
            a.x * b.y + a.y * b.x;
  return vec2(p, e);
}

// fp64 division via Newton–Raphson refinement.
vec2 f64_div(vec2 a, vec2 b) {
  float q1 = a.x / b.x;
  // Compute residual: a - q1 * b
  vec2 r = f64_sub(a, f64_mul(vec2(q1, 0.0), b));
  float q2 = r.x / b.x;
  return f64_add(vec2(q1, 0.0), vec2(q2, 0.0));
}

// fp64 square root via Newton–Raphson on float32 seed.
vec2 f64_sqrt(vec2 a) {
  float x = inversesqrt(a.x);
  float y = a.x * x;
  // y ~ sqrt(a.x); one NR step in fp64
  vec2 y2 = f64_mul(vec2(y, 0.0), vec2(y, 0.0));
  vec2 r  = f64_sub(a, y2);
  float dx = r.x / (2.0 * y);
  return vec2(y + dx, y - (y + dx) + dx);
}

// ===========================================================================
// Bit-manipulation helpers
// ===========================================================================

// Software log2 (MSB position) — findMSB is not guaranteed in GLSL ES 3.00.
uint uint_log2(uint v) {
  uint r = 0u;
  if ((v & 0xFFFF0000u) != 0u) { v >>= 16u; r |= 16u; }
  if ((v & 0xFF00u)     != 0u) { v >>= 8u;  r |= 8u;  }
  if ((v & 0xF0u)       != 0u) { v >>= 4u;  r |= 4u;  }
  if ((v & 0xCu)        != 0u) { v >>= 2u;  r |= 2u;  }
  if ((v & 0x2u)        != 0u) {            r |= 1u;  }
  return r;
}

// De-interleave every other bit — Morton decode for one axis.
uint compact1by1(uint x) {
  x &= 0x55555555u;
  x = (x | (x >> 1u)) & 0x33333333u;
  x = (x | (x >> 2u)) & 0x0f0f0f0fu;
  x = (x | (x >> 4u)) & 0x00ff00ffu;
  x = (x | (x >> 8u)) & 0x0000ffffu;
  return x;
}

// Interleave bits of x into even positions (Morton encode, single axis).
uint spread1by1(uint x) {
  x &= 0x0000ffffu;
  x = (x | (x << 8u)) & 0x00ff00ffu;
  x = (x | (x << 4u)) & 0x0f0f0f0fu;
  x = (x | (x << 2u)) & 0x33333333u;
  x = (x | (x << 1u)) & 0x55555555u;
  return x;
}

// Morton-interleave ix (even bits) and iy (odd bits).
uint morton2d(uint ix, uint iy) {
  return spread1by1(ix) | (spread1by1(iy) << 1u);
}

// ===========================================================================
// NEST decode:  NEST pixel id → (face, ix, iy)
// Returns ivec3(face, ix, iy).
// ===========================================================================
ivec3 nest_to_xyf(uint idLo, uint idHi, uint k) {
  // face = id / nside^2 = id >> (2*k)
  // For k <= 16 the entire face index fits in idHi or the upper bits of idLo.
  uint face;
  uint remLo;  // lower 2k bits of id
  uint remHi;

  if (k <= 16u) {
    // 2k bits fit in a single uint32.
    uint bits2k = 2u * k;
    // face is in the high bits of a 64-bit value: (idHi << (32 - bits2k)) | (idLo >> bits2k)
    // But since k <= 16, bits2k <= 32.
    if (bits2k == 32u) {
      face  = idHi;
      remLo = idLo;
    } else {
      face  = (idHi << (32u - bits2k)) | (idLo >> bits2k);
      remLo = idLo & ((1u << bits2k) - 1u);
    }
    remHi = 0u;
  } else {
    // k > 16: 2k > 32, so the face index spans idHi.
    uint bits2k = 2u * k;          // > 32
    uint shift  = bits2k - 32u;    // how many bits of idHi are part of face (> 0)
    face  = idHi >> shift;
    remHi = idHi & ((1u << shift) - 1u);
    remLo = idLo;
  }

  // De-interleave the 2k-bit remainder into ix and iy.
  // For k <= 16 all bits are in remLo; for k > 16 upper bits spill into remHi.
  uint ix, iy;
  if (k <= 16u) {
    ix = compact1by1(remLo);
    iy = compact1by1(remLo >> 1u);
  } else {
    // Lower 32 bits: remLo holds 32 bits (16 bits per axis).
    uint ixLo = compact1by1(remLo);
    uint iyLo = compact1by1(remLo >> 1u);
    // Upper bits: remHi holds (2k-32) bits = 2*(k-16) bits → (k-16) bits per axis.
    uint ixHi = compact1by1(remHi);
    uint iyHi = compact1by1(remHi >> 1u);
    ix = (ixHi << 16u) | ixLo;
    iy = (iyHi << 16u) | iyLo;
  }

  return ivec3(int(face), int(ix), int(iy));
}

// ===========================================================================
// RING → NEST conversion
// Returns uvec2(nestLo, nestHi).
// ===========================================================================
uvec2 ring_to_nest(uint idLo, uint idHi, uint nside) {
  // ncap = 2 * nside * (nside - 1)  (first ring pixel index in equatorial belt)
  // Use 64-bit arithmetic to avoid overflow at large nside.
  // Represent ncap as uvec2(lo, hi).
  uint ns1     = nside - 1u;
  // 2 * nside * ns1 — may exceed 32 bits for nside >= 46341
  // Use the identity: 2*nside*(nside-1) = 2*(nside^2 - nside)
  // Compute nside^2 in 64-bit first.
  // nside fits in 19 bits (max nside=262144=2^18), so nside^2 fits in 37 bits.
  uint nsHi = 0u, nsLo = 0u;
  {
    // 64-bit multiply: nside * nside
    uint a = nside;
    uint b = nside;
    uint lo = a * b;
    // Compute high word via grade-school multiply.
    uint a_lo = a & 0xFFFFu;
    uint a_hi = a >> 16u;
    uint b_lo = b & 0xFFFFu;
    uint b_hi = b >> 16u;
    uint mid  = a_lo * b_hi + a_hi * b_lo;
    uint hi   = a_hi * b_hi + (mid >> 16u);
    // Add carry from lo computation.
    uint carry_check = a_lo * b_lo;
    uint mid_lo_sum  = (carry_check >> 16u) + (mid & 0xFFFFu);
    hi += (mid_lo_sum >> 16u);
    nsLo = lo;
    nsHi = hi;
  }
  // ncap = 2 * (nside^2 - nside) = 2*nside^2 - 2*nside
  // ncap_lo, ncap_hi  (unsigned 64-bit)
  uint ncapLo, ncapHi;
  {
    // Multiply nsLo/nsHi by 2.
    ncapHi = (nsHi << 1u) | (nsLo >> 31u);
    ncapLo = nsLo << 1u;
    // Subtract 2*nside.
    uint sub = 2u * nside;
    uint prevLo = ncapLo;
    ncapLo -= sub;
    if (ncapLo > prevLo) ncapHi -= 1u;  // borrow
  }

  // npix = 12 * nside^2
  uint npixLo, npixHi;
  {
    // 12 * nside^2 = 8*nside^2 + 4*nside^2
    uint lo8 = nsLo << 3u; uint hi8 = (nsHi << 3u) | (nsLo >> 29u);
    uint lo4 = nsLo << 2u; uint hi4 = (nsHi << 2u) | (nsLo >> 30u);
    npixLo = lo8 + lo4;
    npixHi = hi8 + hi4 + uint(npixLo < lo8);
  }

  // Compare id with ncap and npix-ncap using 64-bit comparisons.
  bool lt_ncap  = (idHi < ncapHi) || (idHi == ncapHi && idLo < ncapLo);
  // equatorial upper bound = npix - ncap
  uint eqUpLo, eqUpHi;
  {
    eqUpLo = npixLo - ncapLo;
    eqUpHi = npixHi - ncapHi - uint(eqUpLo > npixLo);
  }
  bool lt_eqUp  = (idHi < eqUpHi) || (idHi == eqUpHi && idLo < eqUpLo);

  int face;
  uint ix, iy;

  if (lt_ncap) {
    // -----------------------------------------------------------------------
    // NORTH POLAR CAP
    // id < ncap  =>  id fits in 32 bits (ncap < 2^37 but < 2^32 for nside<=23170)
    // For nside > 23170, id can exceed 32 bits.  Use float64 math for i.
    // ipix = idLo (since idHi == 0 for north cap when nside <= ~46340)
    // For larger nside we use float approximation for the sqrt; the integer
    // recovery is done with exact 32-bit residuals.
    // -----------------------------------------------------------------------
    // i = floor( (sqrt(1 + 2*ipix) + 1) / 2 )
    // For idHi == 0:
    float ipix_f = float(idLo) + float(idHi) * 4294967296.0;
    float i_f    = floor((sqrt(1.0 + 2.0 * ipix_f) + 1.0) / 2.0);
    // Clamp to valid range to handle float rounding.
    uint  i_u    = uint(i_f);
    // Verify: 2*i*(i-1) <= ipix < 2*i*(i+1)   (both in float to avoid overflow)
    float i2     = float(i_u);
    while (2.0 * i2 * (i2 - 1.0) > ipix_f) { i2 -= 1.0; }
    while (2.0 * i2 * (i2 + 1.0) <= ipix_f) { i2 += 1.0; }
    i_u = uint(i2);

    // j = ipix - 2*i*(i-1)
    float j_f = ipix_f - 2.0 * i2 * (i2 - 1.0);
    uint  j_u = uint(j_f);
    face   = int(j_u / i_u);
    uint k_u = j_u - uint(face) * i_u;
    ix = nside - i_u + k_u;
    iy = nside - 1u - k_u;

  } else if (lt_eqUp) {
    // -----------------------------------------------------------------------
    // EQUATORIAL BELT
    // k64 = id - ncap
    // -----------------------------------------------------------------------
    uint k64Lo = idLo - ncapLo;
    uint k64Hi = idHi - ncapHi - uint(k64Lo > idLo);

    // ring_size = 4 * nside
    uint ring_size = 4u * nside;
    // ring_size fits in 32 bits for nside <= 2^18.

    // For k64 that fits in 32 bits (k64Hi == 0):
    float k_f   = float(k64Lo) + float(k64Hi) * 4294967296.0;
    float ring_f = float(ring_size);
    float nr_f  = float(nside) - floor(k_f / ring_f);  // ring within equatorial strip (counting from nside downward)
    float s_f   = 1.0 - mod(nr_f, 2.0);               // 1 if even ring, 0 if odd — matches reference
    // j = 2*(k % ring_size) + s
    float kmod_f = mod(k_f, ring_f);
    float j_f    = 2.0 * kmod_f + s_f;

    float jj_f = j_f - 4.0 * float(nside);
    float ii_f = nr_f + 5.0 * float(nside) - 1.0;
    float pp_f = (ii_f + jj_f) / 2.0;
    float qq_f = (ii_f - jj_f) / 2.0;
    float PP   = floor(pp_f / float(nside));
    float QQ   = floor(qq_f / float(nside));
    float V    = 5.0 - (PP + QQ);
    float H    = PP - QQ + 4.0;
    face        = int(4.0 * V + mod(floor(H / 2.0), 4.0));
    ix = uint(mod(pp_f, float(nside)));
    iy = uint(mod(qq_f, float(nside)));

  } else {
    // -----------------------------------------------------------------------
    // SOUTH POLAR CAP
    // p = npix - 1 - id
    // -----------------------------------------------------------------------
    uint tempLo = npixLo - 1u;
    uint borrow1 = uint(npixLo == 0u);
    uint pLo = tempLo - idLo;
    uint borrow2 = uint(idLo > tempLo);
    uint pHi = npixHi - borrow1 - idHi - borrow2;
    // For the south cap pHi should be 0 for valid nside.
    float p_f = float(pLo) + float(pHi) * 4294967296.0;
    float i_f = floor((sqrt(1.0 + 2.0 * p_f) + 1.0) / 2.0);
    float i2  = i_f;
    while (2.0 * i2 * (i2 - 1.0) > p_f) { i2 -= 1.0; }
    while (2.0 * i2 * (i2 + 1.0) <= p_f) { i2 += 1.0; }
    uint i_u = uint(i2);
    float j_f = p_f - 2.0 * i2 * (i2 - 1.0);
    uint  j_u = uint(j_f);
    face   = 11 - int(j_u / i_u);
    uint k_u = j_u - uint(11 - face) * i_u;
    ix = i_u - k_u - 1u;
    iy = k_u;
  }

  // Encode (face, ix, iy) as a NEST pixel id.
  uint mortonBits = morton2d(ix, iy);
  // nest = face * nside^2 + mortonBits  (64-bit)
  uint nestLo, nestHi;
  {
    // 64-bit: face_val * nside^2
    uint f_u = uint(face);
    uint fnsLo = nsLo * f_u;
    uint fnsHi = nsHi * f_u;
    // Check carry from lo multiply.
    // (nsLo * f_u can overflow; compute carry via 16-bit pieces)
    uint ns_lo16 = nsLo & 0xFFFFu;
    uint ns_hi16 = nsLo >> 16u;
    uint mid16   = ns_lo16 * (f_u >> 16u) + ns_hi16 * (f_u & 0xFFFFu);
    fnsHi += (ns_hi16 * (f_u >> 16u)) + (mid16 >> 16u);
    nestLo = fnsLo + mortonBits;
    nestHi = fnsHi + uint(nestLo < fnsLo);
  }

  return uvec2(nestLo, nestHi);
}

// ===========================================================================
// fxy2ki — face coordinates (f, ix, iy) → integer diagonal coords (k, i).
// Returns ivec2(k, i).  These are the exact integer numerators in
//   t = (k / nside) · π/4,   u = π/2 − (i / nside) · π/4
// ===========================================================================
const float PI_4  = 0.78539816339744830962;   // π/4

ivec2 fxy2ki(int f, int ix, int iy, uint nside) {
  int f_row = f / 4;
  int f1    = f_row + 2;
  int f2    = 2 * (f % 4) - (f_row % 2) + 1;

  int v = ix + iy;
  int h = ix - iy;

  int i_int = f1 * int(nside) - v - 1;
  int k_int = f2 * int(nside) + h + 8 * int(nside);

  return ivec2(k_int, i_int);
}

// ===========================================================================
// ki2tu — integer diagonal coords (k, i) → fp64 projection coords (t, u).
// Returns vec4(t.hi, t.lo, u.hi, u.lo).
// ===========================================================================
vec4 ki2tu(int k_int, int i_int, uint nside) {
  vec2 n_f64 = f64_from(float(nside));

  vec2 t = f64_mul(f64_div(f64_from(float(k_int)), n_f64), vec2(PI_4, 0.0));
  vec2 u = f64_sub(vec2(PI_2, 0.0),
             f64_mul(f64_div(f64_from(float(i_int)), n_f64), vec2(PI_4, 0.0)));

  return vec4(t, u);
}

// ===========================================================================
// tu2za — inverse HEALPix projection: (t, u) → (z, a)
// Returns vec4(z.hi, z.lo, a.hi, a.lo) in fp64.
// ===========================================================================
vec4 tu2za(vec2 t, vec2 u) {
  float abs_u = abs(u.x);   // |u| ≈ u.x  (hi word sufficient for branching)
  float sign_u = sign(u.x);

  if (abs_u >= PI_2) {
    // Pole — out of valid range.
    return vec4(sign_u, 0.0, 0.0, 0.0);
  }

  if (abs_u <= PI_4) {
    // Equatorial belt: z = (8 / (3π)) * u,  a = t
    // 8/(3π) in fp64
    float c_hi = 8.0 / (3.0 * PI);
    vec2  c    = vec2(c_hi, 0.0);
    vec2  z    = f64_mul(c, u);
    return vec4(z, t);
  } else {
    // Polar caps.
    // t_t = t mod (π/2) — computed in fp64 to avoid the precision loss that
    // occurs when large t values (~12) are reduced in float32.  The ratio
    // multiplier grows as O(nside) near the poles, amplifying any error in
    // (t_t − π/4) into visible gaps between cells.
    float n_periods = floor(t.x / PI_2);
    vec2 t_t = f64_sub(t, f64_mul(vec2(PI_2, 0.0), f64_from(n_periods)));

    // a = t - ((abs_u - π/4) / (abs_u - π/2)) * (t_t - π/4)
    vec2 abs_u_f64 = vec2(abs_u, u.y * sign_u);
    vec2 num = f64_sub(abs_u_f64, vec2(PI_4, 0.0));
    vec2 den = f64_sub(abs_u_f64, vec2(PI_2, 0.0));
    vec2 ratio = f64_div(num, den);
    vec2 tt_off = f64_sub(t_t, vec2(PI_4, 0.0));
    vec2 a = f64_sub(t, f64_mul(ratio, tt_off));

    // sigma = 4 * |u| / π  — fp64 for latitude precision near the poles.
    vec2 sigma = f64_div(f64_mul(vec2(4.0, 0.0), abs_u_f64), vec2(PI, 0.0));
    vec2 two_minus_sigma = f64_sub(vec2(2.0, 0.0), sigma);
    vec2 tms2 = f64_mul(two_minus_sigma, two_minus_sigma);
    vec2 z = f64_mul(vec2(sign_u, 0.0),
               f64_sub(vec2(1.0, 0.0), f64_mul(vec2(1.0 / 3.0, 0.0), tms2)));

    return vec4(z, a);
  }
}

// ===========================================================================
// Main vertex program
// ===========================================================================
void main() {
  // -------------------------------------------------------------------------
  // Determine corner index: vertices within each instance cycle 0..3.
  // Index buffer pattern [0,1,2, 0,2,3] means gl_VertexID within the instance
  // gives the corner directly (deck.gl passes gl_VertexID per draw call).
  // -------------------------------------------------------------------------
  int cornerIdx = gl_VertexID % 4;

  // -------------------------------------------------------------------------
  // Resolve scheme: convert RING → NEST if needed.
  // -------------------------------------------------------------------------
  uint cLo = cellIdLo;
  uint cHi = cellIdHi;
  if (healpixCells.scheme == 1) {
    uvec2 nest = ring_to_nest(cLo, cHi, healpixCells.nside);
    cLo = nest.x;
    cHi = nest.y;
  }

  // -------------------------------------------------------------------------
  // NEST decode.
  // -------------------------------------------------------------------------
  uint k = uint_log2(healpixCells.nside);  // k = log2(nside)
  ivec3 xyf = nest_to_xyf(cLo, cHi, k);
  int f  = xyf.x;
  int ix = xyf.y;
  int iy = xyf.z;

  // -------------------------------------------------------------------------
  // Face coords → integer diagonal coords (k, i), then apply corner offset
  // as integer ±1 so that adjacent cells sharing a corner compute it from
  // the SAME (k, i) pair — guaranteeing bit-identical fp64 results.
  // -------------------------------------------------------------------------
  ivec2 ki = fxy2ki(f, ix, iy, healpixCells.nside);

  // Corner offsets in integer (k, i) space: [N, W, S, E]
  //   dt = [0, -1, 0, +1]  →  dk (horizontal)
  //   du = [+1, 0, -1, 0]  →  di = -du (u increases when i decreases)
  const int DK[4] = int[4]( 0, -1,  0, +1);
  const int DI[4] = int[4](-1,  0, +1,  0);

  int k_corner = ki.x + DK[cornerIdx];
  int i_corner = ki.y + DI[cornerIdx];

  vec4 tu_corner = ki2tu(k_corner, i_corner, healpixCells.nside);
  vec2 t_corner  = tu_corner.xy;
  vec2 u_corner  = tu_corner.zw;

  // -------------------------------------------------------------------------
  // Inverse projection: (t, u) → (z, a).
  // -------------------------------------------------------------------------
  vec4 za  = tu2za(t_corner, u_corner);
  vec2 z   = za.xy;
  vec2 a   = za.zw;

  // -------------------------------------------------------------------------
  // Convert (z, a) → (lon_deg, lat_deg) in fp64.
  //
  // Longitude: multiply fp64 radian pair by (180/π) to get degrees.
  // The lo word carries the sub-float32-ULP residual needed by project_position.
  // -------------------------------------------------------------------------
  const float DEG_PER_RAD = 57.29577951308232;  // 180 / π

  // fp64 lon in degrees
  vec2 lon_deg_f64 = f64_mul(a, vec2(DEG_PER_RAD, 0.0));

  // Normalize longitude hi word to (-180, 180] — keep lo word unchanged.
  float lon_hi = lon_deg_f64.x;
  float lon_wrap = 360.0 * floor((lon_hi + 180.0) / 360.0);
  lon_hi -= lon_wrap;
  vec2 lon_f64 = vec2(lon_hi, lon_deg_f64.y);

  // fp64 latitude in degrees.
  // asin(z_hi) gives the float32 result; the fp64 correction is:
  //   asin(z_hi + z_lo) ≈ asin(z_hi) + z_lo / sqrt(1 - z_hi²)
  // i.e. z_lo scaled by the asin derivative (1/cos(lat)).
  float z_hi = clamp(z.x, -1.0, 1.0);
  float lat_rad_hi = asin(z_hi);
  float cos_lat = sqrt(max(1.0 - z_hi * z_hi, 1e-12));
  float lat_rad_lo = z.y / cos_lat;
  vec2 lat_deg_f64 = f64_mul(vec2(lat_rad_hi, lat_rad_lo), vec2(DEG_PER_RAD, 0.0));

  // -------------------------------------------------------------------------
  // Pass to deck.gl projection pipeline using the fp64 two-argument overload:
  //   project_position(vec4 position, vec3 position64Low)
  // This preserves sub-float32 precision for high nside (e.g. 262144) where
  // cell corners are ~3.5e-5° apart and float32 ULP at lon≈180° is ~2e-5°.
  // -------------------------------------------------------------------------
  vec4 position = vec4(lon_f64.x, lat_deg_f64.x, 0.0, 1.0);
  geometry.position = position;
  vec4 projected = project_position(position, vec3(lon_f64.y, lat_deg_f64.y, 0.0));
  gl_Position = project_common_position_to_clipspace(projected);

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
  // geometry is injected by deck.gl's shader system
  DECKGL_FILTER_COLOR(fragColor, geometry);
}
`;
