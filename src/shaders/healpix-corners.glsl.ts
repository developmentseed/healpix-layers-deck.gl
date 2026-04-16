export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

in uint faceIx;
in uint instIy;
in vec3 positions;

out vec4 vColor;

// ---------------------------------------------------------------------------
// Inline Dekker / double-single arithmetic (no external module dependency).
// Each "fp64" value is a vec2 (hi, lo) with hi = rounded fp32 and lo = residual.
//
// CRITICAL: Dekker primitives depend on strict fp32 round-to-nearest semantics.
// The GLSL optimizer can (and WILL) algebraically simplify these to return
// lo = 0, e.g. _twoSum: err = (a - (s - bb)) + (b - bb) with bb = s - a
// algebraically reduces to (a - a) + 0 = 0. Empirically measured on WebGL2:
// without barriers, every Dekker lo came back as 0 on the polar branch.
//
// Fix: round-trip the load-bearing intermediate through uintBitsToFloat ∘
// floatBitsToUint. Mathematically identity, but the compiler treats the
// bitcast as opaque, so the intermediate materializes as an honest fp32
// value and algebraic identity-chains break.
// ---------------------------------------------------------------------------
float _seal(float x) { return uintBitsToFloat(floatBitsToUint(x)); }

// Veltkamp split: a = hi + lo with 12-bit mantissas each
vec2 _split(float a) {
  const float SPLIT = 4097.0; // 2^12 + 1
  float t = _seal(a * SPLIT);
  float hi = t - (t - a);
  float lo = a - hi;
  return vec2(hi, lo);
}

// Kahan-Knuth twoSum (no ordering assumption)
vec2 _twoSum(float a, float b) {
  float s  = _seal(a + b);
  float bb = _seal(s - a);
  float err = (a - (s - bb)) + (b - bb);
  return vec2(s, err);
}

// quickTwoSum: requires |a| >= |b|
vec2 _qts(float a, float b) {
  float s = _seal(a + b);
  float err = b - (s - a);
  return vec2(s, err);
}

// twoProd using Veltkamp splitting
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
  vec2 yn  = a * xn;                          // first approximation
  float diff = _sub64(a, _mul64(b, yn)).x;
  vec2 corr  = _twoProd(xn, diff);
  return _add64(yn, corr);
}

vec2 _mul64f(vec2 a, float b) { return _mul64(a, vec2(b, 0.0)); }

// fp64 constants (hi = float32(x), lo = x - float32(x))
const vec2 PI64   = vec2( 3.1415927,  -8.742278e-8 );  // π
const vec2 PI2_64 = vec2( 1.5707964,  -4.371139e-8 );  // π/2
const vec2 PI4_64 = vec2( 0.78539819, -2.1855695e-8);  // π/4

// ---------------------------------------------------------------------------
void main() {
  int face = int(faceIx >> 24u);
  int ix   = int(faceIx & 0xFFFFFFu);
  int iy   = int(instIy);

  // Corner selection: gl_VertexID cycles 0..3 via index buffer [0,1,2,0,2,3]
  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  // integer fxy2tu (exact)
  int f_row = face / 4;
  int f1 = f_row + 2;
  int f2 = 2 * (face - 4 * f_row) - (f_row & 1) + 1;
  int nside = int(healpixCells.nside);

  // Intentionally omits the -1 that healpix-ts fxy2tu has. fxy2tu returns the
  // *center* of pixel (cx, cy); we want the four *corners* of pixel (ix, iy).
  // Using integer (cx, cy) = (ix + {0,1}, iy + {0,1}) with this formula, the
  // quad's four vertices land at (t0, u0±d) and (t0±d, u0) — exactly the N/W/S/E
  // cardinal corners produced by healpix-ts fxyCorners. Adding a -1 here would
  // shift every corner by +d in u, translating the whole cell half a pixel north.
  int i_ring = f1 * nside - cx - cy;
  int k_ring = f2 * nside + (cx - cy) + 8 * nside;

  // Wrap k_ring to (-4*nside, 4*nside] in integer space so t = k_ring·π/(4·nside)
  // stays in (-π, π]. Since 8·nside · π/(4·nside) = 2π *exactly*, this wrap is
  // bit-exact — it avoids the catastrophic cancellation that killed precision
  // when t_fp reached ~12 rad before being reduced to ~0.16 rad.
  int period = 8 * nside;
  k_ring = k_ring - (k_ring / period) * period;
  if (k_ring > 4 * nside) k_ring -= period;

  // Split k_ring = k_int * nside + k_rem. nside is a power of two so
  // k_rem / nside is exact in fp32.
  int k_int = k_ring / nside;
  int k_rem = k_ring - k_int * nside;
  int i_int = i_ring / nside;
  int i_rem = i_ring - i_int * nside;
  float k_frac = float(k_rem) / float(nside);
  float i_frac = float(i_rem) / float(nside);

  // t = (k_int + k_frac) * PI/4  in fp64
  vec2 t_fp = _mul64f(PI4_64, float(k_int));
  t_fp      = _add64(t_fp, _mul64f(PI4_64, k_frac));

  vec2 i_ang = _mul64f(PI4_64, float(i_int));
  i_ang      = _add64(i_ang, _mul64f(PI4_64, i_frac));
  vec2 u_fp  = _sub64(PI2_64, i_ang);

  // tu2za ∘ asin composed in fp64: each branch emits lat_rad_fp / lon_rad_fp
  // directly. Structuring it this way lets the polar branch sidestep asin's
  // catastrophic ULP amplification near z=1 (at lat≈85° the naive chain turns
  // a 1-ULP fp32 error in z into ~16 ULPs in lat because d(asin)/dz = sqrt(3)/|s|).
  float u_hi  = u_fp.x;
  float abs_u = abs(u_hi);

  vec2 lat_rad_fp, lon_rad_fp;
  if (abs_u >= PI2_64.x) {
    // Pole exactly: lat = ±π/2 with the full fp64 residual; lon = 0.
    float sgn = sign(u_hi);
    lat_rad_fp = vec2(sgn * PI2_64.x, sgn * PI2_64.y);
    lon_rad_fp = vec2(0.0);
  } else if (abs_u <= PI4_64.x) {
    // Equatorial: z = (8/3π)·u in fp64, a = t in fp64, then asin(z) with
    // one Newton step (GLSL spec: asin ≤ 4 ULPs, sin/cos ≤ 4 ULPs each).
    //   lat_new = lat_old - (sin(lat_old) - z_true) / cos(lat_old)
    // where z_true = z_hi + z_lo. |u| ≤ π/4 ⇒ |lat| ≤ arcsin(2/3) ≈ 41.8°
    // so cos_lat ≥ 0.745 — no pole guard needed in this branch.
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
    // Polar cap. Canonical tu2za formula:
    //   s = 2 - 4|u|/π         (|s| ≤ 1, shrinks to 0 at the pole)
    //   z = sign(u)·(1 - s²/3)
    //   a = t - ((|u|-π/4)/(|u|-π/2))·((t mod π/2) - π/4)
    //
    // Computing z then asin(z) is cursed: near z=1 we have
    //   d(asin)/dz = 1/sqrt(1-z²) = sqrt(3)/|s|
    // so the 1 fp32 ULP of noise from forming (1 - s²/3) gets blown up by
    // 1/|s| — ~16× at lat 85°, way past the 4-ULP spec budget for asin.
    //
    // Fix: use the half-angle identity
    //   asin(1 - 2w²) = π/2 - 2·asin(w)      (0 ≤ w ≤ 1)
    // Pick w so 2w² = s²/3, i.e. w = |s|/sqrt(6). Then:
    //   |lat| = π/2 - 2·asin(w)    →    lat = sign(u)·|lat|
    // For |s| ≤ 1 → w ≤ 0.4082 so θ = asin(w) ≤ 0.421 rad, giving
    // cos(θ) ≥ 0.91 — Newton on sin(θ) - w = 0 is well-conditioned here
    // (unlike at the pole itself where cos(lat)→0).
    //
    // asin refinement (Plan B):
    //   θ₀ = asin(w)                              GLSL spec: 4 ULP on θ
    //   θ_hi = θ₀ - (sin(θ₀) - w) / cos(θ₀)       one Newton step → ≲ sin-spec noise
    //   θ_lo = -(sin(θ_hi) - w) / cos(θ_hi)       residual captured as Dekker lo
    // δ = 2·asin(w) then trivially becomes (2·θ_hi, 2·θ_lo) — a power-of-2
    // scale is exact in fp32 so the Dekker pair scales componentwise.
    float sgn = sign(u_hi);
    float s   = 2.0 - 4.0 * abs_u / PI64.x;
    const float INV_SQRT_6 = 0.40824829046386; // 1/sqrt(6), rounded to fp32
    float w    = abs(s) * INV_SQRT_6;
    // Seal asin outputs so the compiler can't simplify sin(asin(w)) ≡ w and
    // collapse the Newton correction to 0. Same trick as in the Dekker
    // primitives: bitcast round-trip is an opaque identity.
    float a0   = _seal(asin(w));
    float a_hi = _seal(a0 - (sin(a0) - w) / cos(a0));
    float a_lo = -(sin(a_hi) - w) / cos(a_hi);

    vec2 delta_fp   = vec2(2.0 * a_hi, 2.0 * a_lo);
    vec2 lat_mag_fp = _sub64(PI2_64, delta_fp);
    lat_rad_fp      = vec2(sgn * lat_mag_fp.x, sgn * lat_mag_fp.y);

    // lon: the polar a-formula involves a division of small differences near
    // the pole, but t itself is well-separated from the knee at |u|=π/4,
    // so fp32 here is sufficient. (Empirically the bigger ULPs at the pole
    // are all in lat, not lon.)
    float t_hi = t_fp.x;
    float t_t  = mod(t_hi, PI2_64.x);
    float a_f  = t_hi - ((abs_u - PI4_64.x) / (abs_u - PI2_64.x))
                         * (t_t - PI4_64.x);
    lon_rad_fp = vec2(a_f, 0.0);
  }

  // lon_rad is already in (-π, π] thanks to the integer k_ring wrap above,
  // so no Dekker-subtraction of 2π is needed — that step would introduce
  // cancellation noise we can't afford.

  // rad -> deg in fp64
  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);

  // Hand off to deck.gl using the fp64-aware projection entry point so that
  // the (lon_lo, lat_lo) residuals survive the AUTO_OFFSET Mercator subtraction.
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
