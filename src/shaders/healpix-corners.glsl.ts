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
  // integer fxy2tu (exact)
  int f_row = face / 4;
  int f1 = f_row + 2;
  int f2 = 2 * (face - 4 * f_row) - (f_row & 1) + 1;

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
}
`;
