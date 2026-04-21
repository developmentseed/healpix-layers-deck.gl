/**
 * Dekker / double-single GLSL fragment (concatenate into vertex shader sources).
 */
export const FP64_GLSL: string = /* glsl */ `\
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
`;
