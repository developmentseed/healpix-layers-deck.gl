# High-precision GPU HEALPix corners

## What we were doing

Rendering HEALPix sky-cell corners on the GPU, at high resolution (nside up to 262144) and high map zoom (up to 18). Each cell is a diamond-ish polygon in `(lon, lat)` space. The vertex shader takes a cell id, decodes it into `(f, x, y)` (face plus local coordinates in that face), runs the HEALPix inverse projection to recover `(lon, lat)` at each of the four corners, and hands the result to deck.gl to place it on the Mercator map.

fp32 is all we have on WebGL2. That's the constraint the whole writeup is about.

## What was broken

At zoom 18, the GPU corners drifted off the CPU-truth reference in a way that was very noticeable.

## A few definitions before we go further

Some terms I'll be using, so the math below stops feeling like alphabet soup.

**ULP** — "unit in the last place." The smallest fp32 quantum at a given magnitude. Two adjacent representable fp32 numbers differ by exactly 1 ULP. Strictly, 1 ULP = `2^(e − 23)` where `e` is the value's unbiased exponent, so it's constant across each binade `[2ᵉ, 2ᵉ⁺¹)` and jumps by 2× at each power-of-two boundary. A loose ballpark is `2⁻²³ × |value| ≈ 1.2 × 10⁻⁷ of the value`, accurate within a factor of 2. When I say "10 ULP error," I mean the computed answer is 10 fp32 quanta away from truth.

**Dekker double-single (or double-fp32)** — a trick for getting roughly double the precision without native fp64. You represent a number as a pair `(hi, lo)` of fp32 floats where the true value is `hi + lo` and `|lo| ≤ ½ ulp(hi)`. The pair carries about 48 bits of effective precision, against fp32's 24 (counting the implicit leading 1 in both cases). Named after T. J. Dekker's 1971 paper.

**Dekker primitives** — the exact-arithmetic building blocks that keep that invariant intact. The main ones:
- `twoSum(a, b)` returns `(s, err)` where `s = a + b` computed in fp32 and `err` is the rounding error that `s` dropped on the floor, such that `a + b = s + err` is exact.
- `twoProd(a, b)` is the same idea for multiplication. It relies on `split`, which breaks an fp32 into two halves small enough to multiply without overflowing the mantissa.
- `add64`, `sub64`, `mul64`, `div64` compose the above into full pair-valued arithmetic.

**Newton refinement** — one step of Newton's root-finding method, used here to improve the precision of a single fp32 estimate. For `asin(w)`: if `θ₀ ≈ asin(w)` is your fp32 estimate, solving `sin(θ) = w` with Newton's method gives `θ₁ = θ₀ − (sin θ₀ − w) / cos θ₀`. Because Newton converges quadratically, one step typically doubles the number of correct bits. The correction `−(sin θ₀ − w) / cos θ₀` (note the minus — it's what you *add* to θ₀ to get θ₁) is exactly the shape you want for the `lo` component of a Dekker pair.

**Bitcast barrier (a.k.a. "seal")** — a small trick that stops the compiler from algebraically simplifying an expression. You take a float, reinterpret its bits as a uint (not a value change, just a reinterpretation), then reinterpret back to float. The value is identical. But as far as the compiler's optimizer is concerned, the result is opaque: it can't prove any algebraic relationship between the input and the output, so it can't apply identities like `sin(asin(x)) → x`, or the twoSum collapse `(a − (s − bb)) + (b − bb) → 0` when it knows `bb = s − a` and `s = a + b`. More on why we need that below.

**The HEALPix intermediate variables** — the projection math goes through a few named stages:
- `(f, x, y)` — face index (0..11) plus integer local coordinates inside that face (0..nside−1).
- `(u, t)` — continuous UV-style coordinates derived from `(f, x, y)`. `|u|` picks the branch: `|u| ≤ π/4` is the equatorial belt, `|u| > π/4` is one of the polar caps.
- `(z, a)` — `z` ends up as `sin(lat)`, `a` ends up as longitude (with some per-face offset). So the last step of the inverse is `lat = asin(z)`, `lon = a`.
- `s = 2 − 4|u|/π` — a remapping of `|u|` inside the polar cap that shrinks to 0 at the pole. Convenient because the polar formula is cleaner in `s`.
- `w = |s|/√6` — the argument we'll pass to `asin` after applying the half-angle trick. Small for all relevant inputs.

With those in hand, the rest makes sense.

## Why it was broken

Two things were going wrong at the same time, both contributing to the drift visible at zoom 18. I'll walk through each one, but they aren't ranked by severity: both are load-bearing, and fixing only one still leaves corners off the reference.

### Cause 1 — `asin` amplifies fp32 noise in the polar branch

Quick reminder of the variables from the definitions section, since the next few paragraphs lean on them:
- `u` — the UV-style coordinate whose magnitude picks the branch: equatorial if `|u| ≤ π/4`, polar otherwise.
- `z` — the intermediate that ends up as `sin(lat)`, so the final step is `lat = asin(z)`.
- `s = 2 − 4|u|/π` — a remapping of `|u|` inside the polar cap. 0 at the pole, 1 at the equator-polar boundary. The polar formula is cleaner in `s` than in `u`.

The HEALPix inverse splits by branch.

Equatorial (`|u| ≤ π/4`):
```
z = (8 / 3π) · u
```
`z` stays bounded well away from ±1, so `asin(z)` is well-behaved.

Polar (`|u| > π/4`):
```
s   = 2 − 4·|u| / π        // 0 at the pole, 1 at the equator-polar boundary
z   = sign(u) · (1 − s² / 3)
lat = asin(z)
```
As `s → 0`, `z → 1`, and `asin` becomes a cliff. Its derivative is `d(asin)/dz = 1/√(1 − z²)`, which blows up at `z = ±1`. Expanding in terms of `s`:
```
1 − z² = (s²/3) · (2 − s²/3)
1 / √(1 − z²) = √3 / (|s| · √(2 − s²/3))
```
For small `s`, `√(2 − s²/3) → √2`, so the factor is about `√(3/2) / |s| = √6 / (2|s|) ≈ 1.225 / |s|`. That's a radian-of-z to radian-of-lat amplification. The ULP-to-ULP amplification is half as large: `z ∈ [0.5, 1)` has `ulp(z) = 2⁻²⁴` while `lat ∈ [1, 2)` has `ulp(lat) = 2⁻²³`, so `ulp(z)/ulp(lat) = 1/2`. The ULP-to-ULP factor is therefore `√(3/2) / (2|s|) ≈ 0.612 / |s|`.

For the polar test cell at lat ≈ 85° (cell id `274626645138`, nside=262144): `z = sin 85° ≈ 0.9962`, so `s = √(3(1 − z)) ≈ 0.107`. Plugging in: radian amplification factor ≈ `√(3/2) / 0.107 ≈ 11.4×`; ULP-to-ULP factor ≈ `11.4 / 2 ≈ 5.7×`. Compound that with the rounding inside `1 − s²/3` and you get several ULPs of lat error stacked up by the time `asin` is done. This cause is branch-specific: it doesn't affect equatorial cells.

### Cause 2 — fp32 isn't precise enough at nside=262144

At equatorial cells like the Lisbon test cell (lat ≈ 38.7°, cell id `228532280344`), the `asin` derivative is around 1.28, so there's no dramatic amplification. The drift here is plain fp32 running out of mantissa bits for the cell resolution.

A HEALPix cell at nside=N has average angular side length `√(π/3) / N` radians. At N=262144 that's about 3.9 × 10⁻⁶ rad, which at Earth's radius (~6.37 × 10⁶ m) comes out to roughly 25 m on the ground. fp32 gives you ~23 mantissa bits, which at lat ≈ 0.67 rad works out to roughly 8 × 10⁻⁸ rad per ULP — about half a meter on the ground. So one ULP is ~2% of a cell side, and the projection chain has enough rounding steps that a handful of ULPs adds up to a visible fraction of a cell by the time the shader is done.

The plan for this one was Dekker double-single: carry a `(hi, lo)` pair through the projection so the output has ~48 bits of effective precision instead of fp32's 24. That's where the second problem showed up.

GLSL compilers know that `sin(asin(w))` equals `w` and that `a − (s − (s − a))` equals `0`. They don't know you were exploiting the fact that these identities only hold in exact real arithmetic, not in fp32. So the compiler cheerfully simplifies:

- `sin(asin(w)) → w`, which zeroes out any Newton correction built on that shape.
- `(a − (s − bb)) + (b − bb) → 0` when `bb = s − a`, which zeroes out the twoSum error term.
- `t − (t − a) → a`, which zeroes out the Veltkamp split's high half.

Under any of these simplifications, a Dekker chain silently collapses back to plain fp32 arithmetic while still type-checking and still returning values. The `lo` components all end up as exactly zero. You've done the work, the compiler has undone it, and there's no compile-time signal.

So cause 2 was: fp32 isn't enough, Dekker should work, but the compiler keeps deleting the Dekker part.

## How we actually saw any of this: the readback harnesses

Visual comparison against a CPU reference polygon was the thing that first told us something was wrong. But visual testing is useless for figuring out *what* is wrong. You see "the corner is about 3 pixels northwest of where it should be" and you can't tell whether that's `lat` or `lon` off, whether it's the hi part or the lo part of a Dekker pair, whether it's the asin or the projection step upstream, or whether the shader is even running the branch you think it's running. Pixels round away most of the information you need.

So we built two test harnesses:

- `tmp/gpu-readback.html` — equatorial case, Lisbon-area cell at lat ≈ 38.7°, nside=262144.
- `tmp/gpu-readback-polar.html` — polar case, cell at lat ≈ 85°, lon ≈ −38°, nside=262144.

Each harness is a single HTML file that:

1. Uses WebGL2 transform feedback plus `RASTERIZER_DISCARD` to run the exact production vertex shader on a single cell id, capture the raw fp32 outputs (`lon_hi`, `lon_lo`, `lat_hi`, `lat_lo` for each of the four corners), and pull them back to JavaScript. No fragment shader runs. No pixels. Just the numbers the shader actually emitted.

2. Holds a `TRUTH` table at the top of the file: the CPU-computed ground truth for the four corners, generated by running the healpix-ts `fxyCorners(nside, f, x, y)` routine at f64 precision. This is what the GPU answer is supposed to match after reassembling `hi + lo`.

3. Diffs GPU against TRUTH and prints a table in ULPs. One row per `{corner, axis, hi/lo}` slot. The ULP metric is the key thing here — it lets you talk precisely about what "close" means. "0 ULPs" means byte-identical to truth. "1 ULP" means the last representable mantissa bit is off by one, which is the best you can hope for from a single fp32. A handful of ULPs at zoom 18 is enough to move a corner off its reference.

Why this mattered in practice — concrete examples of things the harnesses caught that we would have missed otherwise:

- **Separating the two causes.** Before any fix, the polar harness and the equatorial harness both showed non-zero ULP diffs, but the polar harness showed considerably more on `lat_hi`. Having the two numbers side-by-side, measured in the same unit, was what let us attribute part of the drift specifically to `asin`-near-1 amplification (which only the polar branch hits) versus the general fp32 floor (which both branches hit). Without that split we would have been guessing at which fix attacks which root cause.

- **"Plan B did nothing" was invisible without the harness.** After adding Newton refinement (Layer 2 below), the shader looked structurally correct and compiled without error. Re-running the polar harness: the output was *byte-identical* to the pre-Plan-B run. Not "a bit better, not quite enough," but literally the same bits. That's the unambiguous signature of the compiler algebraically simplifying the correction away. Without the readback we would have stared at barely-different pixels for hours trying to figure out whether Plan B helped a little, helped a lot, or did nothing.

- **Seals were partially working.** After adding bitcast barriers (Layer 3), the harness showed one corner's `lat_hi` drop from 1 ULP to 0, and a different corner's `lon_hi` *regress* from 2 ULPs to 3. Mixed results like that are the fingerprint of "the compiler respected the seal in some call paths but not others." Without the ULP-level readback you'd just see "still kind of drifty" and have no idea whether the fix was working at all.

- **`lo` is always zero.** All `lat_lo` and `lon_lo` slots on both harnesses, on every fix iteration so far, come back as exactly 0.0. This is the clearest evidence we have that the compiler is still collapsing the Dekker chain somewhere we haven't found. A single nonzero `lo` would tell us the pair-arithmetic path is alive. Zero across the board tells us it isn't. That's the signal the next round of work needs to chase.

The harnesses are tiny, stand-alone, and run in Chrome by double-clicking. They became the main feedback loop — change the shader, refresh the page, read the ULP table. Visual testing went back to being the final sanity check.

## What fixed it

Three layers, applied in order. Each one builds on the previous one.

### Layer 1 — half-angle identity for the polar branch

Instead of computing `asin(z)` where `z` is close to 1 (the cliff), use the identity:

```
asin(1 − 2w²) = π/2 − 2·asin(w),   for w ∈ [0, 1]
```

Pick `w = |s| / √6`. Then `2w² = s²/3 = 1 − |z|`, so `|z| = 1 − 2w²` and the identity gives `asin(|z|) = π/2 − 2·asin(w)`. The overall `sign(u)` on the outside handles the southern polar cap by flipping the whole latitude:

```
lat = sign(u) · (π/2 − 2 · asin(w))
```

Near the pole, `w ≈ 0.044` (for the test cell: `s ≈ 0.107`, `w = s/√6 ≈ 0.044`). `asin(0.044)` is nowhere near the cliff. Its derivative is about 1.001, so there's effectively no amplification. We've traded a numerically nasty `asin` for a numerically trivial one.

This change brought the polar branch down to the same fp32-floor behaviour as the equatorial branch — a couple of ULPs of residual, no more amplification. After this, both branches had the same remaining problem to solve.

### Layer 2 — Newton refinement for the `lo` component

Given `θ₀ = asin(w)` in fp32, Newton's method improves the estimate by subtracting the residual:

```
θ_true ≈ θ₀ − (sin θ₀ − w) / cos θ₀
```

The correction term `(sin θ₀ − w) / cos θ₀` is tiny (on the order of 1 ULP of `θ₀`) and carries the bits that `asin` dropped on the floor. The implementation stores the Newton-improved estimate as `θ_hi` and the residual at that improved estimate as `θ_lo`: `θ_hi = θ₀ − (sin θ₀ − w) / cos θ₀` and `θ_lo = −(sin θ_hi − w) / cos θ_hi`. The pair `(θ_hi, θ_lo)` gives about 48 bits of effective precision for `lat`, against fp32's 24. Strictly, the Dekker invariant `|θ_lo| ≤ ½ ulp(θ_hi)` isn't guaranteed without a renormalizing `quickTwoSum`, but since `θ_hi` has already absorbed one Newton step, `θ_lo` is a second-order correction and in practice satisfies the invariant.

As noted above, this layer did nothing on its own, because the compiler simplified `sin(asin(w))` back to `w` and zeroed out the correction. That's what Layer 3 is for.

### Layer 3 — bitcast barriers (seals)

Define:

```glsl
float _seal(float x) {
    return uintBitsToFloat(floatBitsToUint(x));
}
```

Value unchanged, algebraic opacity restored. Now the compiler sees `sin(_seal(asin(w)))` and can't fold it, because it can't see through the bitcast round-trip.

Apply `_seal` to the load-bearing intermediates in the Newton step:

```glsl
float a0   = _seal(asin(w));                            // seal asin output
float a_hi = _seal(a0 − (sin(a0) − w) / cos(a0));      // seal Newton result
float a_lo = −(sin(a_hi) − w) / cos(a_hi);             // residual → lo
```

And inside the Dekker primitives, wherever an algebraic identity could collapse the error term. `twoSum` is the canonical example:

```glsl
vec2 _twoSum(float a, float b) {
    float s   = _seal(a + b);
    float bb  = _seal(s − a);
    float err = (a − (s − bb)) + (b − bb);
    return vec2(s, err);
}
```

Without the seal on `bb`, the compiler could substitute `bb = s − a` back into the `err` expression and simplify it to 0. With the seal, it can't prove `bb` equals `s − a` anymore, so `err` survives.

This brought the residual down further. Polar worst case ended up at ~0.82 ULPs on `lat_hi`, with one outlier on `lon_hi` at 3 ULPs. Equatorial harness came in under 1 ULP across all slots. Not perfect, but good enough for the current goal.

## Where this ended up

Both harnesses agree within a few ULPs of truth on `hi`. Polar: 0.82 ULPs on `lat_hi`, 3 ULPs on `lon_hi` for the worst corner. Equatorial: under 1 ULP across the board. The two branches now have the same failure mode — the fp32 floor with a compiler-eaten Dekker `lo` — rather than the polar branch being in its own category.

Converting to pixels at zoom 18 lat 85°: Mercator's conformal stretch at lat 85° is `1/cos 85° ≈ 11.47×`. That converts *radians of lat-error* into pixels, but doesn't give the lat-vs-lon ULP ratio on its own. `lat ≈ 1.484 rad ∈ [1, 2)` has `ulp(lat) = 2⁻²³`; `lon ≈ −0.663 rad` with `|lon| ∈ [0.5, 1)` has `ulp(lon) = 2⁻²⁴`. The exact ULP ratio is `2⁻²³ / 2⁻²⁴ = 2`. Net: a single lat-ULP at this location costs about `11.47 × 2 ≈ 23×` more pixels than a single lon-ULP. Which is why 0.82 lat-ULPs (~12 px) outweighs 3 lon-ULPs (~2 px) for pixel impact. At zoom 14 and below both contributions are subpixel. The polar corner is the only place on the globe where the drift is visible at maximum zoom; everywhere else the residual sits safely inside a pixel.

The `lo` component of the Dekker pair is still arriving at the output as exactly 0, on both harnesses, for every corner, in every run. The Newton correction and the pair arithmetic are running in the source, but the compiler is finding a way to collapse them at some level we haven't located yet. Fixing that would bring both branches to sub-ULP on `hi`, but it's the open problem.

---

## Future improvements (AI assistant handoff)

Current state:
- `src/shaders/healpix-corners.glsl.ts` — production vertex shader. Contains Layer 1 (half-angle identity), Layer 2 (Newton refinement), Layer 3 seals on Dekker primitives and asin outputs.
- `tmp/gpu-readback.html` — equatorial harness, cell `228532280344` (Lisbon, lat 38.7°, nside=262144).
- `tmp/gpu-readback-polar.html` — polar harness, cell `274626645138` (lat 85°, lon −38°, nside=262144).

Both harnesses mirror the production shader exactly. Transform feedback + `RASTERIZER_DISCARD`. TRUTH at the top of each file is generated by `healpix-ts` `fxyCorners(nside, f, x, y)` at f64. Regenerate TRUTH with `node tmp/compute-truth.mjs` if you touch nside or cell ids.

Current ULP table (polar harness):
- `ci=0 lat_hi`: 0 ULP, `lat_lo`: 0 (should be nonzero if Dekker survived)
- `ci=1 lat_hi`: 0.82 ULP, `lat_lo`: 0
- `ci=3 lon_hi`: 3 ULP, `lon_lo`: 0

Open problem: `lo` components always 0. Some part of the Dekker chain is still being collapsed by the GLSL compiler. The seals helped but didn't fully block it.

Things to try, rough order of suspected effectiveness:

1. **Seal the return values of helper functions.** `_twoSum` and friends seal internals but return an unsealed `vec2`. The caller may participate in a simplification that sees through the function boundary. Try:
   ```glsl
   vec2 _twoSum(float a, float b) {
       float s   = _seal(a + b);
       float bb  = _seal(s − a);
       float err = (a − (s − bb)) + (b − bb);
       return vec2(_seal(s), _seal(err));
   }
   ```
   Apply the same pattern to `_split`, `_qts`, `_twoProd`, `_add64`, `_sub64`, `_mul64`, `_div64`.

2. **Bit-level Veltkamp split.** Current `_split` uses `SPLIT = 4097.0` and relies on `t − (t − a) ≡ a` being blocked by seal. A stronger version computes the split directly from mantissa bits:
   ```glsl
   vec2 _split(float a) {
       uint bits    = floatBitsToUint(a);
       uint hi_bits = bits & 0xFFFFF000u;   // keep top 11 explicit mantissa bits (12 bits of precision with the implicit leading 1)
       float hi     = uintBitsToFloat(hi_bits);
       float lo     = a − hi;
       return vec2(hi, lo);
   }
   ```
   Exact by construction. No algebraic identity can collapse it because the compiler can't reason about mantissa bit ops as float arithmetic.

3. **Uniform-fed zero barrier.** Upload a uniform `u_zero = 0.0` from JS. In the shader: `float zero = u_zero; x = x + zero;`. The compiler can't prove `u_zero == 0.0` at compile time, so it won't fold `x + zero` away. Less surgical than `_seal`, but more reliable — the opacity is rooted in the uniform boundary, which the compiler always respects.

4. **Inspect the compiled shader.** WebGL doesn't expose the driver's compiled form directly. On Chrome, `--enable-webgl-draft-extensions` plus the graphics inspector can help. Better: port the shader to a native OpenGL project and use `KHR_parallel_shader_compile` with driver disassembly (e.g. `GL_NV_shader_compiler_gcn_dump_spirv` on Mesa). If the `lo` component is being dead-code-eliminated, you'll see it in the disasm and know exactly which pass kills it.

5. **Accept the fp32 floor.** If seals, bit tricks, and uniform barriers all fail, the precision ceiling is roughly what we have now. Going further means emulating fp64 mul/add in integer arithmetic (uint32 mantissa with separate exponent tracking), which is a 5–10× perf hit on a vertex shader that already runs per-cell-corner, per-frame. Probably not worth it for a visualization use case. Worst-case ~12 px on a single corner at zoom 18 lat 85° is already subpixel at zoom 14 and below, and only visible at max zoom if you know which corner to look at.

Expected ceiling if full Dekker survives the compiler: at ~48 bits of effective precision, `ulp(lat) = 2⁻⁴⁸ ≈ 3.55 × 10⁻¹⁵ rad` (lat ∈ [1, 2)), which at 1.22 × 10⁸ px/rad at zoom 18 lat 85° comes out to about `4 × 10⁻⁷ px` — effectively exact at any zoom a user will actually look at.

### Test methodology recap

- `node tmp/compute-truth.mjs` regenerates TRUTH from healpix-ts at f64.
- Open the HTML harness in Chrome. It compiles the vertex shader with transform feedback, reads back the four corners of the test cell, prints a table of `{gpu, truth, diff_ulp}` for each `{corner, axis, hi/lo}` slot.
- Target: `diff_ulp = 0` on `hi`, `diff_ulp` nonzero on `lo` (meaning the lo component is carrying real signal).
- Currently: `diff_ulp` is small on `hi` (good), `lo` is always 0 (bad — the compiler is still simplifying somewhere).

### Where to start

Apply fix #1 (seal function outputs) and re-run the polar harness. If any `lat_lo` or `lon_lo` becomes nonzero, the barrier worked and you can continue with equatorial regression + visual verification. If every `lo` stays 0, move to fix #2 (bit-level split). If fix #2 still produces `lo = 0`, fix #3 (uniform zero barrier) is the nuclear option.

If all three fail to produce a nonzero `lo`, the compiler is eliminating the lo path at a level unreachable from shader source, and fix #5 (accept the floor) is the answer.
