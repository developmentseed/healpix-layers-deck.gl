# HEALPix GPU pipeline tests

Manual checks that **Jest cannot run**: WebGL2 transform feedback reads back `vDecode` / `vCorner` from the same GLSL chain as production (`int64` + `fp64` + `healpix-decompose` + `fxyCorner`) and diffs against tabulated CPU truth.

---

## Before you open the browser

1. **Use a WebGL2-capable browser** (current Chrome, Firefox, or Safari).

2. **Work from the package root** — the directory that contains `package.json` and the `test/` folder.  

   The static server’s “`.`” must be **this** directory so that  
   `http://localhost:<port>/test/gpu/…` maps to `test/gpu/…` on disk.

3. **If you changed any `src/shaders/*.glsl.ts` file**, regenerate the strings the pages import:

   ```bash
   npm run gen:gpu-shaders
   ```

   That overwrites `test/gpu/shader-chunks.gen.mjs`. If you skip this after a shader edit, the readback is testing **stale** GLSL.

---

## Run the readback pages (required: HTTP, not `file://`)

ES modules (`import` in `<script type="module">`) are blocked or behave badly on `file://`. You must serve the **package root** over HTTP.

From the package root:

```bash
npx http-server . -p 8080 -c-1
```

`-c-1` disables caching so you are not looking at an old `shader-chunks.gen.mjs` after regenerating.

Open these URLs (host/port must match your server):

- http://localhost:8080/test/gpu/gpu-readback-nest-equatorial.html  
- http://localhost:8080/test/gpu/gpu-readback-nest-polar.html  
- http://localhost:8080/test/gpu/gpu-readback-ring.html  

**Sanity check:** open DevTools → **Console**. There should be **no** red errors. A failed shader compile throws and may leave the page on “running…”.

---

## How to know it passed

Each page prints:

1. **Decode check** — for all four vertices, `face` / `ix` / `iy` must match the expected triple; every line should show **OK** (green), not **MISMATCH** (red).

2. **Per-corner ULP table** — fields labeled **≤0.5 ULP** should use the green **ok** styling. Yellow (**warn**) is a looser bound; red (**bad**) indicates a real problem.

3. **Worst disagreement** — at the bottom, the worst case should stay **≤ 0.5** fp32 ULPs if everything matches the reference pipeline.

The **RING** page runs three cases in one load; all three should satisfy the same rules.

---

## Node helpers (optional)

Still from the **package root** (so `node_modules` resolves for `healpix-ts`):

```bash
node test/gpu/compute-truth.mjs
node test/gpu/inspect-cell.mjs <cellId>
```

They print NEST decomposition, `(cellIdLo, cellIdHi)`, and corner reference values for comparison with the HTML truth tables.

---

## Jest (CPU only)

```bash
npm test
```

Covers the JS reference decoders and `splitCellIds`; it does **not** execute the GPU pages.
