// Decompose a HEALPix cell and classify it (equatorial vs polar, which face),
// then print the 4 cardinal corner truth values the shader should produce.
import { nest2fxy, fxy2tu, tu2za, fxyCorners, PI_4, PI_2 } from 'healpix-ts';

const nside = 262144;
const cellId = Number(process.argv[2] ?? 274626645138);

const { f, x, y } = nest2fxy(nside, cellId);
console.log(`cell=${cellId}  nside=${nside}`);
console.log(`  face=${f}  x=${x}  y=${y}`);
console.log(
  `  f_row=${Math.floor(f / 4)} (0=north cap, 1=equator, 2=south cap)`
);

// center
const { t: tC, u: uC } = fxy2tu(nside, f, x, y);
const { z: zC, a: aC } = tu2za(tC, uC);
const latC = (Math.asin(zC) * 180) / Math.PI;
let lonC = (aC * 180) / Math.PI;
lonC -= 360 * Math.floor((lonC + 180) / 360);
console.log(`  center t=${tC.toFixed(6)}  u=${uC.toFixed(6)}`);
console.log(`  center lon=${lonC.toFixed(6)}°  lat=${latC.toFixed(6)}°`);

// Equatorial vs polar branch is chosen by |u|:
// - |u| <= PI/4  → equatorial (|lat| <= arcsin(2/3) ≈ 41.8°)
// - |u|  > PI/4  → polar cap
console.log(`  |u|=${Math.abs(uC).toFixed(6)}  PI_4=${PI_4.toFixed(6)}`);
console.log(`  → branch: ${Math.abs(uC) > PI_4 ? 'POLAR' : 'EQUATORIAL'}`);

// Also show the integer k_ring/i_ring values this cell uses.
const f_row = Math.floor(f / 4);
const f1 = f_row + 2;
const f2 = 2 * (f % 4) - (f_row % 2) + 1;
console.log(`  f1=${f1}  f2=${f2}`);
console.log(`  i_ring = f1*nside - cx - cy  (no -1)`);
console.log(`  k_ring = f2*nside + (cx-cy) + 8*nside`);

console.log('\n4-corner truth (N, W, S, E) via fxyCorners:');
const corners = fxyCorners(nside, f, x, y);
const lbls = ['N', 'W', 'S', 'E'];
for (let i = 0; i < corners.length; i++) {
  const [X, Y, Z] = corners[i];
  const lat = (Math.asin(Math.max(-1, Math.min(1, Z))) * 180) / Math.PI;
  let lon = (Math.atan2(Y, X) * 180) / Math.PI;
  lon -= 360 * Math.floor((lon + 180) / 360);
  console.log(`  ${lbls[i]}  lon=${lon.toFixed(10)}  lat=${lat.toFixed(10)}`);
}

// Show the shader's integer corners and the (t, u) each lands at.
console.log('\nShader integer corners and their (t, u, branch):');
for (let ci = 0; ci < 4; ci++) {
  const cx = x + (ci === 0 || ci === 3 ? 1 : 0);
  const cy = y + (ci === 0 || ci === 1 ? 1 : 0);
  // use the (no -1) formula to match shader
  const i_ring = f1 * nside - cx - cy;
  const k_ring = f2 * nside + (cx - cy) + 8 * nside;
  const t_unwrapped = (k_ring / nside) * PI_4;
  const u = PI_2 - (i_ring / nside) * PI_4;
  const branch =
    Math.abs(u) > PI_4 ? 'POLAR' : Math.abs(u) >= PI_2 ? 'POLE' : 'EQUATORIAL';
  console.log(
    `  ci=${ci}  cx=${cx} cy=${cy}  i_ring=${i_ring} k_ring=${k_ring}`
  );
  console.log(
    `         t=${t_unwrapped.toFixed(6)} u=${u.toFixed(6)}  branch=${branch}`
  );
}
