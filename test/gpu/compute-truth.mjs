// Computes the reference values the GPU readback is compared against.
// Run: node test/gpu/compute-truth.mjs
//
// IMPORTANT: a corner of pixel (x, y) is NOT `fxy2tu(nside, f, x+dx, y+dy)` for
// integer dx/dy — that gives you the *center* of a neighboring pixel, which
// sits half a pixel away from the actual corner. The shader's integer-lattice
// corners, combined with the pixel-center formula (no -1), arrange so that
// the four vertices land on the N/W/S/E cardinal corners of pixel (x, y).
// That matches what healpix-ts fxyCorners returns, which is what the reference
// PathLayer renders via cornersNestLonLat.
import { nest2fxy, fxy2tu, tu2za, fxyCorners, PI_4 } from 'healpix-ts';

const nside = 262144;
const cellId = 228532280344;

const { f, x, y } = nest2fxy(nside, cellId);
console.log('nest2fxy:', { f, x, y });

const cellIdLo = cellId >>> 0;
const cellIdHi = Math.floor(cellId / 4294967296);
console.log('cellIdLo / cellIdHi (GPU attributes):', cellIdLo, cellIdHi);
const faceIx = ((f << 24) >>> 0) | (x & 0xffffff);
const instIy = y >>> 0;
console.log('legacy faceIx (u32 hex):', faceIx.toString(16).padStart(8, '0'));
console.log('legacy instIy (u32 hex):', instIy.toString(16).padStart(8, '0'));

// --- Reference via fxyCorners (3D vector → lon/lat) ---------------------------
// fxyCorners returns 4 unit vectors in order [N, W, S, E].
// lon = atan2(Y, X), lat = asin(Z).
function vecToLonLat([X, Y, Z]) {
  const lonRad = Math.atan2(Y, X);
  const latRad = Math.asin(Math.max(-1, Math.min(1, Z)));
  let lonDeg = (lonRad * 180) / Math.PI;
  lonDeg -= 360 * Math.floor((lonDeg + 180) / 360);
  return { lonDeg, latDeg: (latRad * 180) / Math.PI };
}

console.log('\nfxyCorners → lon/lat (N, W, S, E):');
const corners = fxyCorners(nside, f, x, y);
for (let i = 0; i < corners.length; i++) {
  const { lonDeg, latDeg } = vecToLonLat(corners[i]);
  const labels = ['N', 'W', 'S', 'E'];
  console.log(
    `  ${labels[i]}  lon=${lonDeg.toFixed(14)}°  lat=${latDeg.toFixed(14)}°`
  );
  console.log(
    `       fp32 lon=${Math.fround(lonDeg).toFixed(14)}  ` +
      `fp32 lat=${Math.fround(latDeg).toFixed(14)}`
  );
  console.log(
    `       lon_lo=${(lonDeg - Math.fround(lonDeg)).toExponential(3)}  ` +
      `lat_lo=${(latDeg - Math.fround(latDeg)).toExponential(3)}`
  );
}

// --- Shader ci mapping --------------------------------------------------------
// In the vertex shader, integer (cx, cy) = (x + {0,1}, y + {0,1}) for
// ci = 0..3. With i_ring = f1*nside - cx - cy (no -1), the four vertices land
// at (Δt, Δu):
//   ci=0 (cx=x+1, cy=y+1): (0, +d)   → N corner (fxyCorners[0])
//   ci=1 (cx=x,   cy=y+1): (-d, 0)   → W corner (fxyCorners[1])
//   ci=2 (cx=x,   cy=y):   (0, -d)   → S corner (fxyCorners[2])
//   ci=3 (cx=x+1, cy=y):   (+d, 0)   → E corner (fxyCorners[3])
// So shader ci matches fxyCorners index directly.

console.log('\nShader ci → cardinal corner mapping:');
console.log('  ci=0 → N   ci=1 → W   ci=2 → S   ci=3 → E\n');

// --- Corner truth via center + cardinal ±d offset -----------------------------
// This is what the shader *should* be producing if its Dekker pipeline is
// bit-exact relative to fp64 math.
const { t: tC, u: uC } = fxy2tu(nside, f, x, y);
const d = PI_4 / nside;
const offsets = [
  [0, d], // N (ci=0)
  [-d, 0], // W (ci=1)
  [0, -d], // S (ci=2)
  [d, 0] // E (ci=3)
];

console.log('Corner truth via (t_center + Δt, u_center + Δu) → tu2za → asin:');
for (let ci = 0; ci < 4; ci++) {
  const [dt, du] = offsets[ci];
  const { z, a } = tu2za(tC + dt, uC + du);
  const latRad = Math.asin(Math.max(-1, Math.min(1, z)));
  const lonRad = a;
  let lonDeg = (lonRad * 180) / Math.PI;
  lonDeg -= 360 * Math.floor((lonDeg + 180) / 360);
  const latDeg = (latRad * 180) / Math.PI;
  console.log(
    `  ci=${ci}  lon=${lonDeg.toFixed(14)}  lat=${latDeg.toFixed(14)}`
  );
  console.log(
    `         fp32 lon=${Math.fround(lonDeg).toFixed(14)}  ` +
      `fp32 lat=${Math.fround(latDeg).toFixed(14)}`
  );
  console.log(
    `         lon_lo=${(lonDeg - Math.fround(lonDeg)).toExponential(3)}  ` +
      `lat_lo=${(latDeg - Math.fround(latDeg)).toExponential(3)}`
  );
}
