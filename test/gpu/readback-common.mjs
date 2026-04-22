/** Shared WebGL2 transform-feedback helpers for GPU readback pages. */

export const READBACK_FS = `#version 300 es
precision mediump float;
out vec4 fragColor;
void main() { fragColor = vec4(1.0); }
`;

export function buildReadbackVS(
  INT64_GLSL,
  FP64_GLSL,
  HEALPIX_DECOMPOSE_GLSL,
  HEALPIX_CORNERS_GLSL
) {
  return `#version 300 es
precision highp float;
precision highp int;

${INT64_GLSL}
${FP64_GLSL}
${HEALPIX_DECOMPOSE_GLSL}
${HEALPIX_CORNERS_GLSL}

uniform int u_nside;
uniform uint u_log2nside;
uniform uint u_scheme;
uniform uvec2 u_polarLim;
uniform uvec2 u_eqLim;
uniform uvec2 u_npix;

in uint cellIdLo;
in uint cellIdHi;

flat out uvec3 vDecode;
out vec4 vCorner;

void main() {
  uvec2 cellId = uvec2(cellIdLo, cellIdHi);
  uvec3 fxy;
  if (u_scheme == 0u) {
    fxy = decodeNest(cellId, u_log2nside);
  } else {
    fxy = decodeRing(cellId, uint(u_nside), u_polarLim, u_eqLim, u_npix);
  }
  int face = int(fxy.x);
  int ix = int(fxy.y);
  int iy = int(fxy.z);

  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  vec2 lon_rad_fp, lat_rad_fp;
  fxyCorner(face, cx, cy, u_nside, lon_rad_fp, lat_rad_fp);

  vec2 deg_per_rad = _div64(vec2(180.0, 0.0), PI64);
  vec2 lat_deg_fp  = _mul64(lat_rad_fp, deg_per_rad);
  vec2 lon_deg_fp  = _mul64(lon_rad_fp, deg_per_rad);

  vDecode = fxy;
  vCorner = vec4(lon_deg_fp.x, lon_deg_fp.y, lat_deg_fp.x, lat_deg_fp.y);
  gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}
`;
}

export function splitU53(x) {
  return [x >>> 0, Math.floor(x / 4294967296)];
}

export function setReadbackUniforms(gl, prog, nside, scheme) {
  // Uniform setters only affect the *currently used* program (WebGL spec).
  gl.useProgram(prog);
  const polarLim = 2 * nside * (nside - 1);
  const eqLim = polarLim + 8 * nside * nside;
  const npix = 12 * nside * nside;
  const log2n = Math.round(Math.log2(nside));

  gl.uniform1i(gl.getUniformLocation(prog, 'u_nside'), nside);
  gl.uniform1ui(gl.getUniformLocation(prog, 'u_log2nside'), log2n >>> 0);
  gl.uniform1ui(gl.getUniformLocation(prog, 'u_scheme'), scheme >>> 0);

  const locPl = gl.getUniformLocation(prog, 'u_polarLim');
  const locEq = gl.getUniformLocation(prog, 'u_eqLim');
  const locNp = gl.getUniformLocation(prog, 'u_npix');
  const [pl0, pl1] = splitU53(polarLim);
  const [eq0, eq1] = splitU53(eqLim);
  const [np0, np1] = splitU53(npix);
  gl.uniform2ui(locPl, pl0, pl1);
  gl.uniform2ui(locEq, eq0, eq1);
  gl.uniform2ui(locNp, np0, np1);
}

export function compileShader(gl, type, src) {
  const sh = gl.createShader(type);
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    throw new Error(`shader compile failed:\n${log}\n\n${src}`);
  }
  return sh;
}

export function linkTfProgram(gl, vsSrc, fsSrc) {
  const prog = gl.createProgram();
  gl.attachShader(prog, compileShader(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compileShader(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.transformFeedbackVaryings(
    prog,
    ['vDecode', 'vCorner'],
    gl.SEPARATE_ATTRIBS
  );
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error(`link failed: ${gl.getProgramInfoLog(prog)}`);
  }
  return prog;
}

/**
 * One instanced TF draw: 4 points × 1 instance. Returns decode (12 u32) + corner (16 f32).
 */
export function runReadback(gl, prog, cellIdLo, cellIdHi) {
  gl.useProgram(prog);
  const locLo = gl.getAttribLocation(prog, 'cellIdLo');
  const locHi = gl.getAttribLocation(prog, 'cellIdHi');

  const loBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, loBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array([cellIdLo]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(locLo);
  gl.vertexAttribIPointer(locLo, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(locLo, 1);

  const hiBuf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, hiBuf);
  gl.bufferData(gl.ARRAY_BUFFER, new Uint32Array([cellIdHi]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(locHi);
  gl.vertexAttribIPointer(locHi, 1, gl.UNSIGNED_INT, 0, 0);
  gl.vertexAttribDivisor(locHi, 1);

  const tfDecode = gl.createBuffer();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, tfDecode);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, 4 * 3 * 4, gl.STREAM_READ);
  const tfCorner = gl.createBuffer();
  gl.bindBuffer(gl.TRANSFORM_FEEDBACK_BUFFER, tfCorner);
  gl.bufferData(gl.TRANSFORM_FEEDBACK_BUFFER, 4 * 4 * 4, gl.STREAM_READ);

  const tf = gl.createTransformFeedback();
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, tf);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, tfDecode);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, tfCorner);

  gl.bindBuffer(gl.ARRAY_BUFFER, null);
  gl.enable(gl.RASTERIZER_DISCARD);
  gl.beginTransformFeedback(gl.POINTS);
  gl.drawArraysInstanced(gl.POINTS, 0, 4, 1);
  gl.endTransformFeedback();
  gl.disable(gl.RASTERIZER_DISCARD);

  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
  gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, null);
  gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

  const decode = new Uint32Array(12);
  gl.bindBuffer(gl.COPY_READ_BUFFER, tfDecode);
  gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, decode);
  const corner = new Float32Array(16);
  gl.bindBuffer(gl.COPY_READ_BUFFER, tfCorner);
  gl.getBufferSubData(gl.COPY_READ_BUFFER, 0, corner);
  gl.bindBuffer(gl.COPY_READ_BUFFER, null);

  return { decode, corner };
}

export function fp32(x) {
  return Math.fround(x);
}

export function ulp32(x) {
  const ax = Math.abs(Math.fround(x));
  if (ax === 0) return Math.pow(2, -149);
  const buf = new ArrayBuffer(4);
  const f32 = new Float32Array(buf);
  const u32 = new Uint32Array(buf);
  f32[0] = ax;
  u32[0] += 1;
  return f32[0] - ax;
}

export function diffClass(ulps) {
  if (ulps <= 0.51) return 'ok';
  if (ulps <= 4.01) return 'warn';
  return 'bad';
}

export function fmt(x, pad = 22) {
  if (x === 0) return '0'.padEnd(pad);
  const s = (x >= 0 ? ' ' : '') + x.toExponential(6);
  return s.padEnd(pad);
}

export function formatDecodeCheck(
  decode,
  expectedFace,
  expectedIx,
  expectedIy
) {
  let html = '\nDecode check (face, ix, iy) — must match for all 4 vertices:\n';
  let decodeOk = true;
  for (let i = 0; i < 4; i++) {
    const face = decode[i * 3 + 0];
    const ix = decode[i * 3 + 1];
    const iy = decode[i * 3 + 2];
    const ok = face === expectedFace && ix === expectedIx && iy === expectedIy;
    decodeOk = decodeOk && ok;
    html += `  v${i}  face=${face}  ix=${ix}  iy=${iy}  ${
      ok ? '<span class="ok">OK</span>' : '<span class="bad">MISMATCH</span>'
    }\n`;
  }
  return { html, decodeOk };
}

export function formatCornerDiff(corner, TRUTH) {
  const labels = ['N (ci=0)', 'W (ci=1)', 'S (ci=2)', 'E (ci=3)'];
  let html = '';
  html += 'Per-corner GPU readback vs CPU truth\n';
  html += '─'.repeat(112) + '\n';

  for (let i = 0; i < 4; i++) {
    const [lonHi, lonLo, latHi, latLo] = corner.slice(i * 4, i * 4 + 4);
    const t = TRUTH[i];
    const expLonHi = fp32(t.lon);
    const expLonLo = t.lon - expLonHi;
    const expLatHi = fp32(t.lat);
    const expLatLo = t.lat - expLatHi;

    const lonHiErr = lonHi - expLonHi;
    const lonLoErr = lonLo - expLonLo;
    const latHiErr = latHi - expLatHi;
    const latLoErr = latLo - expLatLo;

    const lonUlp = ulp32(expLonHi);
    const latUlp = ulp32(expLatHi);

    html += `\n<b>${labels[i]}</b>\n`;
    html += `  lon truth=${t.lon.toFixed(14)}°   fp32=${expLonHi.toFixed(14)}   lo=${expLonLo.toExponential(3)}\n`;
    html += `  lat truth=${t.lat.toFixed(14)}°   fp32=${expLatHi.toFixed(14)}   lo=${expLatLo.toExponential(3)}\n`;
    html += `  fp32 ulp:  lon=${lonUlp.toExponential(3)}   lat=${latUlp.toExponential(3)}\n`;
    html += '\n';
    html +=
      '  field      GPU                     expected                diff                    ulps\n';
    html += `  lon_hi   <span class="${diffClass(Math.abs(lonHiErr) / lonUlp)}">${fmt(lonHi)}  ${fmt(expLonHi)}  ${fmt(lonHiErr)}  ${(Math.abs(lonHiErr) / lonUlp).toFixed(3)}</span>\n`;
    html += `  lon_lo   <span class="${diffClass(Math.abs(lonLoErr) / lonUlp)}">${fmt(lonLo)}  ${fmt(expLonLo)}  ${fmt(lonLoErr)}  ${(Math.abs(lonLoErr) / lonUlp).toFixed(3)}</span>\n`;
    html += `  lat_hi   <span class="${diffClass(Math.abs(latHiErr) / latUlp)}">${fmt(latHi)}  ${fmt(expLatHi)}  ${fmt(latHiErr)}  ${(Math.abs(latHiErr) / latUlp).toFixed(3)}</span>\n`;
    html += `  lat_lo   <span class="${diffClass(Math.abs(latLoErr) / latUlp)}">${fmt(latLo)}  ${fmt(expLatLo)}  ${fmt(latLoErr)}  ${(Math.abs(latLoErr) / latUlp).toFixed(3)}</span>\n`;
  }

  let worst = { field: '', ulps: 0 };
  for (let i = 0; i < 4; i++) {
    const [lonHi, lonLo, latHi, latLo] = corner.slice(i * 4, i * 4 + 4);
    const t = TRUTH[i];
    const tests = [
      { k: `ci${i}.lon_hi`, g: lonHi, e: fp32(t.lon), u: ulp32(fp32(t.lon)) },
      {
        k: `ci${i}.lon_lo`,
        g: lonLo,
        e: t.lon - fp32(t.lon),
        u: ulp32(fp32(t.lon))
      },
      { k: `ci${i}.lat_hi`, g: latHi, e: fp32(t.lat), u: ulp32(fp32(t.lat)) },
      {
        k: `ci${i}.lat_lo`,
        g: latLo,
        e: t.lat - fp32(t.lat),
        u: ulp32(fp32(t.lat))
      }
    ];
    for (const { k, g, e, u } of tests) {
      const ulps = Math.abs(g - e) / u;
      if (ulps > worst.ulps) worst = { field: k, ulps };
    }
  }

  html += '\n' + '─'.repeat(112) + '\n';
  html += `<b>Worst disagreement:</b> ${worst.field}  =  ${worst.ulps.toFixed(3)} fp32 ULPs\n`;
  html += '\nLegend:  ';
  html += '<span class="ok">≤0.5 ULP (bit-exact-ish)</span>   ';
  html += '<span class="warn">≤4 ULP (GLSL spec-bound)</span>   ';
  html += '<span class="bad">&gt;4 ULP (pipeline bug)</span>\n';
  return html;
}
