export const HEALPIX_VERTEX_SHADER: string = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;

in uint faceIx;
in uint instIy;
in vec3 positions;

out vec4 vColor;

const int jrll[12] = int[12](2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4);
const int jpll[12] = int[12](1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7);

void main() {
  int face = int(faceIx >> 24u);
  int ix   = int(faceIx & 0xFFFFFFu);
  int iy   = int(instIy);

  // Corner selection: index buffer [0,1,2,0,2,3] → gl_VertexID cycles 0,1,2,0,2,3
  //   0 = North (ix+1, iy+1)
  //   1 = West  (ix,   iy+1)
  //   2 = South (ix,   iy)
  //   3 = East  (ix+1, iy)
  int ci = gl_VertexID % 4;
  int cx = ix + ((ci == 0 || ci == 3) ? 1 : 0);
  int cy = iy + ((ci == 0 || ci == 1) ? 1 : 0);

  float nside_f = float(healpixCells.nside);
  float x_n = float(cx) / nside_f;
  float y_n = float(cy) / nside_f;

  // xyf2loc: (face, x_norm, y_norm) → (z, phi)
  float jr = float(jrll[face]) - x_n - y_n;
  float nr;
  float z;

  if (jr < 1.0) {
    nr = jr;
    float tmp = nr * nr / 3.0;
    z = 1.0 - tmp;
  } else if (jr > 3.0) {
    nr = 4.0 - jr;
    float tmp = nr * nr / 3.0;
    z = tmp - 1.0;
  } else {
    nr = 1.0;
    z = (2.0 - jr) * 2.0 / 3.0;
  }

  float tmp_phi = float(jpll[face]) * nr + x_n - y_n;
  if (tmp_phi < 0.0) tmp_phi += 8.0;
  if (tmp_phi >= 8.0) tmp_phi -= 8.0;
  float phi = (nr < 1e-15) ? 0.0 : (PI * 0.25 * tmp_phi) / nr;

  // (z, phi) → (lon°, lat°)
  float lat_deg = asin(clamp(z, -1.0, 1.0)) * (180.0 / PI);
  float lon_deg = phi * (180.0 / PI);
  lon_deg -= 360.0 * floor((lon_deg + 180.0) / 360.0);

  vec4 pos = vec4(lon_deg, lat_deg, 0.0, 1.0);
  geometry.position = pos;
  gl_Position = project_common_position_to_clipspace(project_position(pos));

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
