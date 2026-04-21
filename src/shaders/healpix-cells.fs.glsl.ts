/** Fragment shader — color work happens in DECKGL_FILTER_COLOR. */
export const HEALPIX_CELLS_FS: string = /* glsl */ `\
#version 300 es
precision highp float;

in  vec4 vColor;
out vec4 fragColor;

void main() {
  fragColor = vColor;
  DECKGL_FILTER_COLOR(fragColor, geometry);
}
`;
