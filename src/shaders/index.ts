/**
 * Assembles HEALPix cells vertex and fragment shaders from focused modules.
 */
import { INT64_GLSL } from './int64.glsl';
import { FP64_GLSL } from './fp64.glsl';
import { HEALPIX_DECOMPOSE_GLSL } from './healpix-decompose.glsl';
import { HEALPIX_CORNERS_GLSL } from './healpix-corners.glsl';
import { HEALPIX_CELLS_VS_MAIN } from './healpix-cells.vs.glsl';
import { HEALPIX_CELLS_FS } from './healpix-cells.fs.glsl';

const VS_HEADER = /* glsl */ `\
#version 300 es
#define SHADER_NAME healpix-cells-vertex
precision highp float;
precision highp int;
`;

export const HEALPIX_VERTEX_SHADER: string = [
  VS_HEADER,
  INT64_GLSL,
  FP64_GLSL,
  HEALPIX_DECOMPOSE_GLSL,
  HEALPIX_CORNERS_GLSL,
  HEALPIX_CELLS_VS_MAIN
].join('\n');

export const HEALPIX_FRAGMENT_SHADER: string = HEALPIX_CELLS_FS;
