import type { Texture } from '@luma.gl/core';
import type { ShaderModule } from '@luma.gl/shadertools';

/**
 * Props consumed by the HEALPix color shader module.
 *
 * Scalar uniforms go into the `healpixColorUniforms` block.
 * Texture bindings (`healpixValuesTexture`, `healpixColorMapTexture`) are
 * declared separately in the `vs:#decl` injection in the extension.
 */
export type HealpixColorProps = {
  uMin: number;
  uMax: number;
  uDimensions: number;
  uValuesWidth: number;
  healpixValuesTexture: Texture;
  healpixColorMapTexture: Texture;
};

/**
 * Shader module for GPU color computation.
 *
 * Declares the `healpixColor` uniform block (scalar uniforms).
 * Textures are bound alongside these props via `model.shaderInputs.setProps`.
 */
export const healpixColorShaderModule = {
  name: 'healpixColor',
  vs: `\
uniform healpixColorUniforms {
  float uMin;
  float uMax;
  int uDimensions;
  int uValuesWidth;
} healpixColor;
`,
  uniformTypes: {
    uMin: 'f32',
    uMax: 'f32',
    uDimensions: 'i32',
    uValuesWidth: 'i32'
  }
} as const satisfies ShaderModule<HealpixColorProps>;
