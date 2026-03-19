import type { Texture } from '@luma.gl/core';
import type { ShaderModule } from '@luma.gl/shadertools';

/**
 * Uniform/binding props consumed by the HEALPix color-frame shader module.
 *
 * - `frameIndex` selects the row in the texture (the active animation frame).
 * - `healpixFramesTexture` is a `width=cellCount`, `height=frameCount` RGBA texture.
 */
export type HealpixColorFramesProps = {
  frameIndex: number;
  healpixFramesTexture: Texture;
};

/**
 * Shader module that exposes frame selection uniform(s) to the vertex shader.
 *
 * The texture binding itself is supplied via `model.shaderInputs.setProps()` in the
 * extension draw hook.
 */
export const healpixColorFramesShaderModule = {
  name: 'healpixColorFrames',
  vs: `\
uniform healpixColorFramesUniforms {
  int frameIndex;
} healpixColorFrames;
`,
  uniformTypes: {
    frameIndex: 'i32'
  }
} as const satisfies ShaderModule<HealpixColorFramesProps>;
