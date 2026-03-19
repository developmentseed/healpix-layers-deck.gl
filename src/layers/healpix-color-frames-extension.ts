import { Layer, LayerExtension, LayerProps } from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { healpixColorFramesShaderModule } from './healpix-color-frames-shader-module';

/**
 * Extra props this extension expects to exist on the target primitive layer.
 */
export type HealpixColorFramesExtensionProps = LayerProps & {
  frameTexture: Texture;
  frameIndex: number;
};

/**
 * GLSL declaration injected into the vertex shader.
 *
 * - Declares the texture binding and the per-vertex cell index attribute.
 */
const VERTEX_DECLARATION_INJECT = `
uniform sampler2D healpixFramesTexture;
in float healpixCellIndex;
`;

/**
 * GLSL snippet injected into deck.gl's `DECKGL_FILTER_COLOR` hook.
 *
 * This replaces the incoming `color` with the value sampled from the
 * `(cellIndex, frameIndex)` texel.
 */
const VERTEX_COLOR_FILTER_INJECT = `
int healpixCell = int(healpixCellIndex + 0.5);
vec4 healpixFrameColor = texelFetch(
  healpixFramesTexture,
  ivec2(healpixCell, healpixColorFrames.frameIndex),
  0
);
color = vec4(healpixFrameColor.rgb, healpixFrameColor.a * layer.opacity);
`;

/**
 * Layer extension that enables texture-driven color animation for HEALPix cells.
 *
 * It:
 * 1. Adds a custom per-vertex `healpixCellIndex` attribute.
 * 2. Injects shader code to fetch color from a `(cell, frame)` texture.
 * 3. Binds frame texture + frame index each draw.
 */
class HealpixColorFramesExtension extends LayerExtension {
  static extensionName = 'HealpixColorFramesExtension';

  /**
   * Register the `healpixCellIndex` attribute expected by shader injections.
   */
  initializeState(this: Layer): void {
    this.getAttributeManager()?.add({
      healpixCellIndex: {
        size: 1,
        type: 'float32',
        stepMode: 'dynamic',
        accessor: 'healpixCellIndex',
        defaultValue: 0,
        noAlloc: true
      }
    });
  }

  /**
   * Add shader module and shader injection hooks.
   */
  getShaders(): unknown {
    return {
      modules: [healpixColorFramesShaderModule],
      inject: {
        'vs:#decl': VERTEX_DECLARATION_INJECT,
        'vs:DECKGL_FILTER_COLOR': VERTEX_COLOR_FILTER_INJECT
      }
    };
  }

  /**
   * Push per-draw shader inputs (`frameTexture`, `frameIndex`) into all models.
   */
  draw(
    this: Layer<HealpixColorFramesExtensionProps>,
    _opts: { uniforms: unknown }
  ): void {
    const props = this.props as HealpixColorFramesExtensionProps;
    const { frameTexture, frameIndex } = props;

    for (const model of this.getModels()) {
      model.shaderInputs.setProps({
        healpixColorFrames: {
          frameIndex,
          healpixFramesTexture: frameTexture
        }
      });
    }
  }
}

/**
 * Shared singleton instance used by all HEALPix sublayers.
 */
export const HEALPIX_COLOR_FRAMES_EXTENSION = new HealpixColorFramesExtension();
