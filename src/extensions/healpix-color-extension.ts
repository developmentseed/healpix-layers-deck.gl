import { Layer, LayerExtension, LayerProps } from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { healpixColorShaderModule } from './healpix-color-shader-module';

/** Extra props this extension reads from the host primitive layer. */
export type HealpixColorExtensionProps = LayerProps & {
  valuesTexture: Texture;
  colorMapTexture: Texture;
  uMin: number;
  uMax: number;
  uDimensions: number;
  uValuesWidth: number;
};

/**
 * GLSL declaration injected into the vertex shader.
 * Declares the two texture samplers used for color computation.
 */
const VERTEX_DECLARATION_INJECT = `
uniform highp sampler2D healpixValuesTexture;
uniform mediump sampler2D healpixColorMapTexture;
`;

/**
 * GLSL injected into deck.gl's DECKGL_FILTER_COLOR hook.
 *
 * Samples per-cell float values from the values texture, then computes
 * the output color based on the dimensions mode:
 *
 *   dimensions=1  scalar → normalized through [min,max] → colorMap LUT
 *   dimensions=2  scalar (→ colorMap) + opacity multiplier (second value)
 *   dimensions=3  direct RGB in 0–1; alpha=1
 *   dimensions=4  direct RGBA in 0–1
 *   else          transparent (reserved for future band math)
 */
const VERTEX_COLOR_FILTER_INJECT = `
int healpixCell = gl_InstanceID;
int healpixX = healpixCell % healpixColor.uValuesWidth;
int healpixY = healpixCell / healpixColor.uValuesWidth;
vec4 healpixVals = texelFetch(healpixValuesTexture, ivec2(healpixX, healpixY), 0);

float healpixDenom = healpixColor.uMax - healpixColor.uMin;
float healpixT = healpixDenom == 0.0
  ? 0.0
  : clamp((healpixVals.r - healpixColor.uMin) / healpixDenom, 0.0, 1.0);

vec4 healpixOut;
if (healpixColor.uDimensions == 1) {
  healpixOut = texelFetch(healpixColorMapTexture, ivec2(int(healpixT * 255.0), 0), 0);
} else if (healpixColor.uDimensions == 2) {
  healpixOut = texelFetch(healpixColorMapTexture, ivec2(int(healpixT * 255.0), 0), 0);
  healpixOut.a *= healpixVals.g;
} else if (healpixColor.uDimensions == 3) {
  healpixOut = vec4(healpixVals.rgb, 1.0);
} else if (healpixColor.uDimensions == 4) {
  healpixOut = healpixVals;
} else {
  healpixOut = vec4(0.0);
}
color = vec4(healpixOut.rgb, healpixOut.a * layer.opacity);
`;

/**
 * Layer extension that computes HEALPix cell colors on the GPU.
 *
 * Reads per-cell float values from an RGBA32F texture and converts them
 * to RGBA using a 256×1 colorMap LUT texture, driven by the `dimensions` mode.
 * Replaces `HealpixColorFramesExtension`.
 */
class HealpixColorExtension extends LayerExtension {
  static extensionName = 'HealpixColorExtension';

  getShaders(): unknown {
    return {
      modules: [healpixColorShaderModule],
      inject: {
        'vs:#decl': VERTEX_DECLARATION_INJECT,
        'vs:DECKGL_FILTER_COLOR': VERTEX_COLOR_FILTER_INJECT
      }
    };
  }

  draw(
    this: Layer<HealpixColorExtensionProps>,
    _opts: { uniforms: unknown }
  ): void {
    const {
      valuesTexture,
      colorMapTexture,
      uMin,
      uMax,
      uDimensions,
      uValuesWidth
    } = this.props as HealpixColorExtensionProps;

    for (const model of this.getModels()) {
      model.shaderInputs.setProps({
        healpixColor: {
          uMin,
          uMax,
          uDimensions,
          uValuesWidth,
          healpixValuesTexture: valuesTexture,
          healpixColorMapTexture: colorMapTexture
        }
      });
    }
  }
}

/** Shared singleton used by all HEALPix sublayers. */
export const HEALPIX_COLOR_EXTENSION = new HealpixColorExtension();
