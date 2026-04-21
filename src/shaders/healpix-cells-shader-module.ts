import type { ShaderModule } from '@luma.gl/shadertools';

export type HealpixSchemeCode = 0 | 1;

export type HealpixCellsProps = {
  nside: number;
  log2nside: number;
  scheme: HealpixSchemeCode;
  polarLim: [number, number];
  eqLim: [number, number];
  npix: [number, number];
};

/** Split a non-negative JS number (≤ 2^53 - 1) into [lo, hi] u32 halves. */
export function splitU53(x: number): [number, number] {
  return [x >>> 0, Math.floor(x / 4294967296)];
}

/**
 * Per-draw uniforms for GPU NEST/RING decode from `nside` + `scheme`.
 *
 * `polarLim`, `eqLim`, `npix`, and `log2nside` are derived only from `nside` and
 * are identical for every instance in a draw. Computing them here (once per
 * draw) avoids repeating the same multiplies and `uvec2` splits in every
 * vertex, and keeps the limit values aligned with the JS reference (`splitU53`
 * matches how tests and tooling build comparable u64 wire values).
 */
export function computeHealpixCellsUniforms(
  nside: number,
  scheme: 'nest' | 'ring'
): HealpixCellsProps {
  const polarLimN = 2 * nside * (nside - 1);
  const npixN = 12 * nside * nside;
  const eqLimN = polarLimN + 8 * nside * nside;
  return {
    nside,
    log2nside: Math.round(Math.log2(nside)),
    scheme: scheme === 'nest' ? 0 : 1,
    polarLim: splitU53(polarLimN),
    eqLim: splitU53(eqLimN),
    npix: splitU53(npixN)
  };
}

export const healpixCellsShaderModule = {
  name: 'healpixCells',
  vs: `\
uniform healpixCellsUniforms {
  uint nside;
  uint log2nside;
  uint scheme;
  uvec2 polarLim;
  uvec2 eqLim;
  uvec2 npix;
} healpixCells;
`,
  uniformTypes: {
    nside: 'u32',
    log2nside: 'u32',
    scheme: 'u32',
    polarLim: 'vec2<u32>',
    eqLim: 'vec2<u32>',
    npix: 'vec2<u32>'
  }
} as const satisfies ShaderModule<HealpixCellsProps>;
