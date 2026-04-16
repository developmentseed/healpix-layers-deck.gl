import type { ShaderModule } from '@luma.gl/shadertools';

export type HealpixCellsProps = {
  nside: number;
};

export const healpixCellsShaderModule = {
  name: 'healpixCells',
  vs: `\
uniform healpixCellsUniforms {
  uint nside;
} healpixCells;
`,
  uniformTypes: {
    nside: 'u32'
  }
} as const satisfies ShaderModule<HealpixCellsProps>;
