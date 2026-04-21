import {
  DefaultProps,
  Layer,
  LayerContext,
  picking,
  project32,
  UpdateParameters
} from '@deck.gl/core';
import { Geometry, Model } from '@luma.gl/engine';
import type { RenderPass } from '@luma.gl/core';
import { HEALPIX_FRAGMENT_SHADER, HEALPIX_VERTEX_SHADER } from '../shaders';
import {
  computeHealpixCellsUniforms,
  healpixCellsShaderModule
} from '../shaders/healpix-cells-shader-module';

/** Props for the GPU-instanced HEALPix cell primitive layer. */
export type HealpixCellsPrimitiveLayerProps = {
  nside: number;
  scheme: 'nest' | 'ring';
  instanceCount: number;
};

type _HealpixCellsPrimitiveLayerProps = HealpixCellsPrimitiveLayerProps;

type HealpixCellsPrimitiveLayerMergedProps = _HealpixCellsPrimitiveLayerProps &
  import('@deck.gl/core').LayerProps;

const defaultProps: DefaultProps<_HealpixCellsPrimitiveLayerProps> = {
  nside: { type: 'number', value: 1 },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  instanceCount: { type: 'number', value: 0 }
};

/** Indexed quad template: two triangles, four corner vertices per instance. */
const QUAD_INDICES = new Uint16Array([0, 1, 2, 0, 2, 3]);
const QUAD_POSITIONS = new Float32Array(12);

export class HealpixCellsPrimitiveLayer extends Layer<HealpixCellsPrimitiveLayerMergedProps> {
  static layerName = 'HealpixCellsPrimitiveLayer';
  static defaultProps = defaultProps;

  declare state: { model: Model | null };

  getNumInstances(): number {
    return this.props.instanceCount;
  }

  getShaders(): ReturnType<Layer['getShaders']> {
    return super.getShaders({
      vs: HEALPIX_VERTEX_SHADER,
      fs: HEALPIX_FRAGMENT_SHADER,
      modules: [project32, picking, healpixCellsShaderModule]
    });
  }

  initializeState(_context: LayerContext): void {
    this.getAttributeManager()!.addInstanced({
      cellIdLo: { size: 1, type: 'uint32', noAlloc: true },
      cellIdHi: { size: 1, type: 'uint32', noAlloc: true }
    });
  }

  updateState(params: UpdateParameters<this>): void {
    super.updateState(params);
    if (params.changeFlags.extensionsChanged || !this.state.model) {
      this.state.model?.destroy();
      this.state.model = this._getModel();
      this.getAttributeManager()!.invalidateAll();
    }
  }

  finalizeState(context: LayerContext): void {
    super.finalizeState(context);
    this.state.model?.destroy();
  }

  draw({ renderPass }: { renderPass: RenderPass }): void {
    const { model } = this.state;
    if (!model || this.props.instanceCount === 0) return;

    model.shaderInputs.setProps({
      healpixCells: computeHealpixCellsUniforms(
        this.props.nside,
        this.props.scheme
      )
    });
    model.setInstanceCount(this.props.instanceCount);
    model.draw(renderPass);
  }

  private _getModel(): Model {
    const parameters =
      this.context.device.type === 'webgpu'
        ? {
            depthWriteEnabled: true,
            depthCompare: 'less-equal' as const
          }
        : undefined;

    return new Model(this.context.device, {
      ...this.getShaders(),
      id: `${this.props.id}-model`,
      bufferLayout: this.getAttributeManager()!.getBufferLayouts(),
      geometry: new Geometry({
        topology: 'triangle-list',
        attributes: {
          indices: QUAD_INDICES,
          positions: { size: 3, value: QUAD_POSITIONS }
        }
      }),
      isInstanced: true,
      parameters
    });
  }
}
