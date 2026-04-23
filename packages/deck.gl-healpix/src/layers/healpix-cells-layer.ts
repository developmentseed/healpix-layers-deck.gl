import {
  CompositeLayer,
  DefaultProps,
  Layer,
  LayerExtension,
  UpdateParameters
} from '@deck.gl/core';
import type { Texture } from '@luma.gl/core';
import { splitCellIds } from '../utils/split-cell-ids';
import { HealpixCellsPrimitiveLayer } from './healpix-cells-primitive-layer';
import { HEALPIX_COLOR_EXTENSION } from '../extensions/healpix-color-extension';
import { resolveFrame, type ResolvedFrame } from '../utils/resolve-frame';
import { packValuesData } from '../utils/values-texture';
import type { CellIdArray } from '../types/cell-ids';
import type {
  HealpixCellsLayerProps,
  HealpixFrameObject
} from '../types/layer-props';

type _HealpixCellsLayerProps = {
  nside: number;
  cellIds: CellIdArray;
  scheme: 'nest' | 'ring';
  values: ArrayLike<number> | null;
  min: number;
  max: number;
  dimensions: 1 | 2 | 3 | 4;
  colorMap: Uint8Array | null;
  frames: HealpixFrameObject[] | null;
  currentFrame: number;
};

type HealpixCellsLayerState = {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
  cellIndex: Float32Array;
  valuesTexture: Texture | null;
  colorMapTexture: Texture | null;
  valuesTextureWidth: number;
  prevResolved: ResolvedFrame | null;
};

const defaultProps: DefaultProps<_HealpixCellsLayerProps> = {
  nside: { type: 'number', value: 0 },
  cellIds: { type: 'object', value: new Uint32Array(0), compare: true },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  values: { type: 'object', value: null, compare: true },
  min: { type: 'number', value: 0 },
  max: { type: 'number', value: 1 },
  dimensions: { type: 'number', value: 1 },
  colorMap: { type: 'object', value: null, compare: true },
  frames: { type: 'object', value: null, compare: true },
  currentFrame: { type: 'number', value: 0 }
};

export class HealpixCellsLayer extends CompositeLayer<HealpixCellsLayerProps> {
  static layerName = 'HealpixCellsLayer';
  static defaultProps = defaultProps;

  declare state: HealpixCellsLayerState;

  initializeState(): void {
    this.setState({
      cellIdLo: new Uint32Array(0),
      cellIdHi: new Uint32Array(0),
      cellIndex: new Float32Array(0),
      valuesTexture: null,
      colorMapTexture: null,
      valuesTextureWidth: 1,
      prevResolved: null
    });
    this._rebuildAll();
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return !!changeFlags.propsOrDataChanged;
  }

  updateState({ props }: UpdateParameters<this>): void {
    let resolved: ResolvedFrame;
    try {
      resolved = resolveFrame(props);
    } catch (e) {
      this.raiseError(e as Error, 'HealpixCellsLayer frame resolution failed');
      return;
    }

    const prev = this.state.prevResolved;

    const cellsChanged =
      !prev ||
      resolved.cellIds !== prev.cellIds ||
      resolved.nside !== prev.nside ||
      resolved.scheme !== prev.scheme;

    const valuesChanged =
      !prev ||
      resolved.values !== prev.values ||
      resolved.dimensions !== prev.dimensions ||
      resolved.cellIds.length !== prev.cellIds.length;

    const colorMapChanged = !prev || resolved.colorMap !== prev.colorMap;

    if (cellsChanged) this._splitCells(resolved);
    if (valuesChanged || cellsChanged) this._updateValuesTexture(resolved);
    if (colorMapChanged) this._updateColorMapTexture(resolved);

    this.setState({ prevResolved: resolved });
  }

  finalizeState(): void {
    this.state.valuesTexture?.destroy();
    this.state.colorMapTexture?.destroy();
  }

  renderLayers(): Layer[] {
    const {
      cellIdLo,
      cellIdHi,
      cellIndex,
      valuesTexture,
      colorMapTexture,
      valuesTextureWidth,
      prevResolved
    } = this.state;

    if (!valuesTexture || !colorMapTexture || !prevResolved) return [];

    const { cellIds, nside, scheme, min, max, dimensions } = prevResolved;
    const count = cellIds.length;
    if (count === 0) return [];

    return [
      new HealpixCellsPrimitiveLayer(
        this.getSubLayerProps({
          id: 'cells',
          nside,
          scheme,
          instanceCount: count,
          data: {
            length: count,
            attributes: {
              cellIdLo: { value: cellIdLo, size: 1 },
              cellIdHi: { value: cellIdHi, size: 1 },
              healpixCellIndex: { value: cellIndex, size: 1 }
            }
          },
          valuesTexture,
          colorMapTexture,
          uMin: min,
          uMax: max,
          uDimensions: dimensions,
          uValuesWidth: valuesTextureWidth,
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_EXTENSION
          ]
        })
      )
    ];
  }

  private _rebuildAll(): void {
    let resolved: ResolvedFrame;
    try {
      resolved = resolveFrame(this.props);
    } catch (e) {
      this.raiseError(e as Error, 'HealpixCellsLayer frame resolution failed');
      return;
    }
    this._splitCells(resolved);
    this._updateValuesTexture(resolved);
    this._updateColorMapTexture(resolved);
    this.setState({ prevResolved: resolved });
  }

  private _splitCells(resolved: ResolvedFrame): void {
    const { cellIds } = resolved;
    if (!cellIds?.length) {
      this.setState({
        cellIdLo: new Uint32Array(0),
        cellIdHi: new Uint32Array(0),
        cellIndex: new Float32Array(0)
      });
      return;
    }
    const { cellIdLo, cellIdHi } = splitCellIds(cellIds);
    const cellIndex = Float32Array.from({ length: cellIds.length }, (_, i) => i);
    this.setState({ cellIdLo, cellIdHi, cellIndex });
  }

  private _updateValuesTexture(resolved: ResolvedFrame): void {
    const { values, dimensions, cellIds } = resolved;
    const cellCount = cellIds.length;
    const oldTexture = this.state.valuesTexture;

    const { maxTextureDimension2D: maxTextureSize } =
      this.context.device.limits;
    const { data, width, height } = packValuesData(
      values,
      dimensions,
      cellCount,
      maxTextureSize
    );

    if (height > maxTextureSize) {
      this.raiseError(
        new Error(
          `Cannot pack ${cellCount} cells in values texture: requires ${width}×${height}, max is ${maxTextureSize}×${maxTextureSize}.`
        ),
        'HealpixCellsLayer values texture dimensions exceeded'
      );
      return;
    }

    const texture = this.context.device.createTexture({
      id: `${this.id}-values`,
      width,
      height,
      dimension: '2d',
      format: 'rgba32float',
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        mipmapFilter: 'none',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    texture.copyImageData({ data });

    this.setState({ valuesTexture: texture, valuesTextureWidth: width });
    oldTexture?.destroy();
  }

  private _updateColorMapTexture(resolved: ResolvedFrame): void {
    const { colorMap } = resolved;
    const oldTexture = this.state.colorMapTexture;

    const texture = this.context.device.createTexture({
      id: `${this.id}-colormap`,
      width: 256,
      height: 1,
      dimension: '2d',
      format: 'rgba8unorm',
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        mipmapFilter: 'none',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    texture.copyImageData({ data: colorMap });

    this.setState({ colorMapTexture: texture });
    oldTexture?.destroy();
  }
}
