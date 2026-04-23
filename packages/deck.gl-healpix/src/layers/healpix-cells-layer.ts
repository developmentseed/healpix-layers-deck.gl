/**
 * HealpixCellsLayer — render arbitrary HEALPix cells by ID with GPU color
 * computation.
 */
import {
  CompositeLayer,
  DefaultProps,
  Layer,
  LayerExtension,
  UpdateParameters
} from '@deck.gl/core';
import { SolidPolygonLayer } from '@deck.gl/layers';
import type { Texture } from '@luma.gl/core';
import { expandArrayBuffer } from '../utils/array-buffer';
import { computeGeometry } from '../geometry/compute-geometry';
import { HEALPIX_COLOR_EXTENSION } from '../extensions/healpix-color-extension';
import { resolveFrame, type ResolvedFrame } from '../utils/resolve-frame';
import { packValuesData } from '../utils/values-texture';
import type { CellIdArray } from '../types/cell-ids';
import {
  VERTS_PER_CELL,
  type HealpixCellsLayerProps,
  type HealpixFrameObject
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
  coords: Float32Array | null;
  indexes: Uint32Array | null;
  triangles: Uint32Array | null;
  cellVertexIndices: Float32Array | null;
  valuesTexture: Texture | null;
  colorMapTexture: Texture | null;
  valuesTextureWidth: number;
  ready: boolean;
  /** Last resolved frame — used for change detection in updateState. */
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

/**
 * Composite layer that renders HEALPix cells as polygons whose colors are
 * computed on the GPU from per-cell float values and a colorMap LUT.
 */
export class HealpixCellsLayer extends CompositeLayer<HealpixCellsLayerProps> {
  static layerName = 'HealpixCellsLayer';
  static defaultProps = defaultProps;

  declare state: HealpixCellsLayerState;

  /** Monotonic token used to ignore stale async geometry builds. */
  private _version = 0;

  initializeState(): void {
    this.setState(this._getEmptyState());
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

    const geometryChanged =
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

    if (geometryChanged) this._buildGeometry(resolved);
    if (valuesChanged || geometryChanged) this._updateValuesTexture(resolved);
    if (colorMapChanged) this._updateColorMapTexture(resolved);

    this.setState({ prevResolved: resolved });
  }

  finalizeState(): void {
    this.state.valuesTexture?.destroy();
    this.state.colorMapTexture?.destroy();
  }

  renderLayers(): Layer[] {
    const {
      coords,
      indexes,
      triangles,
      cellVertexIndices,
      valuesTexture,
      colorMapTexture,
      valuesTextureWidth,
      ready,
      prevResolved
    } = this.state;

    if (!ready || !coords || !cellVertexIndices) return [];
    if (!valuesTexture || !colorMapTexture || !prevResolved) return [];

    const { cellIds, min, max, dimensions } = prevResolved;
    const count = cellIds.length;
    if (count === 0) return [];

    return [
      new SolidPolygonLayer(
        this.getSubLayerProps({
          id: 'cells',
          valuesTexture,
          colorMapTexture,
          uMin: min,
          uMax: max,
          uDimensions: dimensions,
          uValuesWidth: valuesTextureWidth,
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_EXTENSION
          ],
          data: {
            length: count,
            startIndices: indexes,
            attributes: {
              getPolygon: { value: coords, size: 2 },
              indices: { value: triangles, size: 1 },
              healpixCellIndex: { value: cellVertexIndices, size: 1 }
            }
          },
          _normalize: false
        })
      )
    ];
  }

  /**
   * Rebuild all resources from scratch (called on first init, before
   * `updateState` has a chance to diff).
   */
  private _rebuildAll(): void {
    let resolved: ResolvedFrame;
    try {
      resolved = resolveFrame(this.props);
    } catch (e) {
      this.raiseError(e as Error, 'HealpixCellsLayer frame resolution failed');
      return;
    }
    this._buildGeometry(resolved);
    this._updateValuesTexture(resolved);
    this._updateColorMapTexture(resolved);
    this.setState({ prevResolved: resolved });
  }

  /** Create an empty state snapshot used on initialization/reset. */
  private _getEmptyState(): HealpixCellsLayerState {
    return {
      coords: null,
      indexes: null,
      triangles: null,
      cellVertexIndices: null,
      valuesTexture: null,
      colorMapTexture: null,
      valuesTextureWidth: 1,
      ready: false,
      prevResolved: null
    };
  }

  /** Compute HEALPix polygon geometry asynchronously. */
  private async _buildGeometry(resolved: ResolvedFrame): Promise<void> {
    const { nside, cellIds, scheme } = resolved;

    this.setState({
      coords: null,
      indexes: null,
      triangles: null,
      cellVertexIndices: null,
      ready: false
    });

    if (!cellIds?.length) return;

    const v = ++this._version;
    const { coords, indexes, triangles } = await computeGeometry(
      nside,
      cellIds,
      scheme
    );
    if (this._version !== v) return;

    const perCellIndex = Float32Array.from(
      { length: cellIds.length },
      (_unused, index) => index
    );
    const cellVertexIndices = expandArrayBuffer(
      perCellIndex,
      VERTS_PER_CELL,
      1
    );

    this.setState({
      coords,
      indexes,
      triangles,
      cellVertexIndices,
      ready: true
    });
  }

  /**
   * Build and upload an RGBA32F values texture.
   *
   * Each texel stores the float values for one cell in channels 0–
   * (dimensions-1). The texture is folded: cell i → texel (i % width,
   * floor(i / width)).
   */
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

  /**
   * Build and upload an RGBA8 256×1 colorMap texture.
   *
   * Index 0 maps to `min`, index 255 maps to `max`.
   */
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
