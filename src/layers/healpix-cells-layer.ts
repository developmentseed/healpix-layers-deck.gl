/**
 * HealpixCellsLayer — render arbitrary HEALPix cells by ID.
 *
 * This composite layer is responsible for:
 * - Computing polygon geometry for requested HEALPix cells.
 * - Building a GPU texture that stores all animation frame colors.
 * - Rendering a `SolidPolygonLayer` sublayer that samples colors from texture.
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
import { VERTS_PER_CELL } from '../types/layer-props';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixCellsLayerProps } from '../types/layer-props';
import { HEALPIX_COLOR_FRAMES_EXTENSION } from '../extensions/healpix-color-frames-extension';

/** Internal prop subset used by default prop declarations. */
type _HealpixCellsLayerProps = {
  nside: number;
  cellIds: CellIdArray;
  scheme: 'nest' | 'ring';
  colorFrames: Uint8Array[];
  currentFrame: number;
};

/** Runtime state owned by `HealpixCellsLayer`. */
type HealpixCellsLayerState = {
  coords: Float32Array | null;
  indexes: Uint32Array | null;
  triangles: Uint32Array | null;
  cellVertexIndices: Uint32Array | null;
  frameTexture: Texture | null;
  cellTextureWidth: number;
  frameCount: number;
  ready: boolean;
};

/**
 * Return shape for packed texture data that is uploaded to the GPU.
 */
type TextureData = {
  colors: Uint8Array;
  width: number;
  height: number;
  depth: number;
  frameCount: number;
};

/** Fallback texel when there are no cells or frames. */
const EMPTY_RGBA_TEXEL = new Uint8Array([0, 0, 0, 0]);

/** deck.gl-style default props for the composite layer. */
const defaultProps: DefaultProps<_HealpixCellsLayerProps> = {
  nside: { type: 'number', value: 0 },
  cellIds: { type: 'object', value: new Uint32Array(0), compare: true },
  // @ts-expect-error deck.gl DefaultProps has no 'string' type.
  scheme: { type: 'string', value: 'nest' },
  colorFrames: {
    type: 'object',
    value: [],
    compare: true
  },
  currentFrame: { type: 'number', value: 0 }
};

/**
 * Composite layer that renders HEALPix cells as polygons and colors them from
 * a GPU texture containing all animation frames.
 */
export class HealpixCellsLayer extends CompositeLayer<HealpixCellsLayerProps> {
  static layerName = 'HealpixCellsLayer';
  static defaultProps = defaultProps;

  declare state: HealpixCellsLayerState;

  /** Monotonic token used to ignore stale async geometry builds. */
  private _version = 0;

  /**
   * Initialize empty state, then kick off initial geometry + color-texture setup.
   */
  initializeState(): void {
    this.setState(this._getEmptyState());
    this._buildGeometry();
    this._updateColorTexture();
  }

  /** Re-run state updates whenever relevant props or data changed. */
  shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return !!changeFlags.propsOrDataChanged;
  }

  /**
   * Recompute geometry or refresh frame texture depending on prop changes.
   */
  updateState({ props, oldProps }: UpdateParameters<this>): void {
    if (
      props.cellIds !== oldProps.cellIds ||
      props.nside !== oldProps.nside ||
      props.scheme !== oldProps.scheme
    ) {
      this._buildGeometry();
    }
    if (
      props.cellIds !== oldProps.cellIds ||
      props.colorFrames !== oldProps.colorFrames
    ) {
      this._updateColorTexture();
    }
  }

  /** Release GPU resources created by this layer. */
  finalizeState(): void {
    this.state.frameTexture?.destroy();
  }

  /**
   * Render one `SolidPolygonLayer` sublayer.
   */
  renderLayers(): Layer[] {
    const {
      coords,
      indexes,
      triangles,
      cellVertexIndices,
      frameTexture,
      cellTextureWidth,
      frameCount,
      ready
    } = this.state;
    if (!ready || !coords) return [];

    const { cellIds, currentFrame } = this.props;
    const count = cellIds.length;
    const frameIndex = this._clampFrameIndex(currentFrame, frameCount);

    if (!frameTexture || !cellVertexIndices) return [];

    return [
      new SolidPolygonLayer(
        this.getSubLayerProps({
          id: 'cells',
          frameTexture,
          frameIndex,
          cellTextureWidth,
          // Preserve user-supplied extensions and append HEALPix texture extension.
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_FRAMES_EXTENSION
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
   * Create an empty state snapshot used on initialization/reset.
   */
  private _getEmptyState(): HealpixCellsLayerState {
    return {
      coords: null,
      indexes: null,
      triangles: null,
      cellVertexIndices: null,
      frameTexture: null,
      cellTextureWidth: 1,
      frameCount: 0,
      ready: false
    };
  }

  /**
   * Compute HEALPix geometry asynchronously.
   */
  private async _buildGeometry(): Promise<void> {
    const { nside, cellIds, scheme } = this.props;

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

    const cellVertexIndices = expandArrayBuffer(
      Uint32Array.from({ length: cellIds.length }, (_unused, index) => index),
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
   * Build and upload the `(cell, frame) -> RGBA` texture to the GPU.
   *
   * The previous texture is destroyed once the new one is ready.
   */
  private _updateColorTexture(): void {
    const { cellIds, colorFrames } = this.props;
    const cellCount = cellIds.length;

    const oldTexture = this.state.frameTexture;
    const data = this._buildTextureData(cellCount, colorFrames);
    const texture = this.context.device.createTexture({
      id: `${this.id}-color-frames`,
      width: data.width,
      height: data.height,
      depth: data.depth,
      dimension: '2d-array',
      format: 'rgba8unorm',
      sampler: {
        minFilter: 'nearest',
        magFilter: 'nearest',
        mipmapFilter: 'none',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge'
      }
    });
    texture.copyImageData({ data: data.colors });

    this.setState({
      frameTexture: texture,
      cellTextureWidth: data.width,
      frameCount: data.frameCount
    });

    oldTexture?.destroy();
  }

  /**
   * Pack per-frame color arrays into one contiguous byte array.
   *
   * Texture layout:
   * - width  = folded row width (`<= maxTextureDimension2D`)
   * - height = rows needed for one frame
   * - depth  = number of frames (texture array layers)
   * - texel  = RGBA for one cell in one frame layer
   */
  private _buildTextureData(
    cellCount: number,
    colorFrames: Uint8Array[]
  ): TextureData {
    const frameCount = colorFrames.length;
    if (cellCount === 0 || frameCount === 0) {
      return {
        colors: EMPTY_RGBA_TEXEL,
        width: 1,
        height: 1,
        depth: 1,
        frameCount: 0
      };
    }

    const { maxTextureDimension2D: maxTextureSize, maxTextureArrayLayers } =
      this.context.device.limits;
    const width = Math.min(cellCount, maxTextureSize);
    const height = Math.ceil(cellCount / width);

    if (height > maxTextureSize) {
      this.raiseError(
        new Error(
          `Cannot pack ${cellCount} cells in texture: requires ${width}x${height}, max is ${maxTextureSize}x${maxTextureSize}.`
        ),
        'HealpixCellsLayer texture dimensions exceeded'
      );
      return {
        colors: EMPTY_RGBA_TEXEL,
        width: 1,
        height: 1,
        depth: 1,
        frameCount: 0
      };
    }

    if (frameCount > maxTextureArrayLayers) {
      this.raiseError(
        new Error(
          `Cannot upload ${frameCount} frames: max texture array layers is ${maxTextureArrayLayers}.`
        ),
        'HealpixCellsLayer texture array depth exceeded'
      );
      return {
        colors: EMPTY_RGBA_TEXEL,
        width: 1,
        height: 1,
        depth: 1,
        frameCount: 0
      };
    }

    const frameSize = cellCount * 4;
    const layerSize = width * height * 4;
    const colors = new Uint8Array(layerSize * frameCount);

    for (let frameIndex = 0; frameIndex < frameCount; frameIndex++) {
      const frame = colorFrames[frameIndex];
      if (frame.length !== frameSize) {
        this.raiseError(
          new Error(
            `Frame ${frameIndex} has ${frame.length} values; expected ${frameSize}.`
          ),
          'HealpixCellsLayer invalid color frame'
        );
        return {
          colors: EMPTY_RGBA_TEXEL,
          width: 1,
          height: 1,
          depth: 1,
          frameCount: 0
        };
      }

      const frameOffset = frameIndex * layerSize;
      for (let cellIndex = 0; cellIndex < cellCount; cellIndex++) {
        const srcOffset = cellIndex * 4;
        const x = cellIndex % width;
        const y = Math.floor(cellIndex / width);
        const dstOffset = frameOffset + (y * width + x) * 4;

        colors[dstOffset] = frame[srcOffset];
        colors[dstOffset + 1] = frame[srcOffset + 1];
        colors[dstOffset + 2] = frame[srcOffset + 2];
        colors[dstOffset + 3] = frame[srcOffset + 3];
      }
    }

    return { colors, width, height, depth: frameCount, frameCount };
  }

  /**
   * Clamp incoming frame index to valid texture row bounds.
   */
  private _clampFrameIndex(currentFrame: number, frameCount: number): number {
    if (frameCount <= 0) return 0;
    return Math.max(0, Math.min(frameCount - 1, Math.floor(currentFrame)));
  }
}
