/**
 * HealpixCellsLayer — render arbitrary HEALPix cells by ID.
 *
 * This composite layer is responsible for:
 * - Splitting cell IDs into u32 low/high halves for GPU NEST/RING decode.
 * - Building a GPU texture that stores all animation frame colors.
 * - Rendering a `HealpixCellsPrimitiveLayer` sublayer that draws cells in the
 *   vertex shader and samples colors from the texture.
 */
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
import { HEALPIX_COLOR_FRAMES_EXTENSION } from '../extensions/healpix-color-frames-extension';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixCellsLayerProps } from '../types/layer-props';

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
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
  frameTexture: Texture | null;
  cellTextureWidth: number;
  frameCount: number;
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
 * Composite layer that renders HEALPix cells on the GPU and colors them from
 * a texture containing all animation frames.
 */
export class HealpixCellsLayer extends CompositeLayer<HealpixCellsLayerProps> {
  static layerName = 'HealpixCellsLayer';
  static defaultProps = defaultProps;

  declare state: HealpixCellsLayerState;

  initializeState(): void {
    this.setState({
      cellIdLo: new Uint32Array(0),
      cellIdHi: new Uint32Array(0),
      frameTexture: null,
      cellTextureWidth: 1,
      frameCount: 0
    });
    this._splitCellIds();
    this._updateColorTexture();
  }

  shouldUpdateState({ changeFlags }: UpdateParameters<this>): boolean {
    return !!changeFlags.propsOrDataChanged;
  }

  updateState({ props, oldProps }: UpdateParameters<this>): void {
    if (props.cellIds !== oldProps.cellIds) {
      this._splitCellIds();
    }
    if (
      props.cellIds !== oldProps.cellIds ||
      props.colorFrames !== oldProps.colorFrames
    ) {
      this._updateColorTexture();
    }
  }

  finalizeState(): void {
    this.state.frameTexture?.destroy();
  }

  renderLayers(): Layer[] {
    const { cellIdLo, cellIdHi, frameTexture, cellTextureWidth, frameCount } =
      this.state;
    const { cellIds, nside, scheme, currentFrame } = this.props;
    const count = cellIds.length;
    if (count === 0 || !frameTexture) return [];

    const frameIndex = Math.max(
      0,
      Math.min(frameCount - 1, Math.floor(currentFrame))
    );

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
              cellIdHi: { value: cellIdHi, size: 1 }
            }
          },
          frameTexture,
          frameIndex,
          cellTextureWidth,
          extensions: [
            ...((this.props.extensions as LayerExtension[]) || []),
            HEALPIX_COLOR_FRAMES_EXTENSION
          ]
        })
      )
    ];
  }

  private _splitCellIds(): void {
    const { cellIds } = this.props;
    if (!cellIds?.length) {
      this.setState({
        cellIdLo: new Uint32Array(0),
        cellIdHi: new Uint32Array(0)
      });
      return;
    }
    const { cellIdLo, cellIdHi } = splitCellIds(cellIds);
    this.setState({ cellIdLo, cellIdHi });
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
}
