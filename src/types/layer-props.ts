import type { CompositeLayerProps } from '@deck.gl/core';
import type { CellIdArray } from './cell-ids';

/** HEALPix pixel numbering scheme. */
export type HealpixScheme = 'nest' | 'ring';

/** Each HEALPix cell polygon = 4 corners + closing vertex = 5 vertices. */
export const VERTS_PER_CELL = 5;

export type { CellIdArray };

export type HealpixCellsLayerProps = {
  nside: number;
  /** HEALPix cell indices. */
  cellIds: CellIdArray;
  /** Numbering scheme. */
  scheme: HealpixScheme;
  /**
   * Per-frame per-cell fill colors.
   *
   * Each frame is an RGBA byte array (`0-255`) and must be
   * `cellIds.length * 4` long.
   */
  colorFrames: Uint8Array[];
  /**
   * Frame index to render from `colorFrames`.
   *
   * The layer clamps this value to `[0, colorFrames.length - 1]`.
   */
  currentFrame: number;
} & CompositeLayerProps;
