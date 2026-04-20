import type {
  HealpixCellsLayerProps,
  HealpixFrameObject
} from '../types/layer-props';
import type { CellIdArray } from '../types/cell-ids';
import { DEFAULT_COLORMAP, validateColorMap } from './color-map';

/** The fully resolved, validated frame ready for GPU upload. */
export type ResolvedFrame = {
  nside: number;
  scheme: 'nest' | 'ring';
  cellIds: CellIdArray;
  values: ArrayLike<number>;
  min: number;
  max: number;
  dimensions: 1 | 2 | 3 | 4;
  colorMap: Uint8Array;
};

/**
 * Resolve the effective frame from layer props.
 *
 * When `props.frames` is set and non-empty, the frame at `props.currentFrame`
 * (clamped to valid range) is merged on top of root props. When `frames` is
 * absent or empty, root props are used directly.
 *
 * Throws with a descriptive message if required fields are missing or invalid.
 */
export function resolveFrame(props: HealpixCellsLayerProps): ResolvedFrame {
  // Determine which frame object to use (empty object = single-frame mode)
  let frame: Partial<HealpixFrameObject> = {};
  if (props.frames && props.frames.length > 0) {
    const idx = Math.max(
      0,
      Math.min(props.frames.length - 1, Math.floor(props.currentFrame ?? 0))
    );
    frame = props.frames[idx] ?? {};
  }

  // nside
  const nside = frame.nside ?? props.nside;
  if (!nside) {
    throw new Error(
      'HealpixCellsLayer: nside is required — set it on the layer or on each frame object'
    );
  }

  // cellIds
  const cellIds = frame.cellIds ?? props.cellIds;
  if (!cellIds) {
    throw new Error(
      'HealpixCellsLayer: cellIds is required — set it on the layer or on each frame object'
    );
  }

  // values
  const values = frame.values ?? props.values;
  if (!values) {
    throw new Error(
      'HealpixCellsLayer: values is required — set it on the layer or on each frame object'
    );
  }

  // colorMap
  const colorMap = frame.colorMap ?? props.colorMap ?? DEFAULT_COLORMAP;
  validateColorMap(colorMap); // throws if wrong length

  const dimensions = (frame.dimensions ??
    props.dimensions ??
    1) as ResolvedFrame['dimensions'];

  // values length check
  const expectedLen = cellIds.length * dimensions;
  if (values.length !== expectedLen) {
    throw new Error(
      `HealpixCellsLayer: values.length (${values.length}) must equal cellIds.length × dimensions (${cellIds.length} × ${dimensions} = ${expectedLen})`
    );
  }

  return {
    nside,
    scheme: frame.scheme ?? props.scheme ?? 'nest',
    cellIds: cellIds as CellIdArray,
    values,
    min: frame.min ?? props.min ?? 0,
    max: frame.max ?? props.max ?? 1,
    dimensions,
    colorMap
  };
}
