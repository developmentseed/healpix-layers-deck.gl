import { nest2fxy, ring2fxy } from 'healpix-ts';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixScheme } from '../types/layer-props';

export type DecomposedCellIds = {
  faceIx: Uint32Array;
  iy: Uint32Array;
};

export function decomposeCellIds(
  cellIds: CellIdArray,
  nside: number,
  scheme: HealpixScheme
): DecomposedCellIds {
  const n = cellIds.length;
  const faceIx = new Uint32Array(n);
  const iy = new Uint32Array(n);
  const toFxy = scheme === 'nest' ? nest2fxy : ring2fxy;

  for (let i = 0; i < n; i++) {
    const { f, x, y } = toFxy(nside, cellIds[i]);
    // bits [31:24] = face (0-11), bits [23:0] = ix (supports nside up to 2^24)
    faceIx[i] = (f << 24) | x;
    iy[i] = y;
  }

  return { faceIx, iy };
}
