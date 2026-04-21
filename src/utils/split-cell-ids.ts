import type { CellIdArray } from '../types/cell-ids';

const TWO32 = 4294967296;

let sharedZero: Uint32Array = new Uint32Array(0);

export function getSharedZeroU32(minLength: number): Uint32Array {
  if (sharedZero.length < minLength) {
    sharedZero = new Uint32Array(Math.max(minLength, 1024));
  }
  return sharedZero;
}

export type SplitCellIds = {
  cellIdLo: Uint32Array;
  cellIdHi: Uint32Array;
};

/**
 * Split cell IDs into low/high u32 halves for two instance attributes → uvec2 in GLSL.
 *
 * Uint32Array / Int32Array: cellIdLo aliases input bytes; cellIdHi is a shared zero buffer.
 * Float64Array / Float32Array: explicit split loop.
 */
export function splitCellIds(cellIds: CellIdArray): SplitCellIds {
  const n = cellIds.length;
  if (cellIds instanceof Uint32Array) {
    return {
      cellIdLo: cellIds,
      cellIdHi: getSharedZeroU32(n)
    };
  }
  if (cellIds instanceof Int32Array) {
    return {
      cellIdLo: new Uint32Array(cellIds.buffer, cellIds.byteOffset, n),
      cellIdHi: getSharedZeroU32(n)
    };
  }
  const lo = new Uint32Array(n);
  const hi = new Uint32Array(n);
  for (let i = 0; i < n; i++) {
    const id = cellIds[i];
    lo[i] = id >>> 0;
    hi[i] = Math.floor(id / TWO32);
  }
  return { cellIdLo: lo, cellIdHi: hi };
}
