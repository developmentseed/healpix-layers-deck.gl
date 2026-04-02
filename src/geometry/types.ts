import type { CellIdArray } from '../types/cell-ids';
import type { HealpixScheme } from '../types/layer-props';

/** Flat-buffer representation of geometry for a batch of HEALPix cells. */
export type GeometryResult = {
  coords: Float32Array;
  indexes: Uint32Array;
  triangles: Uint32Array;
};

export type WorkerTask = {
  nside: number;
  cellIds: CellIdArray;
  scheme: HealpixScheme;
};

export type WorkerMessage = { type: string; data: GeometryResult };
