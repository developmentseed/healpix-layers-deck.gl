/**
 * Geometry generation for a batch of HEALPix cells.
 */
import { cornersNestLonLat, cornersRingLonLat } from 'healpix-ts';
import { earcut } from '@math.gl/polygon';
import type { CellIdArray } from '../types/cell-ids';

self.onmessage = (e: MessageEvent) => {
  const {
    nside,
    cellIds,
    scheme = 'nest'
  } = e.data as {
    nside: number;
    cellIds: CellIdArray;
    scheme?: 'nest' | 'ring';
  };

  const cornersFn = scheme === 'nest' ? cornersNestLonLat : cornersRingLonLat;

  const cells = cellIds.length;
  const coords = new Float32Array(cells * 10);
  const indexes = new Uint32Array(cells);
  const triangles = new Uint32Array(cells * 6);

  for (let i = 0; i < cells; i++) {
    const corners = cornersFn(nside, Number(cellIds[i]));
    const poly = corners.concat(corners[0]).flat();
    coords.set(poly, i * 10);
    indexes[i] = i * 5;

    const triangleIdx = earcut(poly);
    triangles.set(
      triangleIdx.map((idx) => idx + i * 5),
      i * 6
    );
  }

  self.postMessage(
    { type: 'data', data: { coords, indexes, triangles } },
    { transfer: [coords.buffer, indexes.buffer, triangles.buffer] }
  );
};
