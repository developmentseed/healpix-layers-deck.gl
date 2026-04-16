import { nest2fxy } from 'healpix-ts';
import { decomposeCellIds } from './decompose-cell-ids';

describe('decomposeCellIds', () => {
  it('decomposes NEST cell IDs at nside=1', () => {
    const cellIds = new Uint32Array([0, 1, 11]);
    const { faceIx, iy } = decomposeCellIds(cellIds, 1, 'nest');
    // nside=1: each face has 1 pixel. face=cellId, ix=0, iy=0
    expect(faceIx[0]).toBe(0 << 24); // face=0
    expect(faceIx[1]).toBe(1 << 24); // face=1
    expect(faceIx[2]).toBe(11 << 24); // face=11
    expect(iy[0]).toBe(0);
    expect(iy[1]).toBe(0);
    expect(iy[2]).toBe(0);
  });

  it('decomposes NEST cell IDs at nside=8', () => {
    // nest2fxy(8, 63) → {f:0, x:7, y:7}
    const cellIds = new Uint32Array([63]);
    const { faceIx, iy } = decomposeCellIds(cellIds, 8, 'nest');
    expect(faceIx[0]).toBe((0 << 24) | 7);
    expect(iy[0]).toBe(7);
  });

  it('decomposes RING cell IDs at nside=4', () => {
    // Verified: ring2fxy(4,0)→{f:0,x:3,y:3}, ring2fxy(4,1)→{f:1,x:3,y:3},
    //           ring2fxy(4,16)→{f:1,x:2,y:2}
    const cellIds = new Uint32Array([0, 1, 16]);
    const { faceIx, iy } = decomposeCellIds(cellIds, 4, 'ring');
    expect(faceIx[0]).toBe((0 << 24) | 3);
    expect(iy[0]).toBe(3);
    expect(faceIx[1]).toBe((1 << 24) | 3);
    expect(iy[1]).toBe(3);
    expect(faceIx[2]).toBe((1 << 24) | 2);
    expect(iy[2]).toBe(2);
  });

  it('decomposes large NEST cell IDs (Float64Array) at nside=262144', () => {
    // face 1, ix=0, iy=0 → cellId = 1 * 262144^2 = 68719476736
    const cellIds = new Float64Array([68719476736]);
    const { faceIx, iy } = decomposeCellIds(cellIds, 262144, 'nest');
    expect(faceIx[0]).toBe((1 << 24) | 0);
    expect(iy[0]).toBe(0);
  });

  it('packing matches healpix-ts decomposition for non-trivial cell', () => {
    const nside = 1024;
    const cellId = 2_500_000;
    const { f, x, y } = nest2fxy(nside, cellId);
    const { faceIx, iy } = decomposeCellIds(
      new Uint32Array([cellId]),
      nside,
      'nest'
    );
    expect(faceIx[0]).toBe((f << 24) | x);
    expect(iy[0]).toBe(y);
  });

  it('returns empty arrays for empty input', () => {
    const cellIds = new Uint32Array(0);
    const { faceIx, iy } = decomposeCellIds(cellIds, 8, 'nest');
    expect(faceIx.length).toBe(0);
    expect(iy.length).toBe(0);
  });
});
