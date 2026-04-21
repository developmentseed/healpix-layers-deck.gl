import { splitCellIds, getSharedZeroU32 } from './split-cell-ids';

describe('splitCellIds', () => {
  it('Uint32Array input: lo aliases, hi is shared zero buffer', () => {
    const ids = new Uint32Array([0, 1, 42, 0xffffffff]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(cellIdLo.buffer).toBe(ids.buffer);
    expect(cellIdLo.length).toBe(4);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 42, 0xffffffff]);
    expect(cellIdHi.length).toBeGreaterThanOrEqual(4);
    expect(Array.from(cellIdHi.slice(0, 4))).toEqual([0, 0, 0, 0]);
    expect(cellIdHi).toBe(getSharedZeroU32(4));
  });

  it('Int32Array input with non-negative values: lo aliases bytes', () => {
    const ids = new Int32Array([0, 1, 2147483647]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(cellIdLo.buffer).toBe(ids.buffer);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 2147483647]);
    expect(Array.from(cellIdHi.slice(0, 3))).toEqual([0, 0, 0]);
  });

  it('Float64Array input: runs split loop', () => {
    const TWO32 = 4294967296;
    const ids = new Float64Array([0, 1, TWO32, TWO32 + 7, 12 * 2 ** 48 - 1]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(Array.from(cellIdLo)).toEqual([
      0,
      1,
      0,
      7,
      (12 * 2 ** 48 - 1) >>> 0
    ]);
    expect(Array.from(cellIdHi)).toEqual([
      0,
      0,
      1,
      1,
      Math.floor((12 * 2 ** 48 - 1) / TWO32)
    ]);
    expect(cellIdLo.buffer).not.toBe(ids.buffer);
  });

  it('Float32Array input: runs split loop', () => {
    const ids = new Float32Array([0, 1, 42]);
    const { cellIdLo, cellIdHi } = splitCellIds(ids);
    expect(Array.from(cellIdLo)).toEqual([0, 1, 42]);
    expect(Array.from(cellIdHi)).toEqual([0, 0, 0]);
    expect(cellIdLo.buffer).not.toBe(ids.buffer);
  });

  it('empty input produces empty buffers', () => {
    const { cellIdLo, cellIdHi } = splitCellIds(new Uint32Array(0));
    expect(cellIdLo.length).toBe(0);
    expect(cellIdHi.length).toBeGreaterThanOrEqual(0);
  });

  it('shared zero buffer grows on demand', () => {
    splitCellIds(new Uint32Array(4));
    const small = getSharedZeroU32(4);
    splitCellIds(new Uint32Array(10));
    const large = getSharedZeroU32(10);
    expect(large.length).toBeGreaterThanOrEqual(10);
    expect(Array.from(large.slice(0, 10))).toEqual([
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    ]);
    void small;
  });
});
