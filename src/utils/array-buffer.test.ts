import { expandArrayBuffer } from './array-buffer';

describe('expandArrayBuffer', () => {
  it('expands scalar values per vertex', () => {
    const src = new Float32Array([10, 20]);
    const out = expandArrayBuffer(src, 3, 1);
    expect(Array.from(out)).toEqual([10, 10, 10, 20, 20, 20]);
  });

  it('expands vector values per vertex', () => {
    const src = new Float32Array([
      1,
      0,
      0,
      1, // cell 0 rgba
      0,
      1,
      0,
      1 // cell 1 rgba
    ]);
    const out = expandArrayBuffer(src, 2, 4);
    expect(Array.from(out)).toEqual([
      1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
    ]);
  });

  it('keeps the typed array constructor', () => {
    const src = new Uint8Array([1, 2]);
    const out = expandArrayBuffer(src, 2, 1);
    expect(out).toBeInstanceOf(Uint8Array);
  });

  it('throws for invalid dimensions', () => {
    const src = new Float32Array([1, 2, 3]);
    expect(() => expandArrayBuffer(src, 2, 2)).toThrow(
      /must be divisible by dimension/
    );
  });
});
