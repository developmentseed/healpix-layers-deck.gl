import { hashTypedArray } from './hash';

describe('hashTypedArray', () => {
  it('returns a stable hash for the same values', () => {
    const input = new Int32Array([1, 2, 3, 4, 5]);
    expect(hashTypedArray(input)).toBe(hashTypedArray(input));
  });

  it('changes when input content changes', () => {
    const a = new Int32Array([1, 2, 3]);
    const b = new Int32Array([1, 2, 4]);
    expect(hashTypedArray(a)).not.toBe(hashTypedArray(b));
  });

  it('returns an unsigned 32-bit integer', () => {
    const h = hashTypedArray(new Int32Array([42]));
    expect(Number.isInteger(h)).toBe(true);
    expect(h).toBeGreaterThanOrEqual(0);
    expect(h).toBeLessThanOrEqual(0xffffffff);
  });

  it('handles Float64Array via raw words', () => {
    const a = new Float64Array([42, 100]);
    expect(hashTypedArray(a)).toBe(hashTypedArray(a));
    expect(hashTypedArray(a)).not.toBe(
      hashTypedArray(new Float64Array([42, 101]))
    );
  });
});
