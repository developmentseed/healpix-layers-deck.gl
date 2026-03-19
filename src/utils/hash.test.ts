import { hashInt32Array } from './hash';

describe('hashInt32Array', () => {
  it('returns a stable hash for the same values', () => {
    const input = new Int32Array([1, 2, 3, 4, 5]);
    const h1 = hashInt32Array(input);
    const h2 = hashInt32Array(input);
    expect(h1).toBe(h2);
  });

  it('changes when input content changes', () => {
    const a = new Int32Array([1, 2, 3]);
    const b = new Int32Array([1, 2, 4]);
    expect(hashInt32Array(a)).not.toBe(hashInt32Array(b));
  });

  it('returns an unsigned 32-bit integer', () => {
    const h = hashInt32Array(new Int32Array([42]));
    expect(Number.isInteger(h)).toBe(true);
    expect(h).toBeGreaterThanOrEqual(0);
    expect(h).toBeLessThanOrEqual(0xffffffff);
  });
});
