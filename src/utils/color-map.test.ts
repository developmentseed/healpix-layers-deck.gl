import { DEFAULT_COLORMAP, makeColorMap, validateColorMap } from './color-map';

describe('DEFAULT_COLORMAP', () => {
  it('is exactly 1024 bytes (256 × 4)', () => {
    expect(DEFAULT_COLORMAP.length).toBe(1024);
  });

  it('starts with black (0,0,0,255)', () => {
    expect(DEFAULT_COLORMAP[0]).toBe(0);
    expect(DEFAULT_COLORMAP[1]).toBe(0);
    expect(DEFAULT_COLORMAP[2]).toBe(0);
    expect(DEFAULT_COLORMAP[3]).toBe(255);
  });

  it('ends with white (255,255,255,255)', () => {
    expect(DEFAULT_COLORMAP[1020]).toBe(255);
    expect(DEFAULT_COLORMAP[1021]).toBe(255);
    expect(DEFAULT_COLORMAP[1022]).toBe(255);
    expect(DEFAULT_COLORMAP[1023]).toBe(255);
  });

  it('has a linear gray gradient', () => {
    expect(DEFAULT_COLORMAP[128 * 4 + 0]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 1]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 2]).toBe(128);
    expect(DEFAULT_COLORMAP[128 * 4 + 3]).toBe(255);
  });
});

describe('validateColorMap', () => {
  it('does not throw for exactly 1024 bytes', () => {
    expect(() => validateColorMap(new Uint8Array(1024))).not.toThrow();
  });

  it('throws for wrong length with a message mentioning 1024', () => {
    expect(() => validateColorMap(new Uint8Array(100))).toThrow('1024');
    expect(() => validateColorMap(new Uint8Array(0))).toThrow('1024');
    expect(() => validateColorMap(new Uint8Array(1025))).toThrow('1024');
  });
});

describe('makeColorMap', () => {
  it('returns exactly 1024 bytes (256 × 4)', () => {
    const map = makeColorMap(() => '#000');
    expect(map.length).toBe(1024);
    expect(map).toBeInstanceOf(Uint8Array);
  });

  it('invokes the callback 256 times with t in [0, 1] and index in [0, 255]', () => {
    const ts: number[] = [];
    const indices: number[] = [];
    makeColorMap((t, i) => {
      ts.push(t);
      indices.push(i);
      return '#000';
    });
    expect(ts.length).toBe(256);
    expect(indices.length).toBe(256);
    expect(indices[0]).toBe(0);
    expect(indices[255]).toBe(255);
    expect(ts[0]).toBe(0);
    expect(ts[255]).toBe(1);
    expect(ts[128]).toBeCloseTo(128 / 255);
  });

  it('accepts hex strings', () => {
    const map = makeColorMap(() => '#ff0000');
    expect(Array.from(map.slice(0, 4))).toEqual([255, 0, 0, 255]);
    expect(Array.from(map.slice(1020, 1024))).toEqual([255, 0, 0, 255]);
  });

  it('accepts short hex with alpha', () => {
    const map = makeColorMap(() => '#f008');
    expect(Array.from(map.slice(0, 4))).toEqual([255, 0, 0, 0x88]);
  });

  it('accepts 3-tuple byte arrays (alpha defaults to 255)', () => {
    const map = makeColorMap(() => [10, 20, 30]);
    expect(Array.from(map.slice(0, 4))).toEqual([10, 20, 30, 255]);
  });

  it('accepts 4-tuple byte arrays', () => {
    const map = makeColorMap(() => [10, 20, 30, 40]);
    expect(Array.from(map.slice(0, 4))).toEqual([10, 20, 30, 40]);
  });

  it('accepts normalized arrays in 0–1', () => {
    const map = makeColorMap((t) => ({
      normalized: true,
      rgba: [t, 0, 1 - t]
    }));
    expect(Array.from(map.slice(0, 4))).toEqual([0, 0, 255, 255]);
    expect(Array.from(map.slice(1020, 1024))).toEqual([255, 0, 0, 255]);
  });

  it('clamps byte values outside 0–255', () => {
    const map = makeColorMap(() => [-10, 300, 128]);
    expect(Array.from(map.slice(0, 4))).toEqual([0, 255, 128, 255]);
  });

  it('clamps normalized values outside 0–1', () => {
    const map = makeColorMap(() => ({
      normalized: true,
      rgba: [-0.5, 2, 0.5]
    }));
    expect(Array.from(map.slice(0, 4))).toEqual([0, 255, 128, 255]);
  });

  it('throws on invalid hex', () => {
    expect(() => makeColorMap(() => '#zzz')).toThrow('Invalid hex color');
  });

  it('throws on array with wrong channel count', () => {
    expect(() =>
      makeColorMap(() => [1, 2] as unknown as [number, number, number])
    ).toThrow('3 or 4 channels');
  });
});
