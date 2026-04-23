import { packValuesData } from './values-texture';

const MAX = 4096;

describe('packValuesData', () => {
  it('dim=1: fills only R channel, G/B/A are 0', () => {
    const { data, width, height } = packValuesData([0.5], 1, 1, MAX);
    expect(width).toBe(1);
    expect(height).toBe(1);
    expect(data.length).toBe(4);
    expect(data[0]).toBeCloseTo(0.5); // R
    expect(data[1]).toBe(0); // G
    expect(data[2]).toBe(0); // B
    expect(data[3]).toBe(0); // A
  });

  it('dim=2: fills R and G, B/A are 0', () => {
    const { data } = packValuesData([0.3, 0.7], 2, 1, MAX);
    expect(data[0]).toBeCloseTo(0.3);
    expect(data[1]).toBeCloseTo(0.7);
    expect(data[2]).toBe(0);
    expect(data[3]).toBe(0);
  });

  it('dim=3: fills R, G, B; A is 0', () => {
    const { data } = packValuesData([0.1, 0.2, 0.3], 3, 1, MAX);
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    expect(data[2]).toBeCloseTo(0.3);
    expect(data[3]).toBe(0);
  });

  it('dim=4: fills all channels', () => {
    const { data } = packValuesData([0.1, 0.2, 0.3, 0.4], 4, 1, MAX);
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    expect(data[2]).toBeCloseTo(0.3);
    expect(data[3]).toBeCloseTo(0.4);
  });

  it('two cells, dim=1: each in a separate texel', () => {
    const { data, width } = packValuesData([0.2, 0.8], 1, 2, MAX);
    expect(width).toBe(2);
    expect(data[0]).toBeCloseTo(0.2); // cell 0 at x=0
    expect(data[4]).toBeCloseTo(0.8); // cell 1 at x=1
  });

  it('multiple cells, dim=2: each cell gets its own texel', () => {
    const values = [0.1, 0.2, 0.3, 0.4]; // 2 cells × 2 dims
    const { data } = packValuesData(values, 2, 2, MAX);
    expect(data[0]).toBeCloseTo(0.1);
    expect(data[1]).toBeCloseTo(0.2);
    expect(data[4]).toBeCloseTo(0.3);
    expect(data[5]).toBeCloseTo(0.4);
  });

  it('folds into 2D when cellCount > maxTextureSize', () => {
    // 5 cells, maxTextureSize=3 → width=3, height=2
    const values = [0.1, 0.2, 0.3, 0.4, 0.5];
    const { data, width, height } = packValuesData(values, 1, 5, 3);
    expect(width).toBe(3);
    expect(height).toBe(2);
    // cell 3: x=0, y=1 → dstBase=(1*3+0)*4=12
    expect(data[12]).toBeCloseTo(0.4);
    // cell 4: x=1, y=1 → dstBase=(1*3+1)*4=16
    expect(data[16]).toBeCloseTo(0.5);
  });

  it('returns 1×1 zero texel for cellCount=0', () => {
    const { data, width, height } = packValuesData([], 1, 0, MAX);
    expect(width).toBe(1);
    expect(height).toBe(1);
    expect(data.length).toBe(4);
    expect(data[0]).toBe(0);
  });
});
