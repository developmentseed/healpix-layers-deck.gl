import { resolveFrame } from './resolve-frame';
import { DEFAULT_COLORMAP } from './color-map';
import type { HealpixCellsLayerProps } from '../types/layer-props';

const validIds = new Uint32Array([1, 2, 3]);
const validValues = new Float32Array([0.1, 0.2, 0.3]); // dim=1, 3 cells

function makeProps(
  overrides: Partial<HealpixCellsLayerProps>
): HealpixCellsLayerProps {
  return {
    nside: 64,
    cellIds: validIds,
    values: validValues,
    ...overrides
  } as HealpixCellsLayerProps;
}

describe('resolveFrame — single-frame mode (no frames array)', () => {
  it('uses root props and applies defaults', () => {
    const result = resolveFrame(makeProps({}));
    expect(result.nside).toBe(64);
    expect(result.cellIds).toBe(validIds);
    expect(result.values).toBe(validValues);
    expect(result.scheme).toBe('nest');
    expect(result.min).toBe(0);
    expect(result.max).toBe(1);
    expect(result.dimensions).toBe(1);
    expect(result.colorMap).toBe(DEFAULT_COLORMAP);
  });

  it('uses explicit root overrides', () => {
    const myColorMap = new Uint8Array(1024);
    const result = resolveFrame(
      makeProps({
        scheme: 'ring',
        min: -1,
        max: 10,
        dimensions: 3,
        colorMap: myColorMap,
        values: new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) // 3 cells × 3 dims
      })
    );
    expect(result.scheme).toBe('ring');
    expect(result.min).toBe(-1);
    expect(result.max).toBe(10);
    expect(result.dimensions).toBe(3);
    expect(result.colorMap).toBe(myColorMap);
  });
});

describe('resolveFrame — multi-frame mode', () => {
  it('uses the current frame at currentFrame index', () => {
    const vals0 = new Float32Array([0.1, 0.2, 0.3]);
    const vals1 = new Float32Array([0.4, 0.5, 0.6]);
    const result = resolveFrame(
      makeProps({
        frames: [{ values: vals0 }, { values: vals1 }],
        currentFrame: 1
      })
    );
    expect(result.values).toBe(vals1);
  });

  it('frame fields override root props', () => {
    const frameIds = new Uint32Array([9, 8]);
    const frameVals = new Float32Array([0.5, 0.6]);
    const result = resolveFrame(
      makeProps({
        frames: [{ cellIds: frameIds, values: frameVals, min: -5, max: 5 }],
        currentFrame: 0
      })
    );
    expect(result.cellIds).toBe(frameIds);
    expect(result.values).toBe(frameVals);
    expect(result.min).toBe(-5);
    expect(result.max).toBe(5);
  });

  it('root props fill gaps not set on frame', () => {
    const myColorMap = new Uint8Array(1024);
    const result = resolveFrame(
      makeProps({
        colorMap: myColorMap,
        frames: [{ values: validValues }],
        currentFrame: 0
      })
    );
    expect(result.colorMap).toBe(myColorMap);
    expect(result.nside).toBe(64); // from root
  });

  it('clamps currentFrame to [0, frames.length - 1]', () => {
    const vals0 = new Float32Array([0.1, 0.2, 0.3]);
    const vals1 = new Float32Array([0.4, 0.5, 0.6]);
    const props = makeProps({
      frames: [{ values: vals0 }, { values: vals1 }]
    });

    expect(resolveFrame({ ...props, currentFrame: 99 }).values).toBe(vals1);
    expect(resolveFrame({ ...props, currentFrame: -5 }).values).toBe(vals0);
  });

  it('empty frames array falls back to root props', () => {
    const result = resolveFrame(makeProps({ frames: [] }));
    expect(result.values).toBe(validValues);
  });
});

describe('resolveFrame — validation', () => {
  it('throws if nside is missing', () => {
    expect(() =>
      resolveFrame({
        cellIds: validIds,
        values: validValues
      } as HealpixCellsLayerProps)
    ).toThrow(/nside/);
  });

  it('throws if cellIds is missing', () => {
    expect(() =>
      resolveFrame({ nside: 64, values: validValues } as HealpixCellsLayerProps)
    ).toThrow(/cellIds/);
  });

  it('throws if values is missing', () => {
    expect(() =>
      resolveFrame({ nside: 64, cellIds: validIds } as HealpixCellsLayerProps)
    ).toThrow(/values/);
  });

  it('throws if colorMap has wrong length', () => {
    expect(() =>
      resolveFrame(makeProps({ colorMap: new Uint8Array(100) }))
    ).toThrow(/1024/);
  });

  it('throws if values.length !== cellIds.length * dimensions', () => {
    expect(
      () => resolveFrame(makeProps({ values: new Float32Array([0.1, 0.2]) })) // 2 values, 3 cells × dim=1
    ).toThrow(/values\.length/);
  });
});
