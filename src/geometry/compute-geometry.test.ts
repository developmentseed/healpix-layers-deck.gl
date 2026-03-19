type GeometryResult = {
  coords: Float32Array;
  indexes: Uint32Array;
  triangles: Uint32Array;
};

const runMock = jest.fn<Promise<GeometryResult[]>, [unknown[]]>();

jest.mock('../utils/worker-pool', () => ({
  WorkerPool: jest.fn().mockImplementation(() => ({
    run: runMock
  }))
}));

function makeChunk(count: number): GeometryResult {
  const coords = new Float32Array(count * 10);
  const indexes = new Uint32Array(count);
  const triangles = new Uint32Array(count * 6);

  for (let i = 0; i < count; i++) {
    indexes[i] = i * 5;
    const base = i * 5;
    const triOffset = i * 6;
    triangles[triOffset + 0] = base + 0;
    triangles[triOffset + 1] = base + 1;
    triangles[triOffset + 2] = base + 2;
    triangles[triOffset + 3] = base + 0;
    triangles[triOffset + 4] = base + 2;
    triangles[triOffset + 5] = base + 3;
  }

  return { coords, indexes, triangles };
}

describe('computeGeometry', () => {
  beforeEach(() => {
    jest.resetModules();
    runMock.mockReset();
    Object.defineProperty(globalThis, 'navigator', {
      configurable: true,
      value: { hardwareConcurrency: 4 }
    });
  });

  it('dispatches small batches in one worker task', async () => {
    const single = makeChunk(2);
    runMock.mockResolvedValueOnce([single]);

    const { computeGeometry } = await import('./compute-geometry');
    const result = await computeGeometry(4, new Int32Array([10, 11]), 'nest');

    expect(runMock).toHaveBeenCalledTimes(1);
    expect(runMock.mock.calls[0][0]).toHaveLength(1);
    expect(result).toBe(single);
  });

  it('splits and merges large batches with vertex offsets', async () => {
    const total = 10_001;
    const ids = new Int32Array(total);
    for (let i = 0; i < total; i++) ids[i] = i;

    const counts = [2501, 2501, 2501, 2498];
    runMock.mockResolvedValueOnce(counts.map((c) => makeChunk(c)));

    const { computeGeometry } = await import('./compute-geometry');
    const merged = await computeGeometry(8, ids, 'ring');

    expect(runMock).toHaveBeenCalledTimes(1);
    expect(runMock.mock.calls[0][0]).toHaveLength(4);
    expect(merged.indexes).toHaveLength(total);
    expect(merged.triangles).toHaveLength(total * 6);

    // First cell of second chunk should be offset by first chunk vertex count.
    expect(merged.indexes[counts[0]]).toBe(counts[0] * 5);
  });

  it('reuses cache for same geometry key', async () => {
    runMock.mockResolvedValueOnce([makeChunk(1)]);

    const { computeGeometry } = await import('./compute-geometry');
    const a = await computeGeometry(2, new Int32Array([42]), 'nest');
    const b = await computeGeometry(2, new Int32Array([42]), 'nest');

    expect(a).toBe(b);
    expect(runMock).toHaveBeenCalledTimes(1);
  });
});
