/**
 * Packs per-cell interleaved float values into an `RGBA32F` 2D texture layout.
 *
 * The texture is folded: cell `i` maps to texel `(i % width, floor(i / width))`.
 * Each texel has 4 floats (RGBA32F). Channels 0 through `dimensions-1` are
 * filled; the rest remain 0.
 *
 * @param values         Interleaved float values. Length = cellCount × dimensions.
 * @param dimensions     Number of values per cell (1–4).
 * @param cellCount      Total number of cells.
 * @param maxTextureSize GPU max texture dimension (from device limits).
 */
export function packValuesData(
  values: ArrayLike<number>,
  dimensions: number,
  cellCount: number,
  maxTextureSize: number
): { data: Float32Array; width: number; height: number } {
  if (cellCount === 0) {
    return { data: new Float32Array(4), width: 1, height: 1 };
  }

  const channelCount = Math.min(dimensions, 4);
  const width = Math.min(cellCount, maxTextureSize);
  const height = Math.ceil(cellCount / width);
  const data = new Float32Array(width * height * 4);

  for (let i = 0; i < cellCount; i++) {
    const x = i % width;
    const y = Math.floor(i / width);
    const dstBase = (y * width + x) * 4;
    const srcBase = i * dimensions;
    for (let d = 0; d < channelCount; d++) {
      data[dstBase + d] = (values as number[])[srcBase + d] ?? 0;
    }
  }

  return { data, width, height };
}
