/**
 * Expands a typed array by replicating each element `vertexPerCell` times.
 *
 * Given a source buffer where each logical entry is `dimension` contiguous
 * values (e.g. `[x0, y0, x1, y1, ...]`), this produces a new buffer of the
 * same type where every entry is repeated for each vertex that belongs to the
 * corresponding cell.  The resulting buffer has length
 * `cellBuffer.length * vertexPerCell`.
 *
 * This is useful when you have per-cell data (e.g. color or value) that must
 * be fed to a GPU shader as per-vertex attributes.
 *
 * @example
 * // Source: 2 cells, each stored as a scalar (dimension = 1)
 * const src = new Float32Array([10, 20]);
 * // Expand to 3 vertices per cell
 * const dst = expandArrayBuffer(src, 3, 1);
 * // dst → Float32Array [10, 10, 10, 20, 20, 20]
 *
 * @template T - A typed array type (e.g. `Float32Array`, `Uint8Array`).
 * @param cellBuffer   - The source typed array.  Its logical length is
 *                       `cellBuffer.length / dimension` cells.
 * @param vertexPerCell - Number of vertices each cell expands to.
 * @param dimension     - Number of consecutive values that form a single
 *                        logical entry (e.g. 2 for 2-D coordinates, 4 for
 *                        RGBA color).
 * @returns A new typed array of the same constructor with every cell entry
 *          duplicated `vertexPerCell` times.
 */
export function expandArrayBuffer<T extends ArrayBufferView>(
  cellBuffer: T,
  vertexPerCell: number,
  dimension: number
): T {
  // @ts-expect-error - this disables type checking for the length property.
  const len = cellBuffer.length;
  const entries = len / dimension;
  if (!Number.isInteger(entries)) {
    throw new Error(
      'expandArrayBuffer: cellBuffer length must be divisible by dimension'
    );
  }
  const Ctor = cellBuffer.constructor as any as { new (n: number): T };
  const expanded = new Ctor(len * vertexPerCell);

  for (let i = 0; i < entries; i++) {
    const srcOffset = i * dimension;
    for (let v = 0; v < vertexPerCell; v++) {
      const dstOffset = (i * vertexPerCell + v) * dimension;
      for (let c = 0; c < dimension; c++) {
        (expanded as any)[dstOffset + c] = (cellBuffer as any)[srcOffset + c];
      }
    }
  }
  return expanded;
}
