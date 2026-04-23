/**
 * FNV-1a hash of  typed array. Used to build geometry cache keys so that
 * two calls with the same cell IDs (same content, different reference) hit
 * the same cache entry.
 *
 * @param arr - The typed array to hash.
 * @returns 32-bit unsigned hash value.
 */
export function hashTypedArray(
  arr: Float32Array | Float64Array | Uint32Array | Int32Array
): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < arr.length; i++) {
    h ^= arr[i];
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
