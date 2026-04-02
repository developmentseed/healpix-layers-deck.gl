/**
 * Typed arrays accepted as HEALPix cell index buffers.
 *
 * 64-bit element types (`Float64Array`) are supported for large indices. Values
 * are passed to `healpix-ts` as JS numbers;
 */
export type CellIdArray =
  | Int32Array
  | Uint32Array
  | Float64Array
  | Float32Array;
