/**
 * Zarr layout: `values` float32 [npix, NBANDS] (row-major), `cell_id` int64 [npix].
 * Column order matches `attributes.bands` in the group zarr.json (see `BAND_ORDER`).
 */
import * as zarr from 'zarrita';
import { BAND_INDEX, NBANDS } from './sentinel-zarr-bands';

/** Fixed HEALPix resolution for this dataset. */
export const NSIDE = 32768;

export type SentinelHealpixZarr = {
  nside: number;
  npix: number;
  nbands: number;
  cellIds: Float64Array;
  valuesFlat: Float32Array;
};

export async function loadSentinelHealpixZarr(
  baseUrl: string
): Promise<SentinelHealpixZarr> {
  const store = new zarr.FetchStore(baseUrl);
  const group = await zarr.open.v3(zarr.root(store));

  const valuesNode = await zarr.open(group.resolve('values'), {
    kind: 'array'
  });
  const cellNode = await zarr.open(group.resolve('cell_id'), { kind: 'array' });

  const valuesChunk = await zarr.get(valuesNode);
  const cellChunk = await zarr.get(cellNode);
  const ids64 = cellChunk.data as BigInt64Array;

  const cellIds = new Float64Array(ids64.length);
  for (let i = 0; i < ids64.length; i++) {
    cellIds[i] = Number(ids64[i]);
  }

  return {
    nside: NSIDE,
    npix: valuesChunk.data.length / NBANDS,
    nbands: NBANDS,
    cellIds,
    valuesFlat: new Float32Array(valuesChunk.data as Float32Array)
  };
}

export function extractColumn(
  z: SentinelHealpixZarr,
  col: number
): Float32Array {
  const { npix, nbands, valuesFlat } = z;
  if (col < 0 || col >= nbands) {
    throw new Error(`Band column out of range: ${col}`);
  }
  const out = new Float32Array(npix);
  for (let i = 0; i < npix; i++) {
    out[i] = valuesFlat[i * nbands + col]!;
  }
  return out;
}

const RGB_NUM = 0.3;

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

function stretchRgb(x: number): number {
  return clamp01(x / RGB_NUM);
}

const COMPOSITE_COLS: Record<
  'true_color' | 'infrared_false_color' | 'swir',
  { r: number; g: number; b: number }
> = {
  true_color: { r: BAND_INDEX.b04, g: BAND_INDEX.b03, b: BAND_INDEX.b02 },
  infrared_false_color: {
    r: BAND_INDEX.b8a,
    g: BAND_INDEX.b04,
    b: BAND_INDEX.b03
  },
  swir: { r: BAND_INDEX.b12, g: BAND_INDEX.b8a, b: BAND_INDEX.b04 }
};

export function buildCompositeRgb(
  mode: 'true_color' | 'infrared_false_color' | 'swir',
  z: SentinelHealpixZarr
): Float32Array {
  const n = z.npix;
  const { nbands, valuesFlat } = z;
  const { r: rc, g: gc, b: bc } = COMPOSITE_COLS[mode];
  const out = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const base = i * nbands;
    out[i * 3] = stretchRgb(valuesFlat[base + rc]!);
    out[i * 3 + 1] = stretchRgb(valuesFlat[base + gc]!);
    out[i * 3 + 2] = stretchRgb(valuesFlat[base + bc]!);
  }
  return out;
}

export function buildNdvi(data: SentinelHealpixZarr): Float32Array {
  const { nbands, valuesFlat, npix } = data;
  const i4 = BAND_INDEX.b04;
  const i8a = BAND_INDEX.b8a;
  const out = new Float32Array(npix);
  const eps = 1e-6;
  for (let i = 0; i < npix; i++) {
    const base = i * nbands;
    const b4 = valuesFlat[base + i4]!;
    const b8a = valuesFlat[base + i8a]!;
    const s = b8a + b4;
    out[i] = s < eps ? 0 : (b8a - b4) / (s + eps);
  }
  return out;
}
