/**
 * Band column order in `values` (10 columns). Must match group `bands` metadata.
 */
export const BAND_ORDER = [
  'b01',
  'b02',
  'b03',
  'b04',
  'b05',
  'b06',
  'b07',
  'b8a',
  'b11',
  'b12'
] as const;

export const NBANDS = BAND_ORDER.length;

export type BandLabel = (typeof BAND_ORDER)[number];

export const BAND_INDEX = BAND_ORDER.reduce(
  (acc, bl, index) => {
    acc[bl] = index;
    return acc;
  },
  {} as Record<BandLabel, number>
);

export const BAND_CHOICES: { value: BandLabel; label: string }[] =
  BAND_ORDER.map((bl: BandLabel) => ({
    value: bl,
    label: `Band ${bl} (single)`
  }));
