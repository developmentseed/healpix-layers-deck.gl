/**
 * Validates that `colorMap` is exactly 256 × 4 = 1024 bytes.
 * Throws with a descriptive message if not.
 */
export function validateColorMap(colorMap: Uint8Array): void {
  if (colorMap.length !== 1024) {
    throw new Error(
      `HealpixCellsLayer: colorMap must be exactly 256 × 4 = 1024 bytes, got ${colorMap.length}`
    );
  }
}

// The following types are the same, but used for clarity in the docs.
/** RGB or RGBA channels in 0–255. */
export type Uint8ColorArray =
  | readonly [number, number, number]
  | readonly [number, number, number, number];

/** RGB or RGBA channels in 0–1. */
export type NormalizedColorArray =
  | readonly [number, number, number]
  | readonly [number, number, number, number];

/**
 * A value returned from the `makeColorMap` callback.
 *
 * Can be:
 *  - A CSS-like hex string (`#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`).
 *  - A 3- or 4-tuple of bytes (`0`–`255`).
 *  - `{ normalized: true, rgba: [...] }` with channels in `0`–`1`.
 */
export type ColorMapCallbackValue =
  | string
  | Uint8ColorArray
  | { readonly normalized: true; readonly rgba: NormalizedColorArray };

function clampByte(n: number): number {
  return Math.max(0, Math.min(255, Math.round(n)));
}

function clampUnit(n: number): number {
  return Math.max(0, Math.min(1, n));
}

/**
 * Parse a hex color string into RGBA bytes (0–255).
 * Supports `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA` (leading `#` optional).
 */
function parseHexColorToRgba255(hex: string): [number, number, number, number] {
  let s = hex.trim();
  if (s.startsWith('#')) s = s.slice(1);

  if (s.length === 3 || s.length === 4) {
    const expand = (i: number) =>
      parseInt(s.slice(i, i + 1) + s.slice(i, i + 1), 16);
    const r = expand(0);
    const g = expand(1);
    const b = expand(2);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
      throw new Error(`Invalid hex color: ${hex}`);
    }
    if (s.length === 3) return [r, g, b, 255];
    const a = expand(3);
    if (Number.isNaN(a)) throw new Error(`Invalid hex color: ${hex}`);
    return [r, g, b, a];
  }

  if (s.length === 6 || s.length === 8) {
    const r = parseInt(s.slice(0, 2), 16);
    const g = parseInt(s.slice(2, 4), 16);
    const b = parseInt(s.slice(4, 6), 16);
    if (Number.isNaN(r) || Number.isNaN(g) || Number.isNaN(b)) {
      throw new Error(`Invalid hex color: ${hex}`);
    }
    if (s.length === 6) return [r, g, b, 255];
    const a = parseInt(s.slice(6, 8), 16);
    if (Number.isNaN(a)) throw new Error(`Invalid hex color: ${hex}`);
    return [r, g, b, a];
  }

  throw new Error(`Invalid hex color: ${hex}`);
}

/** Normalize any supported callback color value to RGBA in 0–255. */
function normalizeCallbackValue(
  value: ColorMapCallbackValue
): [number, number, number, number] {
  if (typeof value === 'string') {
    return parseHexColorToRgba255(value);
  }

  if (Array.isArray(value)) {
    if (value.length === 3) {
      return [
        clampByte(value[0]),
        clampByte(value[1]),
        clampByte(value[2]),
        255
      ];
    }
    if (value.length === 4) {
      return [
        clampByte(value[0]),
        clampByte(value[1]),
        clampByte(value[2]),
        clampByte(value[3])
      ];
    }
    throw new Error(
      `Color array must have 3 or 4 channels; got length ${value.length}`
    );
  }

  if (
    value &&
    typeof value === 'object' &&
    'normalized' in value &&
    value.normalized === true
  ) {
    const c = value.rgba as readonly number[];
    const n = c.length;
    if (n === 3) {
      return [
        clampByte(clampUnit(c[0]) * 255),
        clampByte(clampUnit(c[1]) * 255),
        clampByte(clampUnit(c[2]) * 255),
        255
      ];
    }
    if (n === 4) {
      return [
        clampByte(clampUnit(c[0]) * 255),
        clampByte(clampUnit(c[1]) * 255),
        clampByte(clampUnit(c[2]) * 255),
        clampByte(clampUnit(c[3]) * 255)
      ];
    }
    throw new Error(
      `Normalized rgba must have 3 or 4 channels; got length ${n}`
    );
  }

  throw new Error('Invalid colorMap callback value');
}

/**
 * Build a 256-entry RGBA colorMap LUT suitable for `HealpixCellsLayer.colorMap`.
 *
 * The callback is invoked for each of the 256 slots with the normalized
 * position `t = i / 255` in `[0, 1]` and the raw byte index `i` in
 * `[0, 255]`. Return one of:
 *   - a CSS hex string (`#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`)
 *   - a 3- or 4-tuple of bytes in `0`–`255`
 *   - `{ normalized: true, rgba: [r, g, b] | [r, g, b, a] }` with channels in `0`–`1`
 *
 * Alpha defaults to `255` when a 3-channel form is used. The resulting
 * buffer is exactly `256 × 4 = 1024` bytes long.
 *
 * @example
 * ```ts
 * // Red → blue gradient
 * const map = makeColorMap((t) => ({
 *   normalized: true,
 *   rgba: [1 - t, 0, t]
 * }));
 *
 * // Three-stop hex ramp
 * const steps = makeColorMap((_, i) => (i < 85 ? '#f00' : i < 170 ? '#0f0' : '#00f'));
 * ```
 */
export function makeColorMap(
  getColor: (t: number, index: number) => ColorMapCallbackValue
): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const color = normalizeCallbackValue(getColor(t, i));
    out.set(color, i * 4);
  }
  return out;
}

/**
 * Default colorMap: linear black (0,0,0,255) → white (255,255,255,255) gradient.
 * 256 entries × 4 bytes (RGBA) = 1024 bytes.
 */
export const DEFAULT_COLORMAP: Uint8Array = makeColorMap((_, i) => [i, i, i]);
