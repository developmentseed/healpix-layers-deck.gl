![HEALPix Banner](./docs/healpix-banner.png)

<p align='center'>
  <a href='https://github.com/developmentseed/healpix-ts'>HEALPix Typescript</a> | <a href='https://github.com/developmentseed/healpix-layers-deck.gl'>HEALPix Deck.gl Layer</a> 
</p>


# HEALPix Deck.gl Layer

A [deck.gl](https://deck.gl/) layer for rendering [HEALPix](https://healpix.sourceforge.io/) (Hierarchical Equal Area isoLatitude Pixelization) cells on a map.  
It is especially suited for animating a large number of cells: per-cell values are uploaded once to the GPU and a small colorMap lookup is applied every frame on the vertex shader.

https://github.com/user-attachments/assets/4166d5d5-65e3-4309-a63a-0a2d0cdf275d

## Installation

```bash
npm install healpix-layers-deck.gl
```

Peer dependencies (`@deck.gl/core`, `@deck.gl/layers`) must be provided by the host application.

```bash
npm install @deck.gl/core @deck.gl/layers
```

## Usage

### Single frame

Pass per-cell numeric `values` plus a `[min, max]` range. The layer normalizes each value and maps it through a 256-entry `colorMap` LUT on the GPU. If `colorMap` is omitted a linear black-to-white ramp is used.

```ts
import { HealpixCellsLayer } from 'healpix-layers-deck.gl';

const cellIds = new Uint32Array([0, 1, 2, 3]);
const values = new Float32Array([0.1, 0.4, 0.7, 1.0]);

const layer = new HealpixCellsLayer({
  id: 'healpix',
  nside: 64,
  cellIds,
  values,
  min: 0,
  max: 1
});
```

### Multi-frame animation

Provide a `frames` array whose entries override the root-level defaults. Advance `currentFrame` to switch between them — no GPU re-upload happens unless the underlying typed array changes.

```ts
import { HealpixCellsLayer } from 'healpix-layers-deck.gl';

const cellIds = new Uint32Array([0, 1, 2, 3]);

const layer = new HealpixCellsLayer({
  id: 'healpix',
  nside: 64,
  cellIds,
  min: 0,
  max: 1,
  frames: [
    { values: new Float32Array([0.0, 0.25, 0.5, 0.75]) },
    { values: new Float32Array([1.0, 0.75, 0.5, 0.25]) }
  ],
  currentFrame: 0
});
```

Each frame may override any root-level field (`nside`, `scheme`, `cellIds`, `values`, `min`, `max`, `dimensions`, `colorMap`). Fields omitted on a frame fall back to the root value.

### Direct RGB / RGBA values

Skip the colorMap and push color directly to the GPU by setting `dimensions` to `3` or `4`. Values are interpreted as normalized channels (`0.0`–`1.0`) interleaved per cell.

```ts
const layer = new HealpixCellsLayer({
  id: 'healpix',
  nside: 64,
  cellIds: new Uint32Array([0, 1]),
  dimensions: 3,
  // cell 0 → red, cell 1 → green
  values: new Float32Array([1, 0, 0, 0, 1, 0])
});
```

### Custom colorMap

A `colorMap` is a `Uint8Array` of exactly **256 × 4 = 1024 bytes** in RGBA order. Index `0` maps to `min`, index `255` to `max`.

The `makeColorMap` helper builds one from a callback that is invoked 256 times with the normalized position `t = i / 255` and the raw byte index `i`. Return a hex string, a `[r, g, b]`/`[r, g, b, a]` tuple in `0`–`255`, or `{ normalized: true, rgba: [...] }` in `0`–`1`.

```ts
import { HealpixCellsLayer, makeColorMap } from 'healpix-layers-deck.gl';

// Red → blue gradient
const colorMap = makeColorMap((t) => ({
  normalized: true,
  rgba: [1 - t, 0, t]
}));

// Three-stop hex ramp
const stepped = makeColorMap((_, i) =>
  i < 85 ? '#f00' : i < 170 ? '#0f0' : '#00f'
);

new HealpixCellsLayer({ /* ... */, colorMap });
```

You can also build the buffer yourself — the layer will accept any `Uint8Array` that is exactly `1024` bytes long.

## API

### `HealpixCellsLayer`

A `CompositeLayer` that renders HEALPix cells as filled polygons whose colors are computed on the GPU from per-cell float `values`.

| Prop           | Type                    | Default       | Description                                                                                        |
| -------------- | ----------------------- | ------------- | -------------------------------------------------------------------------------------------------- |
| `nside`        | `number`                | `0`           | HEALPix resolution parameter (power of 2). Required on the layer or on every frame.  |
| `cellIds`      | `CellIdArray`           | `Uint32Array(0)` | HEALPix cell indices to render. Required on the layer or on every frame.                        |
| `scheme`       | `'nest' \| 'ring'`      | `'nest'`      | Pixel numbering scheme.                                                                            |
| `values`       | `ArrayLike<number>`     | —             | Interleaved per-cell float values. Length = `cellIds.length × dimensions`. Required when `frames` is absent. |
| `min`          | `number`                | `0`           | Value mapped to colorMap index 0.                                                                  |
| `max`          | `number`                | `1`           | Value mapped to colorMap index 255.                                                                |
| `dimensions`   | `1 \| 2 \| 3 \| 4`      | `1`           | Number of values per cell. See table below.                                                        |
| `colorMap`     | `Uint8Array` (1024 B)   | black → white | 256-entry RGBA LUT used when `dimensions` is `1` or `2`.                                           |
| `frames`       | `HealpixFrameObject[]`  | —             | Optional animation frames; each may override any root field.                                       |
| `currentFrame` | `number`                | `0`           | Active index into `frames`. Clamped to `[0, frames.length - 1]`.                                   |

### `dimensions` modes

| `dimensions` | Interpretation                                                                 |
| ------------ | ------------------------------------------------------------------------------ |
| `1`          | Scalar → normalized through `[min, max]` → colorMap LUT → RGBA                 |
| `2`          | Scalar (→ colorMap) + opacity multiplier (`0`–`1`) in the second value         |
| `3`          | Direct RGB in `0`–`1`; `colorMap` / `min` / `max` ignored; alpha = `1`         |
| `4`          | Direct RGBA in `0`–`1`; `colorMap` / `min` / `max` ignored                     |

`values` is always an interleaved flat array: cell `i` occupies indices `i * dimensions` through `i * dimensions + dimensions - 1`.

### `HealpixFrameObject`

Every field is optional and falls back to the matching root-level prop. `values` is the only field that must be set somewhere (root or frame).

```ts
type HealpixFrameObject = {
  nside?: number;
  scheme?: 'nest' | 'ring';
  cellIds?: CellIdArray;
  values?: ArrayLike<number>;
  min?: number;
  max?: number;
  dimensions?: 1 | 2 | 3 | 4;
  colorMap?: Uint8Array;
};
```

### Geometry worker

Cell polygon geometry is computed on the CPU in a Web Worker pool. If the default worker loader does not work for your bundler you can supply a custom factory:

```ts
import { setWorkerFactory, setWorkerUrl } from 'healpix-layers-deck.gl';

// Provide an explicit worker URL …
setWorkerUrl(new URL('healpix-layers-deck.gl/worker', import.meta.url));

// … or supply a custom factory that returns a ready-to-use Worker instance.
setWorkerFactory(() => new Worker(/* ... */));
```

### `makeColorMap(getColor)`

Build a 256-entry RGBA colorMap (1024 bytes) from a callback.

```ts
import { makeColorMap } from 'healpix-layers-deck.gl';

const viridisLike = makeColorMap((t) => ({
  normalized: true,
  rgba: [t * t, Math.sqrt(t), 1 - t]
}));
```

The callback receives `(t: number, index: number)` where `t = index / 255` in `[0, 1]`. Return one of:

- a CSS hex string — `#RGB`, `#RGBA`, `#RRGGBB`, `#RRGGBBAA`
- a 3- or 4-tuple of bytes in `0`–`255` (alpha defaults to `255`)
- `{ normalized: true, rgba: [r, g, b] | [r, g, b, a] }` with channels in `0`–`1`

Values outside their valid range are clamped.

### Types

```ts
import type {
  HealpixCellsLayerProps,
  HealpixFrameObject,
  HealpixScheme,
  CellIdArray,
  ColorMapCallbackValue
} from 'healpix-layers-deck.gl';
```

- **`HealpixScheme`** — `'nest' | 'ring'`
- **`CellIdArray`** — `Int32Array | Uint32Array | Float32Array | Float64Array`
- **`HealpixCellsLayerProps`** — Full prop type for the layer.
- **`HealpixFrameObject`** — One animation frame; see above.
- **`ColorMapCallbackValue`** — Return type accepted by the `makeColorMap` callback.

## Development

```bash
npm install
npm run build         # one-shot build (CJS + ESM + types)
npm run build:watch   # watch mode
npm run lint          # ESLint
npm test              # Jest
```

## License

MIT — see [LICENSE](LICENSE).
