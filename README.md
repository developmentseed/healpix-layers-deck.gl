# healpix-layers-deck.gl

A [deck.gl](https://deck.gl/) layer for rendering [HEALPix](https://healpix.jpl.nasa.gov/) (Hierarchical Equal Area isoLatitude Pixelization) cells on a map.

## Installation

```bash
npm install healpix-layers-deck.gl
```

Peer dependencies (`@deck.gl/core`, `@deck.gl/layers`) must be provided by the host application.

## Usage

```ts
import { HealpixCellsLayer } from 'healpix-layers-deck.gl';

const cellIds = new Int32Array([0, 1, 2, 3]);

// Each frame is one RGBA color per cell in uint8 format (0-255).
const frame0 = new Uint8Array([
  255, 0, 0, 255,   // cell 0
  0, 255, 0, 255,   // cell 1
  0, 0, 255, 255,   // cell 2
  255, 255, 0, 255  // cell 3
]);

const frame1 = new Uint8Array([
  255, 255, 255, 255,
  255, 128, 0, 255,
  128, 0, 255, 255,
  0, 255, 255, 255
]);

const layer = new HealpixCellsLayer({
  id: 'healpix',
  nside: 64,
  cellIds,
  scheme: 'nest',
  colorFrames: [frame0, frame1],
  currentFrame: 0
});
```

### Animation Model

`HealpixCellsLayer` uploads all provided frames into a single GPU texture:

- texture **width** = `cellIds.length`
- texture **height** = `colorFrames.length`
- each texel = one `RGBA` color for one `(cell, frame)` pair

At render time, the layer only changes `currentFrame`, and the shader samples
the corresponding row from the texture.

## API

### `HealpixCellsLayer`

A `CompositeLayer` that renders HEALPix cells as filled polygons.

| Prop | Type | Default | Description |
| --- | --- | --- | --- |
| `nside` | `number` | `0` | HEALPix resolution parameter (must be a power of 2). |
| `cellIds` | `Int32Array` | `Int32Array(0)` | HEALPix cell indices to render. |
| `scheme` | `'nest' \| 'ring'` | `'nest'` | Pixel numbering scheme. |
| `colorFrames` | `Uint8Array[]` | `[]` | Color animation frames. Each frame must be `cellIds.length * 4` in RGBA byte order (`0-255`). |
| `currentFrame` | `number` | `0` | Frame index to render. Values are clamped into valid range. |

All standard deck.gl `CompositeLayer` props (e.g. `visible`, `opacity`, `pickable`) are also accepted.

### Types

```ts
import type { HealpixCellsLayerProps, HealpixScheme } from 'healpix-layers-deck.gl';
```

- **`HealpixScheme`** — `'nest' | 'ring'`
- **`HealpixCellsLayerProps`** — Full prop type for the layer.

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
