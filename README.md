![HEALPix Banner](./docs/healpix-banner.png)

<p align="center">
  <a href="https://github.com/developmentseed/healpix-ts">HEALPix Typescript</a> | <a href="https://github.com/developmentseed/healpix-layers-deck.gl">HEALPix deck.gl</a>
</p>

This repository is a **monorepo** for HEALPix-related [deck.gl](https://deck.gl/) libraries. **npm workspaces** and [**Lerna**](https://lerna.js.org/) orchestrate builds, tests, and releases: run the usual scripts from the **repository root** unless you are working inside a single package.

https://github.com/user-attachments/assets/4166d5d5-65e3-4309-a63a-0a2d0cdf275d

## Packages

| Package | Description |
| -------- | ----------- |
| [`@developmentseed/deck.gl-healpix`](./packages/deck.gl-healpix/) | deck.gl layer for rendering [HEALPix](https://healpix.sourceforge.io/) cells on a map, with GPU-side colormaps and multi-frame animation. |

Each package has its own **README** with installation, API usage, and examples—start there for day-to-day integration work.

## Development (root)

```bash
npm install
npm run build         # lerna: build all packages
npm run build:watch   # lerna: watch mode
npm run ts-check      # tsc --noEmit per package
npm run test
npm run lint
npm run clean
```

Versioning and publishing (maintainers):

```bash
npm run versionup      # bump versions; no git tag / push
npm run publish:from   # lerna publish from-package (skips private workspaces)
```

## License

**MIT** — see [LICENSE](./LICENSE).
