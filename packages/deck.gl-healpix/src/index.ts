export { HealpixCellsLayer } from './layers/healpix-cells-layer';
export { setWorkerUrl, setWorkerFactory } from './config';
export { makeColorMap } from './utils/color-map';
export type {
  ColorMapCallbackValue,
  NormalizedColorArray,
  Uint8ColorArray
} from './utils/color-map';
export type { CellIdArray } from './types/cell-ids';
export type {
  HealpixCellsLayerProps,
  HealpixFrameObject,
  HealpixScheme
} from './types/layer-props';
