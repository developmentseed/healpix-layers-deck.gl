import { useEffect, useMemo, useState } from 'react';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import { Box, Field, Flex, NativeSelect, Slider, Text } from '@chakra-ui/react';
import {
  HealpixCellsLayer,
  makeColorMap
} from '@developmentseed/deck.gl-healpix';
import { rgb } from 'd3';

import {
  ColorSchemeSelect,
  schemeFns,
  type ColorSchemeName
} from '$shared/components/color-scheme';
import { DeckGlOverlay } from '$shared/components/deckgl-overlay';

import {
  type SentinelHealpixZarr,
  buildCompositeRgb,
  buildNdvi,
  extractColumn,
  loadSentinelHealpixZarr
} from './sentinel-zarr';
import { BAND_CHOICES, BAND_INDEX, BandLabel } from './sentinel-zarr-bands';

const ZARR_URL = `${import.meta.env.VITE_BASE_URL}/sentinel-healpix.zarr`;

export type BandVisualizationMode =
  | 'true_color'
  | 'infrared_false_color'
  | 'ndvi'
  | 'swir'
  | BandLabel;

const COMPOSITE_OPTIONS: { value: BandVisualizationMode; label: string }[] = [
  { value: 'true_color', label: 'True color' },
  { value: 'infrared_false_color', label: 'Infrared false color' },
  { value: 'swir', label: 'SWIR composite' },
  { value: 'ndvi', label: 'NDVI' }
];

export const VISUALIZATION_OPTIONS: {
  value: BandVisualizationMode;
  label: string;
}[] = [...COMPOSITE_OPTIONS, ...BAND_CHOICES];

function isNdviMode(v: BandVisualizationMode): boolean {
  return v === 'ndvi';
}

function isSingleBandMode(v: BandVisualizationMode): v is BandLabel {
  return v in BAND_INDEX;
}

function defaultDisplayRange(mode: BandVisualizationMode): [number, number] {
  if (mode === 'ndvi') return [0.0, 0.6];
  if (isSingleBandMode(mode)) return [0, 0.25];
  return [0, 0.3];
}

export default function PageColor() {
  const [viewState, setViewState] = useState({
    longitude: -9.75081,
    latitude: 39.27385,
    zoom: 8.5
  });

  const [projection, setProjection] = useState<'globe' | 'mercator'>(
    'mercator'
  );

  const [zarrData, setZarrData] = useState<SentinelHealpixZarr | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadPending, setLoadPending] = useState(true);

  const [visualization, setVisualization] =
    useState<BandVisualizationMode>('true_color');
  const [indexMin, setIndexMin] = useState(0);
  const [indexMax, setIndexMax] = useState(1);
  const [colorScheme, setColorScheme] =
    useState<ColorSchemeName>('interpolateViridis');

  useEffect(() => {
    let cancelled = false;
    setLoadPending(true);
    setLoadError(null);
    void loadSentinelHealpixZarr(ZARR_URL)
      .then((d) => {
        if (!cancelled) {
          setZarrData(d);
        }
      })
      .catch((e) => {
        if (!cancelled) {
          setZarrData(null);
          setLoadError(e instanceof Error ? e.message : String(e));
        }
      })
      .finally(() => {
        if (!cancelled) setLoadPending(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    const [lo, hi] = defaultDisplayRange(visualization);
    setIndexMin(lo);
    setIndexMax(hi);
  }, [visualization]);

  const ndvi = isNdviMode(visualization);
  const singleBand = isSingleBandMode(visualization);
  const showScalarControls = ndvi || singleBand;

  const displayRangeMin = ndvi ? -1 : 0;
  const displayRangeMax = 1;

  const colorMap = useMemo(
    () =>
      makeColorMap((t) => {
        const c = rgb(schemeFns[colorScheme](t));
        return [c.r, c.g, c.b, 255];
      }),
    [colorScheme]
  );

  const layers = useMemo(() => {
    if (!zarrData) return [];
    const { nside, cellIds } = zarrData;
    if (isNdviMode(visualization)) {
      const values = buildNdvi(zarrData);
      return [
        new HealpixCellsLayer({
          id: 'healpix-zarr-ndvi',
          nside,
          scheme: 'nest',
          cellIds,
          values,
          dimensions: 1,
          min: indexMin,
          max: indexMax,
          colorMap
        })
      ];
    }
    if (isSingleBandMode(visualization)) {
      const col = BAND_INDEX[visualization];
      const values = extractColumn(zarrData, col);
      return [
        new HealpixCellsLayer({
          id: 'healpix-zarr-band',
          nside,
          scheme: 'nest',
          cellIds,
          values,
          dimensions: 1,
          min: indexMin,
          max: indexMax,
          colorMap
        })
      ];
    }
    const values = buildCompositeRgb(
      visualization as 'true_color' | 'infrared_false_color' | 'swir',
      zarrData
    );
    return [
      new HealpixCellsLayer({
        id: 'healpix-zarr-rgb',
        nside,
        scheme: 'nest',
        cellIds,
        values,
        dimensions: 3,
        min: 0,
        max: 1,
        colorMap
      })
    ];
  }, [zarrData, visualization, colorMap, indexMin, indexMax]);

  return (
    <Flex w='100%' h='100%' flexFlow='column' position='relative'>
      <Flex
        position='absolute'
        top={4}
        left={4}
        zIndex={1000}
        bg='white'
        borderRadius='md'
        boxShadow='md'
        p={4}
        minW='220px'
        flexFlow='column'
        gap={4}
        w='30rem'
        maxH='90vh'
        overflowY='auto'
      >
        <Text fontStyle='italic'>
          Sentinel 2 scene with 10 bands in healpix
          <br />
          <Text as='span' fontSize='sm'>
            Cell coloring is computed on the GPU according the the values and
            number of bands. Visualizations with 3 bands are rendered directly
            as RGB. Visualizations with 1 band are mapped to a color scheme and
            possibly rescaled.
          </Text>
        </Text>

        {loadPending && <Text fontSize='sm'>Loading Zarr…</Text>}
        {loadError && (
          <Text color='red.fg' fontSize='sm'>
            {loadError}
          </Text>
        )}

        <Field.Root>
          <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
            Projection
          </Field.Label>
          <NativeSelect.Root size='sm'>
            <NativeSelect.Field
              value={projection}
              onChange={(e) =>
                setProjection(e.currentTarget.value as 'globe' | 'mercator')
              }
            >
              <option value='globe'>Globe</option>
              <option value='mercator'>Mercator</option>
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field.Root>

        <Field.Root>
          <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
            Visualization
          </Field.Label>
          <NativeSelect.Root size='sm'>
            <NativeSelect.Field
              value={visualization}
              onChange={(e) =>
                setVisualization(e.currentTarget.value as BandVisualizationMode)
              }
            >
              {VISUALIZATION_OPTIONS.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field.Root>

        {showScalarControls && (
          <Field.Root>
            <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
              {ndvi ? 'NDVI rescale' : 'Rescale'}
            </Field.Label>
            <Text fontSize='xs' color='fg.muted' mb={2}>
              {ndvi
                ? 'Typical index roughly −1 … 1. Min/max set the display stretch.'
                : 'Adjust min/max for single-band display stretch (0…1).'}
            </Text>
            <Slider.Root
              width='100%'
              min={displayRangeMin}
              max={displayRangeMax}
              step={0.002}
              value={[indexMin, indexMax]}
              onValueChange={({ value }) => {
                const [a, b] = value;
                const lo = Math.min(a, b);
                const hi = Math.max(a, b);
                setIndexMin(lo);
                setIndexMax(hi);
              }}
            >
              <Slider.Control>
                <Slider.Track>
                  <Slider.Range />
                </Slider.Track>
                <Slider.Thumbs />
              </Slider.Control>
            </Slider.Root>
            <Flex justify='space-between' mt={1} fontSize='sm' gap={4}>
              <Text>Min: {indexMin.toFixed(3)}</Text>
              <Text>Max: {indexMax.toFixed(3)}</Text>
            </Flex>
          </Field.Root>
        )}

        {showScalarControls && (
          <ColorSchemeSelect
            scheme={colorScheme}
            onSchemeChange={setColorScheme}
          />
        )}
      </Flex>

      <Box flex='1' position='relative' minH={0}>
        <Map
          {...viewState}
          projection={projection}
          onMove={(event) => setViewState(event.viewState)}
          mapStyle={`https://api.maptiler.com/maps/aquarelle-v4/style.json?key=${import.meta.env.VITE_MAPTILER_KEY}`}
          style={{ width: '100%', height: '100%' }}
        >
          {!loadPending && !loadError && zarrData && (
            <DeckGlOverlay layers={layers} />
          )}
        </Map>
      </Box>
    </Flex>
  );
}
