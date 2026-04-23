import { useEffect, useMemo, useRef, useState } from 'react';
import Map from 'react-map-gl/maplibre';
import 'maplibre-gl/dist/maplibre-gl.css';
import {
  Box,
  Button,
  Field,
  Flex,
  NativeSelect,
  Slider,
  Text
} from '@chakra-ui/react';
import {
  HealpixCellsLayer,
  HealpixScheme,
  makeColorMap
} from '@developmentseed/deck.gl-healpix';

import { DeckGlOverlay } from '$shared/components/deckgl-overlay.tsx';
import { ColorSchemeSelect, schemeFns } from '$shared/components/color-scheme';
import { rgb } from 'd3';

// Powers of 2 from 8 to 128 for the nside slider steps.
const NSIDE_VALUES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024];

/** Number of animation frames for a complete sweep through all cell IDs. */
const FRAMES_PER_CYCLE = 60;

export default function PageAnimation() {
  const [viewState, setViewState] = useState({
    latitude: 0,
    longitude: 0,
    zoom: 1
  });

  const [projection, setProjection] = useState<'globe' | 'mercator'>('globe');
  const [colorScheme, setColorScheme] =
    useState<keyof typeof schemeFns>('interpolateViridis');
  const [scheme, setScheme] = useState<HealpixScheme>('ring');
  // Slider value is the index into NSIDE_VALUES.
  const [nsideIdx, setNsideIdx] = useState(1);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [fps, setFps] = useState(0);

  const animationFrameRef = useRef<number | null>(null);
  const lastFrameTimeRef = useRef<number | null>(null);
  const fpsWindowStartRef = useRef<number | null>(null);
  const framesInWindowRef = useRef(0);

  const nside = NSIDE_VALUES[nsideIdx];
  const numCells = 12 * nside * nside;
  const half = Math.floor(numCells / 2);

  // Animated cell IDs — a sliding window of half the sphere's cells.
  // The start index advances by (numCells / FRAMES_PER_CYCLE) each frame so
  // one full cycle visits every starting position.
  const animatedCellIds = useMemo(() => {
    const step = Math.max(1, Math.floor(numCells / FRAMES_PER_CYCLE));
    const start = (currentFrame * step) % numCells;
    const ids = new Int32Array(half);
    for (let i = 0; i < half; i++) {
      ids[i] = (start + i) % numCells;
    }
    return ids;
  }, [nside, currentFrame]);

  // Reset frame counter when nside changes.
  useEffect(() => {
    setCurrentFrame(0);
  }, [nside]);

  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
        animationFrameRef.current = null;
      }
      lastFrameTimeRef.current = null;
      fpsWindowStartRef.current = null;
      framesInWindowRef.current = 0;
      setFps(0);
      return;
    }

    const targetFps = 40;
    const frameDurationMs = 1000 / targetFps;

    const tick = (timestamp: number) => {
      if (lastFrameTimeRef.current === null) {
        lastFrameTimeRef.current = timestamp;
      }
      if (fpsWindowStartRef.current === null) {
        fpsWindowStartRef.current = timestamp;
      }

      const elapsedSinceLastFrame = timestamp - lastFrameTimeRef.current;
      if (elapsedSinceLastFrame >= frameDurationMs) {
        const stepCount = Math.floor(elapsedSinceLastFrame / frameDurationMs);
        lastFrameTimeRef.current += stepCount * frameDurationMs;

        setCurrentFrame((prev) => prev + stepCount);
        framesInWindowRef.current += stepCount;
      }

      const fpsWindowElapsed = timestamp - fpsWindowStartRef.current;
      if (fpsWindowElapsed >= 500) {
        setFps((framesInWindowRef.current * 1000) / fpsWindowElapsed);
        framesInWindowRef.current = 0;
        fpsWindowStartRef.current = timestamp;
      }

      animationFrameRef.current = requestAnimationFrame(tick);
    };

    animationFrameRef.current = requestAnimationFrame(tick);

    return () => {
      if (animationFrameRef.current !== null) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      animationFrameRef.current = null;
    };
  }, [isPlaying]);

  const layers = useMemo(() => {
    const colorMap = makeColorMap((t) => {
      const c = rgb(schemeFns[colorScheme](t));
      return [c.r, c.g, c.b, 255];
    });

    return [
      new HealpixCellsLayer({
        id: 'healpix-cells',
        cellIds: animatedCellIds,
        nside,
        scheme,
        values: new Float32Array(animatedCellIds),
        min: 0,
        max: nside * nside * 12,
        dimensions: 1,
        colorMap
      })
    ];
  }, [nside, scheme, animatedCellIds, colorScheme]);

  return (
    <Flex w='100%' h='100%' direction='column' position='relative'>
      {/* ── Controls panel (top-left) ── */}
      <Flex
        position='absolute'
        top={4}
        left={4}
        zIndex={1000}
        bg='white'
        borderRadius='md'
        boxShadow='md'
        p={4}
        w='30rem'
        flexFlow='column'
        gap={4}
      >
        <Text fontStyle='italic'>
          HEALPix cells dynamically generated
          <br />
          <Text as='span' fontSize='sm'>
            Half of the total cells for the chosen nside are generated and
            animated, shifting them over the different frames.
          </Text>
        </Text>
        {/* Projection dropdown */}
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

        {/* Nside slider */}
        <Field.Root>
          <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
            Nside: {nside}
          </Field.Label>
          <Slider.Root
            w='100%'
            min={0}
            max={NSIDE_VALUES.length - 1}
            step={1}
            value={[nsideIdx]}
            onValueChange={(details) => {
              const newIdx = details.value[0];
              setNsideIdx(newIdx);
            }}
          >
            <Slider.Control>
              <Slider.Track>
                <Slider.Range />
              </Slider.Track>
              <Slider.Thumb index={0} />
            </Slider.Control>
          </Slider.Root>
        </Field.Root>

        {/* Numbering scheme dropdown */}
        <Field.Root>
          <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
            Numbering scheme
          </Field.Label>
          <NativeSelect.Root size='sm'>
            <NativeSelect.Field
              value={scheme}
              onChange={(e) =>
                setScheme(e.currentTarget.value as HealpixScheme)
              }
            >
              <option value='nest'>Nest</option>
              <option value='ring'>Ring</option>
            </NativeSelect.Field>
            <NativeSelect.Indicator />
          </NativeSelect.Root>
        </Field.Root>

        {/* Color scheme selector */}
        <ColorSchemeSelect
          scheme={colorScheme}
          onSchemeChange={setColorScheme}
        />

        <Field.Root>
          <Field.Label fontSize='sm' fontWeight='semibold' mb={1}>
            Animation
          </Field.Label>
          <Flex alignItems='center' gap={3}>
            <Button size='sm' onClick={() => setIsPlaying((prev) => !prev)}>
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Text fontSize='sm'>Cells: {half.toLocaleString()}</Text>
            <Text fontSize='sm'>Frame: {currentFrame}</Text>
            <Text fontSize='sm'>FPS: {fps.toFixed(1)}</Text>
          </Flex>
        </Field.Root>
      </Flex>
      {/* Map area */}
      <Box flex='1' position='relative'>
        <Map
          {...viewState}
          projection={projection}
          onMove={(event) => setViewState(event.viewState)}
          mapStyle={`https://api.maptiler.com/maps/satellite-v4/style.json?key=${import.meta.env.VITE_MAPTILER_KEY}`}
          style={{ width: '100%', height: '100%' }}
        >
          <DeckGlOverlay layers={layers} />
        </Map>
      </Box>
    </Flex>
  );
}
