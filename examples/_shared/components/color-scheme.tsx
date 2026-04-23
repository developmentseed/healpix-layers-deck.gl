import React, { useMemo } from 'react';
import {
  Box,
  Portal,
  Select,
  SelectValueChangeDetails,
  Stack,
  createListCollection
} from '@chakra-ui/react';
import { scaleSequential } from 'd3';

import {
  schemeFns,
  type ColorSchemeName
} from '../utils/sequential-color-scale';

export { schemeFns, type ColorSchemeName };

function makeBg(scheme: keyof typeof schemeFns) {
  const scale = scaleSequential((t) => schemeFns[scheme](t)).domain([0, 1]);
  const colors = [];
  for (let i = 0; i <= 1; i += 0.01) {
    colors.push(scale(i));
  }
  return `linear-gradient(to right, ${colors.join(', ')})`;
}

interface ColorSchemeSelectProps {
  scheme: keyof typeof schemeFns;
  onSchemeChange: (scheme: keyof typeof schemeFns) => void;
}

export function ColorSchemeSelect(props: ColorSchemeSelectProps) {
  const { scheme, onSchemeChange } = props;

  const [value, setter] = useMemo(
    () => [
      [scheme],
      (e: SelectValueChangeDetails) =>
        onSchemeChange(e.value[0] as keyof typeof schemeFns)
    ],
    [scheme, onSchemeChange]
  );

  return (
    <Select.Root collection={schemes} value={value} onValueChange={setter}>
      <Select.HiddenSelect />
      <Select.Label>Color scheme</Select.Label>
      <Select.Control>
        <Select.Trigger>
          <Select.ValueText placeholder='Select color scheme' />
        </Select.Trigger>
        <Select.IndicatorGroup>
          <Select.Indicator />
        </Select.IndicatorGroup>
      </Select.Control>
      <Portal>
        <Select.Positioner>
          <Select.Content>
            {schemes.items.map((scheme) => (
              <Select.Item item={scheme} key={scheme.value}>
                <Stack gap='0' w='80%'>
                  <Select.ItemText>{scheme.label}</Select.ItemText>
                  <Box
                    height='1rem'
                    width='100%'
                    borderRadius='4px'
                    boxShadow='0 0 1px 1px rgba(255,255,255,0.48)'
                    bg={scheme.bg}
                  />
                </Stack>
                <Select.ItemIndicator />
              </Select.Item>
            ))}
          </Select.Content>
        </Select.Positioner>
      </Portal>
    </Select.Root>
  );
}

const schemes = createListCollection<{
  label: string;
  value: keyof typeof schemeFns;
  bg: string;
}>({
  items: [
    {
      label: 'Viridis',
      value: 'interpolateViridis',
      bg: makeBg('interpolateViridis')
    },
    {
      label: 'Plasma',
      value: 'interpolatePlasma',
      bg: makeBg('interpolatePlasma')
    },
    {
      label: 'Cool',
      value: 'interpolateCool',
      bg: makeBg('interpolateCool')
    },
    {
      label: 'Rainbow',
      value: 'interpolateRainbow',
      bg: makeBg('interpolateRainbow')
    },
    {
      label: 'RdYlBu (reversed)',
      value: 'interpolateRdYlBu',
      bg: makeBg('interpolateRdYlBu')
    }
  ]
});
