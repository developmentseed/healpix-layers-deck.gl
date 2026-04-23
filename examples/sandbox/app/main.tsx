import { useEffect } from 'react';
import { createRoot } from 'react-dom/client';
import { Route, Routes } from 'react-router';
import { Flex } from '@chakra-ui/react';

import PageAnimation from '$pages/animation';
import PageColor from '$pages/color';
import { PageLayout, PageNavLink } from '$shared/components/page-layout';

// Root component.
function Root() {
  useEffect(() => {
    dispatchEvent(new Event('app-ready'));
  }, []);

  return (
    <PageLayout
      title='HEALPix Sandbox'
      navSlot={
        <Flex gap={2} flexWrap='wrap' justifyContent='flex-end'>
          <PageNavLink to='/'>Cell Rendering</PageNavLink>
          <PageNavLink to='/color'>Color Visualization</PageNavLink>
        </Flex>
      }
    >
      <Routes>
        <Route path='/' element={<PageAnimation />} />
        <Route path='/color' element={<PageColor />} />
      </Routes>
    </PageLayout>
  );
}

const rootNode = document.querySelector('#app-container')!;
const root = createRoot(rootNode);
root.render(<Root />);
