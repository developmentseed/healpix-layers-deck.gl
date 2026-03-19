import { readFileSync } from 'fs';
import path from 'path';
import nodeResolve from '@rollup/plugin-node-resolve';
import peerDepsExternal from 'rollup-plugin-peer-deps-external';
import replace from '@rollup/plugin-replace';
import typescript from '@rollup/plugin-typescript';
import { dts } from 'rollup-plugin-dts';

const dirname = process.cwd();
const pkg = JSON.parse(readFileSync(path.join(dirname, './package.json')));

const env = process.env.NODE_ENV || 'development';

/**
 * Rollup virtual-module plugin that exposes the pre-built worker IIFE as a
 * string constant so it can be turned into a Blob URL at runtime.
 * Must run AFTER the worker entry has been emitted (first config in the array).
 */
function inlineWorkerPlugin() {
  const VIRTUAL_ID = 'virtual:tile-grid-worker';
  const RESOLVED_ID = '\0' + VIRTUAL_ID;

  return {
    name: 'inline-worker',
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_ID;
    },
    load(id) {
      if (id === RESOLVED_ID) {
        const code = readFileSync(
          path.join(dirname, 'dist/tile-grid.worker.js'),
          'utf-8'
        );
        return `export default ${JSON.stringify(code)};`;
      }
    }
  };
}

export default [
  // 1. Self-contained worker (IIFE, all deps bundled)
  {
    input: path.join(dirname, './src/workers/tile-grid.worker.ts'),
    plugins: [
      nodeResolve(),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: false,
        outDir: './dist'
      })
    ],
    output: {
      file: path.join(dirname, 'dist/tile-grid.worker.js'),
      format: 'iife',
      sourcemap: true
    }
  },
  // 2. ESM and CJS (inlines worker code from step 1)
  {
    input: path.join(dirname, './src/index.ts'),
    plugins: [
      inlineWorkerPlugin(),
      peerDepsExternal(),
      replace({
        preventAssignment: true,
        'process.env.NODE_ENV': JSON.stringify(env),
        'process.env.PACKAGE_VERSION': JSON.stringify(pkg.version)
      }),
      typescript({
        tsconfig: './tsconfig.json',
        declaration: true,
        declarationDir: './dist/types',
        outDir: './dist'
      })
    ],
    output: [
      { file: pkg.main, format: 'cjs', sourcemap: true, exports: 'named' },
      { file: pkg.module, format: 'es', sourcemap: true }
    ]
  },
  // 3. Bundled type declarations
  {
    input: path.join(dirname, 'dist/types/index.d.ts'),
    output: [{ file: path.join(dirname, 'dist/index.d.ts'), format: 'es' }],
    plugins: [dts()]
  }
];
