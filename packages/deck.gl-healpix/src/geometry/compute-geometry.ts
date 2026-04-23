/**
 * Geometry computation pipeline for HEALPix cells.
 *
 * - **Worker pool** dispatches cell ID batches to `tile-grid.worker.ts`.
 *   Small batches (<10k cells) use a single worker; larger ones are split
 *   across up to 8 workers and merged afterward.
 *
 * - **LRU cache** (512 entries) keyed on nside + scheme + content hash
 *   avoids recomputation when panning back to a previously visible tile.
 *
 * - **In-flight dedup** ensures the same cell set isn't computed twice
 *   concurrently.
 */

import { getWorkerFactory } from '../config';
import { WorkerPool } from '../utils/worker-pool';
import { hashTypedArray } from '../utils/hash';
import type { GeometryResult, WorkerMessage, WorkerTask } from './types';
import type { CellIdArray } from '../types/cell-ids';
import type { HealpixScheme } from '../types/layer-props';

const POOL_THRESHOLD = 10_000;
const MAX_WORKERS = Math.min(navigator.hardwareConcurrency ?? 4, 8);

let geometryPool: WorkerPool<WorkerTask, WorkerMessage, GeometryResult> | null =
  null;

function getPool(): WorkerPool<WorkerTask, WorkerMessage, GeometryResult> {
  if (!geometryPool) {
    geometryPool = new WorkerPool<WorkerTask, WorkerMessage, GeometryResult>({
      createWorker: () => getWorkerFactory()(),
      maxConcurrent: MAX_WORKERS,
      getResult: (e) => (e.data.type === 'data' ? e.data.data : undefined)
    });
  }
  return geometryPool;
}

const MAX_CACHE_ENTRIES = 512;
const geometryCache = new Map<string, GeometryResult>();
const inflightGeometry = new Map<string, Promise<GeometryResult>>();

function cacheKey(
  nside: number,
  scheme: HealpixScheme,
  cellIds: CellIdArray
): string {
  return `${nside}:${scheme}:${cellIds.length}:${hashTypedArray(cellIds)}`;
}

export async function computeGeometry(
  nside: number,
  cellIds: CellIdArray,
  scheme: HealpixScheme
): Promise<GeometryResult> {
  const key = cacheKey(nside, scheme, cellIds);

  const cached = geometryCache.get(key);
  if (cached) {
    geometryCache.delete(key);
    geometryCache.set(key, cached);
    return cached;
  }

  const inflight = inflightGeometry.get(key);
  if (inflight) return inflight;

  const promise = computeUncached(nside, cellIds, scheme);
  inflightGeometry.set(key, promise);

  try {
    const result = await promise;

    if (geometryCache.size >= MAX_CACHE_ENTRIES) {
      const oldest = geometryCache.keys().next().value!;
      geometryCache.delete(oldest);
    }
    geometryCache.set(key, result);

    return result;
  } finally {
    inflightGeometry.delete(key);
  }
}

async function computeUncached(
  nside: number,
  cellIds: CellIdArray,
  scheme: HealpixScheme
): Promise<GeometryResult> {
  if (cellIds.length < POOL_THRESHOLD) {
    const [result] = await getPool().run([{ nside, cellIds, scheme }]);
    return result;
  }

  const chunkSize = Math.ceil(cellIds.length / MAX_WORKERS);
  const tasks: WorkerTask[] = [];
  for (let i = 0; i < cellIds.length; i += chunkSize) {
    tasks.push({
      nside,
      cellIds: cellIds.slice(i, Math.min(i + chunkSize, cellIds.length)),
      scheme
    });
  }

  const results = await getPool().run(tasks);

  const totalCells = cellIds.length;
  const coords = new Float32Array(totalCells * 10);
  const indexes = new Uint32Array(totalCells);
  const triangles = new Uint32Array(totalCells * 6);

  let cellOffset = 0;
  for (const r of results) {
    const count = r.indexes.length;
    const vertexBase = cellOffset * 5;

    coords.set(r.coords, cellOffset * 10);

    for (let j = 0; j < count; j++) {
      indexes[cellOffset + j] = r.indexes[j] + vertexBase;
    }
    for (let j = 0; j < r.triangles.length; j++) {
      triangles[cellOffset * 6 + j] = r.triangles[j] + vertexBase;
    }

    cellOffset += count;
  }

  return { coords, indexes, triangles };
}
