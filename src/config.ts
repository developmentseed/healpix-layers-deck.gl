import inlinedWorkerCode from 'virtual:tile-grid-worker';

type WorkerFactory = () => Worker;

let _workerFactory: WorkerFactory | null = null;

function createBlobWorker(): Worker {
  const blob = new Blob([inlinedWorkerCode], {
    type: 'application/javascript'
  });
  return new Worker(URL.createObjectURL(blob));
}

/**
 * Override the default worker with a URL pointing to a self-hosted copy
 * of the tile-grid worker script.
 *
 * @example
 * ```ts
 * import { setWorkerUrl } from 'healpix-layers-deck.gl';
 * setWorkerUrl(new URL('healpix-layers-deck.gl/worker', import.meta.url));
 * ```
 */
export function setWorkerUrl(url: string | URL): void {
  _workerFactory = () => new Worker(url);
}

/**
 * Override the default worker with a custom factory.
 * Use this when you need full control over worker instantiation
 * (e.g. module workers, custom headers, service-worker proxying).
 */
export function setWorkerFactory(factory: WorkerFactory): void {
  _workerFactory = factory;
}

export function getWorkerFactory(): WorkerFactory {
  return _workerFactory ?? createBlobWorker;
}
