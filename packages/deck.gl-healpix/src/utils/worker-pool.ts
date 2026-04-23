const DEFAULT_MAX_CONCURRENT = 10;

// ────────────────────────────────────────────────────────────
// Public types
// ────────────────────────────────────────────────────────────

export interface WorkerPoolConfig<TMessage, TResult> {
  /**
   * Factory that creates a new `Worker` instance.
   *
   * @example
   * ```ts
   * createWorker: () =>
   *   new Worker(new URL('./my.worker.ts', import.meta.url), { type: 'module' })
   * ```
   */
  createWorker: () => Worker;

  /**
   * Maximum number of workers running concurrently.
   * @default 10
   */
  maxConcurrent?: number;

  /**
   * Extract a result from a worker `MessageEvent`.
   * Return `undefined` to ignore the message (e.g. progress updates).
   * When a non-`undefined` value is returned the task is considered complete.
   *
   * @default `(e) => e.data` — treats the entire `event.data` as the result.
   */
  getResult?: (
    event: MessageEvent<TMessage>,
    taskIndex: number
  ) => TResult | undefined;
}

// ────────────────────────────────────────────────────────────
// Implementation
// ────────────────────────────────────────────────────────────

/**
 * A fixed-size pool of Web Workers that executes tasks concurrently.
 *
 * The pool is configured once with a worker factory, concurrency limit,
 * and an optional result-extraction callback. Call {@link run} to dispatch
 * a batch of tasks — workers are reused across tasks and results are
 * returned **in the same order** as the input array.
 *
 * @example
 * ```ts
 * const pool = new WorkerPool({
 *   createWorker: () =>
 *     new Worker(new URL('./my.worker.ts', import.meta.url), { type: 'module' }),
 *   maxConcurrent: 8,
 *   getResult: (e) => (e.data.type === 'done' ? e.data.payload : undefined),
 * });
 *
 * const results = await pool.run(tasks, signal);
 * ```
 */
export class WorkerPool<TTask, TMessage, TResult> {
  private readonly createWorker: () => Worker;
  private readonly maxConcurrent: number;
  private readonly getResult: (
    event: MessageEvent<TMessage>,
    taskIndex: number
  ) => TResult | undefined;

  constructor(config: WorkerPoolConfig<TMessage, TResult>) {
    this.createWorker = config.createWorker;
    this.maxConcurrent = config.maxConcurrent ?? DEFAULT_MAX_CONCURRENT;
    this.getResult =
      config.getResult ??
      ((e: MessageEvent<TMessage>) => e.data as unknown as TResult);
  }

  /**
   * Execute a list of tasks across the worker pool.
   *
   * @param tasks  Array of task payloads to dispatch via `postMessage`.
   * @param signal Optional `AbortSignal` for cancellation.
   * @returns      Results in the same order as the input `tasks`.
   */
  async run(tasks: TTask[], signal?: AbortSignal): Promise<TResult[]> {
    const numTasks = tasks.length;

    if (numTasks === 0) return [];

    // ── Create workers for this run ──
    const poolSize = Math.min(this.maxConcurrent, numTasks);
    const workers: Worker[] = Array.from({ length: poolSize }, () =>
      this.createWorker()
    );

    const terminateAll = () => {
      workers.forEach((w) => w.terminate());
    };

    // Bail out immediately if already aborted.
    if (signal?.aborted) {
      terminateAll();
      throw signal.reason;
    }

    const results = new Array<TResult>(numTasks);
    let nextTask = 0;
    let completedTasks = 0;

    const assignTask = (
      worker: Worker,
      resolve: () => void,
      reject: (err: unknown) => void
    ) => {
      if (signal?.aborted || nextTask >= numTasks) return;

      const taskIdx = nextTask++;
      const task = tasks[taskIdx];

      worker.onmessage = (event: MessageEvent<TMessage>) => {
        const result = this.getResult(event, taskIdx);
        if (result === undefined) return; // Not a completion message.

        results[taskIdx] = result;
        completedTasks++;

        if (completedTasks === numTasks) {
          resolve();
        } else {
          assignTask(worker, resolve, reject);
        }
      };

      worker.onerror = (error) => reject(error);
      worker.postMessage(task);
    };

    await new Promise<void>((resolve, reject) => {
      const onAbort = () => {
        terminateAll();
        reject(signal!.reason);
      };
      signal?.addEventListener('abort', onAbort, { once: true });

      const wrappedResolve = () => {
        signal?.removeEventListener('abort', onAbort);
        resolve();
      };

      const wrappedReject = (err: unknown) => {
        signal?.removeEventListener('abort', onAbort);
        terminateAll();
        reject(err);
      };

      for (const worker of workers) {
        assignTask(worker, wrappedResolve, wrappedReject);
      }
    });

    // Guard against abort arriving between the await settling and here.
    signal?.throwIfAborted();

    terminateAll();

    return results;
  }
}
