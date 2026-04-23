import { WorkerPool } from './worker-pool';

type WorkerLike = {
  onmessage:
    | ((event: MessageEvent<{ done: true; value: number }>) => void)
    | null;
  onerror: ((error: unknown) => void) | null;
  postMessage: (task: number) => void;
  terminate: () => void;
};

class MockWorker implements WorkerLike {
  onmessage:
    | ((event: MessageEvent<{ done: true; value: number }>) => void)
    | null = null;
  onerror: ((error: unknown) => void) | null = null;
  terminated = false;

  postMessage(task: number): void {
    setTimeout(() => {
      this.onmessage?.({
        data: { done: true, value: task * 2 }
      } as MessageEvent<{ done: true; value: number }>);
    }, 0);
  }

  terminate(): void {
    this.terminated = true;
  }
}

describe('WorkerPool', () => {
  it('returns results in task order', async () => {
    const created: MockWorker[] = [];
    const pool = new WorkerPool<number, { done: true; value: number }, number>({
      createWorker: () => {
        const w = new MockWorker();
        created.push(w);
        return w as unknown as Worker;
      },
      maxConcurrent: 2,
      getResult: (e) => (e.data.done ? e.data.value : undefined)
    });

    const result = await pool.run([1, 2, 3, 4]);
    expect(result).toEqual([2, 4, 6, 8]);
    expect(created.every((w) => w.terminated)).toBe(true);
  });

  it('returns empty array for empty tasks', async () => {
    const pool = new WorkerPool<number, { done: true; value: number }, number>({
      createWorker: () => new MockWorker() as unknown as Worker
    });
    await expect(pool.run([])).resolves.toEqual([]);
  });
});
