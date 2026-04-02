describe('tile-grid.worker', () => {
  it('registers a message handler and posts geometry buffers', async () => {
    const postMessage = jest.fn();
    const selfMock: {
      onmessage: ((ev: MessageEvent) => void) | null;
      postMessage: typeof postMessage;
    } = {
      onmessage: null,
      postMessage
    };

    Object.defineProperty(globalThis, 'self', {
      configurable: true,
      value: selfMock
    });

    await import('./tile-grid.worker');

    expect(typeof selfMock.onmessage).toBe('function');

    selfMock.onmessage?.({
      data: {
        nside: 1,
        cellIds: new Int32Array([0]),
        scheme: 'nest'
      }
    } as MessageEvent);

    expect(postMessage).toHaveBeenCalledTimes(1);
    const [message, transfer] = postMessage.mock.calls[0];
    expect(message.type).toBe('data');
    expect(message.data.coords).toBeInstanceOf(Float32Array);
    expect(message.data.indexes).toBeInstanceOf(Uint32Array);
    expect(message.data.triangles).toBeInstanceOf(Uint32Array);
    expect(Array.isArray(transfer.transfer)).toBe(true);
  });
});
