import { nest2fxy, ring2fxy } from 'healpix-ts';
import {
  u64_add,
  u64_sub,
  u64_mul32,
  u64_shr,
  u64_shl,
  u64_and,
  u64_lt,
  u64_div32,
  u64_isqrt,
  toBig,
  fromBig,
  decodeNest,
  compact1By1,
  decodeRing,
  ringUniforms
} from './gpu-decode-reference';

const TWO32 = 4294967296n;

describe('u64_* helpers', () => {
  describe('roundtrip', () => {
    it('toBig / fromBig roundtrip on boundaries', () => {
      const cases = [0n, 1n, TWO32 - 1n, TWO32, TWO32 + 1n, (1n << 52n) - 1n];
      for (const x of cases) {
        expect(toBig(fromBig(x))).toBe(x);
      }
    });
  });

  describe('u64_add', () => {
    it('no carry', () => {
      expect(u64_add([1, 0], [2, 0])).toEqual([3, 0]);
    });
    it('carry into hi', () => {
      expect(u64_add([0xffffffff, 0], [1, 0])).toEqual([0, 1]);
    });
    it('full 64-bit add, no overflow', () => {
      const a = fromBig(0xdeadbeefcafef00dn);
      const b = fromBig(0x0000000112345678n);
      expect(toBig(u64_add(a, b))).toBe(0xdeadbef0dd334685n);
    });
  });

  describe('u64_sub', () => {
    it('no borrow', () => {
      expect(u64_sub([5, 0], [2, 0])).toEqual([3, 0]);
    });
    it('borrow from hi', () => {
      expect(u64_sub([0, 1], [1, 0])).toEqual([0xffffffff, 0]);
    });
    it('full 64-bit sub', () => {
      const a = fromBig(0x100000000n);
      const b = fromBig(1n);
      expect(toBig(u64_sub(a, b))).toBe(0xffffffffn);
    });
  });

  describe('u64_mul32', () => {
    it('small * small', () => {
      expect(u64_mul32(3, 7)).toEqual([21, 0]);
    });
    it('big * big overflows into hi', () => {
      expect(toBig(u64_mul32(0xffffffff, 0xffffffff))).toBe(
        0xfffffffe00000001n
      );
    });
    it('2^16 * 2^16 = 2^32', () => {
      expect(toBig(u64_mul32(0x10000, 0x10000))).toBe(0x100000000n);
    });
    // Regression: bit 15 of `(mid & 0xffff)` being set makes `(x << 16)`
    // wrap to a negative int32 in JS. Earlier the `loFull >= TWO32` check
    // missed that overflow and produced hi-1 for inputs like 131071².
    it('131071^2 (bit-15 mid carry)', () => {
      expect(toBig(u64_mul32(131071, 131071))).toBe(17179607041n);
    });
    it('131070^2 (bit-15 mid carry)', () => {
      expect(toBig(u64_mul32(131070, 131070))).toBe(17179344900n);
    });
  });

  describe('u64_shr', () => {
    it('shift 0 is identity', () => {
      expect(u64_shr([0xdeadbeef, 0xcafef00d], 0)).toEqual([
        0xdeadbeef, 0xcafef00d
      ]);
    });
    it('shift 4 within lo', () => {
      expect(toBig(u64_shr(fromBig(0x123456789abcdef0n), 4))).toBe(
        0x0123456789abcdefn
      );
    });
    it('shift 32 moves hi to lo', () => {
      expect(u64_shr([0xdeadbeef, 0xcafef00d], 32)).toEqual([0xcafef00d, 0]);
    });
    it('shift 48', () => {
      expect(toBig(u64_shr(fromBig(0x0123456789abcdefn), 48))).toBe(0x123n);
    });
    it('shift 63', () => {
      expect(u64_shr([0, 0x80000000], 63)).toEqual([1, 0]);
    });
  });

  describe('u64_shl', () => {
    it('shift 32 moves lo to hi', () => {
      expect(u64_shl([0xdeadbeef, 0], 32)).toEqual([0, 0xdeadbeef]);
    });
    it('shift 1 with cross-half carry', () => {
      expect(u64_shl([0x80000000, 0], 1)).toEqual([0, 1]);
    });
    it('shift 48', () => {
      expect(toBig(u64_shl(fromBig(0x123n), 48))).toBe(0x0123000000000000n);
    });
  });

  describe('u64_and', () => {
    it('mask lower 24 bits', () => {
      const v = fromBig(0xdeadbeefcafef00dn);
      const mask = fromBig(0xffffffn);
      expect(toBig(u64_and(v, mask))).toBe(0xfef00dn);
    });
  });

  describe('u64_lt', () => {
    it('hi dominates', () => {
      expect(u64_lt([100, 1], [1, 2])).toBe(true);
    });
    it('tie on hi, lo decides', () => {
      expect(u64_lt([1, 5], [2, 5])).toBe(true);
      expect(u64_lt([5, 5], [2, 5])).toBe(false);
    });
    it('equal → false', () => {
      expect(u64_lt([5, 5], [5, 5])).toBe(false);
    });
  });

  describe('u64_div32', () => {
    it('exact division', () => {
      const out = u64_div32(fromBig(1000n), 10);
      expect(out.q).toBe(100);
      expect(out.r).toBe(0);
    });
    it('with remainder', () => {
      const out = u64_div32(fromBig(1003n), 10);
      expect(out.q).toBe(100);
      expect(out.r).toBe(3);
    });
    it('large dividend', () => {
      const n = (1n << 50n) + 12345n;
      const d = 4_000_003;
      const out = u64_div32(fromBig(n), d);
      const q = n / BigInt(d);
      const r = n - q * BigInt(d);
      expect(BigInt(out.q)).toBe(q);
      expect(BigInt(out.r)).toBe(r);
    });
    it('k=2^51, d=2^26 (equatorial worst case)', () => {
      const n = 1n << 51n;
      const d = 1 << 26;
      const out = u64_div32(fromBig(n), d);
      expect(BigInt(out.q)).toBe(n / BigInt(d));
      expect(out.r).toBe(0);
    });
  });

  describe('u64_isqrt', () => {
    it('square numbers', () => {
      expect(u64_isqrt(fromBig(0n))).toBe(0);
      expect(u64_isqrt(fromBig(1n))).toBe(1);
      expect(u64_isqrt(fromBig(4n))).toBe(2);
      expect(u64_isqrt(fromBig(9n))).toBe(3);
      expect(u64_isqrt(fromBig(100000000n))).toBe(10000);
    });
    it('non-square rounds down', () => {
      expect(u64_isqrt(fromBig(2n))).toBe(1);
      expect(u64_isqrt(fromBig(3n))).toBe(1);
      expect(u64_isqrt(fromBig(8n))).toBe(2);
      expect(u64_isqrt(fromBig(99n))).toBe(9);
    });
    it('large value near 2^49 (polar cap worst case)', () => {
      const n = (1n << 49n) - 7n;
      const expected = 23726566n; // floor(sqrt(2^49 - 7))
      expect(BigInt(u64_isqrt(fromBig(n)))).toBe(expected);
    });
    it('2^52 - 1 (upper end of safe-integer domain)', () => {
      const n = (1n << 52n) - 1n;
      const q = BigInt(u64_isqrt(fromBig(n)));
      expect(q * q <= n).toBe(true);
      expect((q + 1n) * (q + 1n) > n).toBe(true);
    });
  });
});

describe('compact1By1', () => {
  it('extracts even bits from a 32-bit word', () => {
    expect(compact1By1(0b01010101)).toBe(0b1111);
    expect(compact1By1(0x55555555)).toBe(0xffff);
    expect(compact1By1(0xaaaaaaaa)).toBe(0);
  });
});

describe('decodeNest', () => {
  const NSIDES = [1, 2, 4, 8, 256, 1 << 12, 1 << 15, 1 << 16, 1 << 20, 1 << 24];

  for (const nside of NSIDES) {
    it(`matches nest2fxy across 12 faces at nside=${nside}`, () => {
      const log2n = Math.log2(nside);
      const nside2 = BigInt(nside) * BigInt(nside);
      for (let face = 0; face < 12; face++) {
        const ids = [
          0n,
          nside2 - 1n,
          nside2 / 2n,
          (nside2 * 3n) / 7n
        ].map((k) => BigInt(face) * nside2 + k);
        for (const id of ids) {
          const truth = nest2fxy(nside, Number(id));
          const cellId = fromBig(id);
          const got = decodeNest(cellId, log2n);
          expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
        }
      }
    });
  }

  it('random NEST ids at nside=2^24', () => {
    const nside = 1 << 24;
    const log2n = 24;
    const nside2 = BigInt(nside) * BigInt(nside);
    let state = 0x9e3779b97f4a7c15n;
    const rand = () => {
      state = (state * 6364136223846793005n + 1442695040888963407n) &
              0xffffffffffffffffn;
      return state;
    };
    for (let trial = 0; trial < 200; trial++) {
      const f = Number(rand() % 12n);
      const k = rand() % nside2;
      const id = BigInt(f) * nside2 + k;
      const truth = nest2fxy(nside, Number(id));
      const got = decodeNest(fromBig(id), log2n);
      expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
    }
  });
});

describe('decodeRing', () => {
  const NSIDES_SMALL = [1, 2, 4, 8, 256];
  const NSIDES_LARGE = [1 << 12, 1 << 15, 1 << 16, 1 << 20, 1 << 24];

  function checkRange(nside: number, ids: bigint[]): void {
    const u = ringUniforms(nside);
    for (const id of ids) {
      const truth = ring2fxy(nside, Number(id));
      const got = decodeRing(fromBig(id), nside, u.polarLim, u.eqLim, u.npix);
      expect({ f: got.face, x: got.ix, y: got.iy }).toEqual(truth);
    }
  }

  for (const nside of NSIDES_SMALL) {
    it(`matches ring2fxy exhaustively at nside=${nside}`, () => {
      const npix = 12 * nside * nside;
      const ids: bigint[] = [];
      for (let i = 0; i < npix; i++) ids.push(BigInt(i));
      checkRange(nside, ids);
    });
  }

  for (const nside of NSIDES_LARGE) {
    it(`matches ring2fxy at boundaries, nside=${nside}`, () => {
      const n = BigInt(nside);
      const polar = 2n * n * (n - 1n);
      const eq = polar + 8n * n * n;
      const npix = 12n * n * n;
      const ids = [
        0n,
        1n,
        polar - 1n,
        polar,
        polar + 1n,
        polar + 4n * n,
        eq - 1n,
        eq,
        eq + 1n,
        npix - 1n
      ];
      checkRange(nside, ids);
    });

    it(`matches ring2fxy on 200 random ids, nside=${nside}`, () => {
      const n = BigInt(nside);
      const npix = 12n * n * n;
      let state = (BigInt(nside) * 0x9e3779b97f4a7c15n) & 0xffffffffffffffffn;
      const rand = () => {
        state = (state * 6364136223846793005n + 1442695040888963407n) &
                0xffffffffffffffffn;
        return state;
      };
      const ids: bigint[] = [];
      for (let i = 0; i < 200; i++) ids.push(rand() % npix);
      checkRange(nside, ids);
    });
  }
});
