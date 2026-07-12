"""
Range Coder + Adaptive Models - The Entropy Spine of KGC

Every regime of the Kaleidoscope Geometric Codec ends here:
    weights  -> lattice indices  -> range coder
    bytes    -> predictor CDFs   -> range coder
    context  -> consolidation residuals -> range coder

This replaces zlib as the bit-emitting engine, so the geometric
structures (E8 sites, Golay cosets, RG shells) are what actually
determine the compressed size — the Law of Compression Debt made
executable: the "program" is (model + coder), the "residual" is
the bits the model could not predict.

Implementation: carry-less Subbotin-style range coder (32-bit range,
byte-wise renormalization) with cumulative-frequency interface.
Deterministic, dependency-free, exactly invertible.
"""

from typing import List, Optional
import numpy as np

# Coder constants
TOP = 1 << 24  # renormalization threshold
BOT = 1 << 16  # maximum total frequency ("total" must be <= BOT)
MASK32 = 0xFFFFFFFF


class RangeEncoder:
    """Carry-less range encoder. Symbols are encoded via (cum, freq, tot)."""

    def __init__(self) -> None:
        self.low = 0
        self.range_ = MASK32
        self.out = bytearray()

    def encode(self, cum: int, freq: int, tot: int) -> None:
        """Encode a symbol occupying [cum, cum+freq) of a total tot scale."""
        assert 0 < freq and cum + freq <= tot <= BOT
        r = self.range_ // tot
        # All arithmetic is mod 2^32: intentional wrap replaces carry handling
        self.low = (self.low + r * cum) & MASK32
        self.range_ = r * freq
        # Renormalize
        while True:
            if (self.low ^ ((self.low + self.range_) & MASK32)) < TOP:
                pass  # top byte settled -> shift out
            elif self.range_ < BOT:
                self.range_ = (-self.low) & (BOT - 1)  # carry-less underflow fix
            else:
                break
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
            self.range_ = (self.range_ << 8) & MASK32

    def finish(self) -> bytes:
        for _ in range(4):
            self.out.append((self.low >> 24) & 0xFF)
            self.low = (self.low << 8) & MASK32
        return bytes(self.out)


class RangeDecoder:
    """Mirror of RangeEncoder."""

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0
        self.low = 0
        self.range_ = MASK32
        self.code = 0
        for _ in range(4):
            self.code = ((self.code << 8) | self._next_byte()) & MASK32

    def _next_byte(self) -> int:
        b = self.data[self.pos] if self.pos < len(self.data) else 0
        self.pos += 1
        return b

    def decode_freq(self, tot: int) -> int:
        """Return the cumulative-frequency slot the next symbol falls in."""
        self.r = self.range_ // tot
        f = ((self.code - self.low) & MASK32) // self.r
        return min(f, tot - 1)

    def decode_update(self, cum: int, freq: int, tot: int) -> None:
        self.low = (self.low + self.r * cum) & MASK32
        self.range_ = self.r * freq
        while True:
            if (self.low ^ ((self.low + self.range_) & MASK32)) < TOP:
                pass
            elif self.range_ < BOT:
                self.range_ = (-self.low) & (BOT - 1)
            else:
                break
            self.code = ((self.code << 8) | self._next_byte()) & MASK32
            self.low = (self.low << 8) & MASK32
            self.range_ = (self.range_ << 8) & MASK32


class AdaptiveModel:
    """
    Adaptive frequency model over `n_symbols` with periodic rescaling.

    Uses a Fenwick (binary indexed) tree so encode/decode are O(log n)
    even for large alphabets (e.g. 4096 Golay cosets).
    """

    INCREMENT = 32
    # Rescale early: capping `total` bounds the worst-case cost of a
    # never-seen symbol at log2(LIMIT) bits, which keeps near-random
    # data close to 8 bits/byte instead of expanding.
    LIMIT = 4096

    def __init__(self, n_symbols: int) -> None:
        self.n = n_symbols
        self.tree = np.zeros(n_symbols + 1, dtype=np.int64)
        self.total = 0
        for s in range(n_symbols):  # uniform prior of 1
            self._add(s, 1)

    def _add(self, sym: int, delta: int) -> None:
        i = sym + 1
        while i <= self.n:
            self.tree[i] += delta
            i += i & (-i)
        self.total += delta

    def _cum(self, sym: int) -> int:
        """Cumulative frequency of symbols < sym."""
        i, s = sym, 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return int(s)

    def freq(self, sym: int) -> int:
        return self._cum(sym + 1) - self._cum(sym)

    def find(self, cumfreq: int) -> int:
        """Find symbol whose interval contains cumfreq."""
        idx = 0
        bitmask = 1 << (self.n.bit_length())
        rem = cumfreq
        while bitmask:
            nxt = idx + bitmask
            if nxt <= self.n and self.tree[nxt] <= rem:
                idx = nxt
                rem -= self.tree[nxt]
            bitmask >>= 1
        return idx  # symbol index

    def update(self, sym: int) -> None:
        self._add(sym, self.INCREMENT)
        if self.total >= self.LIMIT:
            self._rescale()

    def _rescale(self) -> None:
        freqs = np.array([self.freq(s) for s in range(self.n)], dtype=np.int64)
        freqs = np.maximum(freqs // 2, 1)
        self.tree[:] = 0
        self.total = 0
        for s in range(self.n):
            self._add(s, int(freqs[s]))

    # --- coder plumbing ---
    def encode(self, enc: RangeEncoder, sym: int) -> None:
        cum = self._cum(sym)
        f = self._cum(sym + 1) - cum
        enc.encode(cum, f, self.total)
        self.update(sym)

    def decode(self, dec: RangeDecoder) -> int:
        cf = dec.decode_freq(self.total)
        sym = self.find(cf)
        cum = self._cum(sym)
        f = self._cum(sym + 1) - cum
        dec.decode_update(cum, f, self.total)
        self.update(sym)
        return sym


class Order1ByteModel:
    """
    Order-1 adaptive byte model: one AdaptiveModel per previous-byte context.
    Contexts are created lazily to keep memory proportional to what's seen.
    """

    def __init__(self) -> None:
        self.ctx: dict = {}

    def _model(self, prev: int) -> AdaptiveModel:
        m = self.ctx.get(prev)
        if m is None:
            m = AdaptiveModel(256)
            self.ctx[prev] = m
        return m

    def compress(self, data: bytes) -> bytes:
        enc = RangeEncoder()
        prev = 0
        for b in data:
            self._model(prev).encode(enc, b)
            prev = b
        return enc.finish()

    def decompress(self, blob: bytes, n: int) -> bytes:
        dec = RangeDecoder(blob)
        out = bytearray()
        prev = 0
        for _ in range(n):
            b = self._model(prev).decode(dec)
            out.append(b)
            prev = b
        return bytes(out)


class Order2ByteModel:
    """
    Order-2 adaptive byte model with lazy hashed contexts.

    Contexts are (prev2, prev1) pairs. Stronger than order-1 on text
    and source code at the cost of slower adaptation on tiny inputs.
    """

    def __init__(self) -> None:
        self.ctx: dict = {}

    def _model(self, c2: int, c1: int) -> AdaptiveModel:
        key = (c2 << 8) | c1
        m = self.ctx.get(key)
        if m is None:
            m = AdaptiveModel(256)
            self.ctx[key] = m
        return m

    def compress(self, data: bytes) -> bytes:
        enc = RangeEncoder()
        c2 = c1 = 0
        for b in data:
            self._model(c2, c1).encode(enc, b)
            c2, c1 = c1, b
        return enc.finish()

    def decompress(self, blob: bytes, n: int) -> bytes:
        dec = RangeDecoder(blob)
        out = bytearray()
        c2 = c1 = 0
        for _ in range(n):
            b = self._model(c2, c1).decode(dec)
            out.append(b)
            c2, c1 = c1, b
        return bytes(out)


class GeometricContextModel:
    """
    E8-trajectory context mixing (the Kaleidoscope move).

    The context id for the next byte is not just the previous byte:
    it is the E8 lattice site of the recent byte-window embedding.
    The data's trajectory through E8 space IS the context state —
    "memory as angle, rotation, phase, and retrieval path."

    Encoder and decoder replay the identical trajectory, so no side
    information is required: the geometry is regenerated, not stored.
    """

    WINDOW = 8

    def __init__(self, scale: float = 24.0) -> None:
        from e8zip.core.e8_lattice import E8Lattice

        self.lattice = E8Lattice()
        self.scale = scale
        self.ctx: dict = {}
        self.window = bytearray(self.WINDOW)

    def _site_id(self) -> int:
        v = np.frombuffer(bytes(self.window), dtype=np.uint8).astype(np.float64)
        v = (v - 127.5) / self.scale  # center + scale into lattice range
        _, point, _ = self.lattice.nearest_lattice_point(v)
        # 2*point is integral for both cosets -> exact integer key
        key = tuple(np.rint(2.0 * point).astype(np.int64).tolist())
        return hash(key) & 0xFFFF

    def _model(self, ctx_id: int) -> AdaptiveModel:
        m = self.ctx.get(ctx_id)
        if m is None:
            m = AdaptiveModel(256)
            self.ctx[ctx_id] = m
        return m

    def _push(self, b: int) -> None:
        self.window[:-1] = self.window[1:]
        self.window[-1] = b

    def compress(self, data: bytes) -> bytes:
        enc = RangeEncoder()
        for b in data:
            ctx_id = (self._site_id() << 8) ^ self.window[-1]
            self._model(ctx_id & 0xFFFF).encode(enc, b)
            self._push(b)
        return enc.finish()

    def decompress(self, blob: bytes, n: int) -> bytes:
        dec = RangeDecoder(blob)
        out = bytearray()
        for _ in range(n):
            ctx_id = (self._site_id() << 8) ^ self.window[-1]
            b = self._model(ctx_id & 0xFFFF).decode(dec)
            out.append(b)
            self._push(b)
        return bytes(out)


def encode_indices(indices: np.ndarray, n_symbols: int) -> bytes:
    """Entropy-code an integer index stream (lattice/coset ids) adaptively."""
    enc = RangeEncoder()
    model = AdaptiveModel(n_symbols)
    for s in indices:
        model.encode(enc, int(s))
    return enc.finish()


def decode_indices(blob: bytes, count: int, n_symbols: int) -> np.ndarray:
    dec = RangeDecoder(blob)
    model = AdaptiveModel(n_symbols)
    out = np.empty(count, dtype=np.int64)
    for i in range(count):
        out[i] = model.decode(dec)
    return out
