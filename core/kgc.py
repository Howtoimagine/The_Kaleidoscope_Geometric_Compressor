"""
KGC - Kaleidoscope Geometric Codec (the unified compressor)

One codec, three regimes, one law. Every archive is a Compression Debt
triple, the Glass Network's Law 9 made executable:

    L(x) = L(address) + L(program) + L(residual)

    address  - what the compressed thing points back to: sha256 of the
               source, member ids, conserved mass. Lossy compression
               WITHOUT addresses is refused (Law 9: "compression lacks
               source addresses" -> unlawful).
    program  - the executable reconstruction recipe: which codec, which
               lattice, which seeds. Not stored bytes - a procedure.
    residual - the entropy-coded bits the geometric prior could not
               predict. This is the only part that scales with surprise.

Regimes (three rotations of the same object):

    BYTES   (lossless)  predictor -> range coder. The E8 trajectory of
            the data is the context state; geometry is regenerated on
            decode, never stored.
    TENSOR  (rate-distortion) randomized Hadamard incoherence -> exact
            Leech/E8 lattice VQ -> entropy-coded (g_idx, case, z).
            QuIP#-family weight compression with honest bits/weight.
    CONSOLIDATE (RG) rows collapse to canonical Leech sites at a chosen
            scale; conserved mass and member addresses ride along, so
            the collapse is reversible (with residuals) or auditable
            (without). The KMind consolidation tick as a file format.

truth_status ladder (from the Glass Network language decoder):
    0 exact_recovery | 1 reconstruction | 2 interpretation | 3 confabulation
Every decompression reports which rung you are standing on.
"""

import hashlib
import struct
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from e8zip.core.entropy import (
    AdaptiveModel,
    GeometricContextModel,
    Order1ByteModel,
    Order2ByteModel,
    RangeDecoder,
    RangeEncoder,
)
from e8zip.core.golay import decode_24, encode_12_to_24
from e8zip.core.leech_lattice import (
    batch_nearest_leech_point,
    leech_index_decompose,
    leech_index_reconstruct,
)
from e8zip.core import transforms

MAGIC = b"KGC2"
VERSION = 2

# Regimes
REGIME_BYTES = 1
REGIME_TENSOR = 2
REGIME_CONSOLIDATE = 3

# truth_status rungs
EXACT_RECOVERY = 0
RECONSTRUCTION = 1
INTERPRETATION = 2
CONFABULATION = 3

TRUTH_NAMES = {
    EXACT_RECOVERY: "exact_recovery",
    RECONSTRUCTION: "reconstruction",
    INTERPRETATION: "interpretation",
    CONFABULATION: "confabulation",
}

# Byte-regime programs
PROG_RAW = 0
PROG_ORDER1 = 1
PROG_GEOMETRIC = 2
PROG_ORDER2 = 3


class Law9Error(ValueError):
    """Raised when lossy compression is attempted without source addresses."""


@dataclass
class DebtRecord:
    """The address section of an archive: what reconstruction owes."""

    source_sha256: bytes
    source_length: int
    conserved_mass: float = 0.0
    member_count: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)


def _sha(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _pack_stream(payload: bytes) -> bytes:
    return struct.pack("<I", len(payload)) + payload


def _unpack_stream(data: bytes, offset: int) -> Tuple[bytes, int]:
    (n,) = struct.unpack_from("<I", data, offset)
    offset += 4
    return data[offset : offset + n], offset + n


def _pack_control(version: int, regime: int, truth: int, flags: int) -> int:
    """Pack the four control fields into a 12-bit Golay message."""
    return (
        (version & 0x7) | ((regime & 0x7) << 3) | ((truth & 0x3) << 6) | ((flags & 0x3) << 8)
    )


def _unpack_control(msg: int) -> Tuple[int, int, int, int]:
    return msg & 0x7, (msg >> 3) & 0x7, (msg >> 6) & 0x3, (msg >> 8) & 0x3


def _header(regime: int, truth: int, debt: DebtRecord) -> bytes:
    # Control block (version/regime/truth/flags) is Golay-protected: it
    # decides how the rest of the archive is parsed, so a few flipped
    # bits here must not brick the whole file. Heals <=3 bit errors.
    codeword = encode_12_to_24(_pack_control(VERSION, regime, truth, 0))
    return (
        MAGIC
        + codeword.to_bytes(3, "little")
        + struct.pack("<Q", debt.source_length)
        + debt.source_sha256
        + struct.pack("<dI", debt.conserved_mass, debt.member_count)
    )


HEADER_SIZE = 4 + 3 + 8 + 32 + 12


def _parse_header(data: bytes) -> Tuple[int, int, DebtRecord, int]:
    if data[:4] != MAGIC:
        raise ValueError("not a KGC2 archive")
    codeword = int.from_bytes(data[4:7], "little")
    decoded = decode_24(codeword)
    if decoded is None:
        raise ValueError(
            "KGC2 header control block corrupted beyond repair (>=4 bit errors)"
        )
    _, msg, _errors_corrected = decoded
    version, regime, truth, _flags = _unpack_control(msg)
    if version != VERSION:
        raise ValueError(f"unsupported KGC version {version}")
    (source_length,) = struct.unpack_from("<Q", data, 7)
    sha = data[15:47]
    mass, members = struct.unpack_from("<dI", data, 47)
    debt = DebtRecord(sha, source_length, mass, members)
    return regime, truth, debt, HEADER_SIZE


# ======================================================================
# Regime 1: BYTES (lossless)
# ======================================================================


def compress_bytes(data: bytes, mode: str = "strong") -> bytes:
    """
    Lossless byte compression through the range coder.

    mode: 'fast'      -> order-1 adaptive context
          'strong'    -> order-2 adaptive context (default)
          'geometric' -> E8-trajectory context (experimental: the data's
                         path through E8 space is the context state)
    Falls back to raw storage whenever the model fails to shrink the
    data (random input stays ~1.0x instead of expanding).
    """
    debt = DebtRecord(_sha(data), len(data))

    if mode == "fast":
        prog, coded = PROG_ORDER1, Order1ByteModel().compress(data)
    elif mode == "strong":
        prog, coded = PROG_ORDER2, Order2ByteModel().compress(data)
    else:
        prog, coded = PROG_GEOMETRIC, GeometricContextModel().compress(data)

    if len(coded) >= len(data):  # honest fallback, never expand
        prog, coded = PROG_RAW, data

    return _header(REGIME_BYTES, EXACT_RECOVERY, debt) + struct.pack("<B", prog) + coded


def decompress_bytes(data: bytes) -> Tuple[bytes, Dict[str, Any]]:
    regime, truth, debt, off = _parse_header(data)
    if regime != REGIME_BYTES:
        raise ValueError("archive is not byte-regime")
    (prog,) = struct.unpack_from("<B", data, off)
    payload = data[off + 1 :]

    if prog == PROG_RAW:
        out = payload[: debt.source_length]
    elif prog == PROG_ORDER1:
        out = Order1ByteModel().decompress(payload, debt.source_length)
    elif prog == PROG_ORDER2:
        out = Order2ByteModel().decompress(payload, debt.source_length)
    elif prog == PROG_GEOMETRIC:
        out = GeometricContextModel().decompress(payload, debt.source_length)
    else:
        raise ValueError(f"unknown byte program {prog}")

    verified = _sha(out) == debt.source_sha256
    if not verified:
        raise ValueError("KGC integrity failure: sha256 mismatch after decode")
    return out, {"truth_status": TRUTH_NAMES[truth], "verified": True}


# ======================================================================
# Regime 2: TENSOR (rate-distortion lattice VQ)
# ======================================================================


TRANSFORM_KINDS = ("hadamard", "rotation", "none")


def _zigzag(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 2 * z, -2 * z - 1).astype(np.int64)


def _unzigzag(u: np.ndarray) -> np.ndarray:
    return np.where(u % 2 == 0, u // 2, -(u + 1) // 2).astype(np.int64)


def compress_tensor(
    w: np.ndarray,
    scale: float = 4.0,
    hadamard_dim: int = 128,
    seed: int = 42,
    transform: str = "hadamard",
) -> bytes:
    """
    Lattice-VQ compression of a weight/embedding tensor.

    Pipeline: flatten -> incoherence transform -> unit-var normalize ->
    * scale -> 24-D blocks -> exact Leech CVP -> entropy-coded
    (g_idx, case, zigzag(z)).

    `scale` is the rate-distortion knob: higher = more bits, higher SNR.
    `transform` selects the incoherence pass ('hadamard': QuIP#-style,
    O(n log n); 'rotation': dense random orthogonal, O(n^2), stronger
    decorrelation - see core/transforms.py). This is deliberately lossy
    (like every PTQ method); the archive carries the source hash so
    downstream fidelity is auditable.
    """
    if transform not in TRANSFORM_KINDS:
        raise ValueError(f"transform must be one of {TRANSFORM_KINDS}")
    w = np.asarray(w, dtype=np.float64)
    orig_shape = w.shape
    flat = w.ravel()
    n = flat.size

    src_bytes = np.asarray(w, dtype=np.float32).tobytes()
    debt = DebtRecord(_sha(src_bytes), len(src_bytes))
    debt.conserved_mass = float(np.sum(flat**2))

    # Incoherence pass over rows of hadamard_dim
    pad_h = (-n) % hadamard_dim
    padded = np.concatenate([flat, np.zeros(pad_h)])
    rows = padded.reshape(-1, hadamard_dim)
    inc = transforms.forward(rows, transform, seed).ravel()

    sigma = float(np.std(inc)) or 1.0
    x = inc / sigma * scale

    # 24-D blocks
    pad_b = (-x.size) % 24
    xb = np.concatenate([x, np.zeros(pad_b)]).reshape(-1, 24)

    points, _, _ = batch_nearest_leech_point(xb)
    g_idx, case, z = leech_index_decompose(points)
    zz = _zigzag(z.ravel())
    z_alpha = int(zz.max()) + 1 if zz.size else 1

    # Entropy-code the three streams
    enc = RangeEncoder()
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    m_z = AdaptiveModel(max(z_alpha, 2))
    for i in range(len(g_idx)):
        m_g.encode(enc, int(g_idx[i]))
        m_c.encode(enc, int(case[i]))
    for u in zz:
        m_z.encode(enc, int(u))
    coded = enc.finish()

    transform_id = TRANSFORM_KINDS.index(transform)
    params = struct.pack(
        "<IIddIIBB",
        n,
        len(g_idx),
        sigma,
        scale,
        hadamard_dim,
        seed,
        len(orig_shape),
        transform_id,
    ) + struct.pack(f"<{len(orig_shape)}I", *orig_shape)
    return (
        _header(REGIME_TENSOR, RECONSTRUCTION, debt)
        + params
        + struct.pack("<I", max(z_alpha, 2))
        + _pack_stream(coded)
    )


def decompress_tensor(data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    regime, truth, debt, off = _parse_header(data)
    if regime != REGIME_TENSOR:
        raise ValueError("archive is not tensor-regime")

    n, n_blocks, sigma, scale, hadamard_dim, seed, ndim, transform_id = struct.unpack_from(
        "<IIddIIBB", data, off
    )
    transform = TRANSFORM_KINDS[transform_id]
    off += struct.calcsize("<IIddIIBB")
    shape = struct.unpack_from(f"<{ndim}I", data, off)
    off += 4 * ndim
    (z_alpha,) = struct.unpack_from("<I", data, off)
    off += 4
    coded, off = _unpack_stream(data, off)

    dec = RangeDecoder(coded)
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    m_z = AdaptiveModel(z_alpha)
    g_idx = np.empty(n_blocks, dtype=np.int64)
    case = np.empty(n_blocks, dtype=np.int64)
    for i in range(n_blocks):
        g_idx[i] = m_g.decode(dec)
        case[i] = m_c.decode(dec)
    zz = np.empty(n_blocks * 24, dtype=np.int64)
    for i in range(n_blocks * 24):
        zz[i] = m_z.decode(dec)
    z = _unzigzag(zz).reshape(n_blocks, 24)

    points = leech_index_reconstruct(g_idx, case, z)
    x = points.ravel()

    # Undo scale, undo incoherence
    inc = x / scale * sigma
    pad_h = (-n) % hadamard_dim
    total = n + pad_h
    rows = inc[:total].reshape(-1, hadamard_dim)
    flat = transforms.inverse(rows, transform, seed).ravel()[:n]

    w = flat.reshape(shape).astype(np.float32)
    return w, {
        "truth_status": TRUTH_NAMES[truth],
        "source_sha256": debt.source_sha256.hex(),
        "conserved_mass": debt.conserved_mass,
        "bits_per_weight": None,  # filled by caller who knows archive size
    }


# ======================================================================
# Regime 3: CONSOLIDATE (RG collapse of row sets to Leech sites)
# ======================================================================


def consolidate(
    rows: np.ndarray,
    rg_scale: float = 1.0,
    keep_residuals: bool = True,
    member_ids: Optional[List[str]] = None,
) -> bytes:
    """
    RG consolidation of an (N, 24) row set (embeddings, KV entries).

    Rows are quantized to canonical Leech sites at coarse-graining scale
    `rg_scale` (smaller = coarser: rows are multiplied by rg_scale
    before snapping, the block-spin "zoom out"). Rows landing on the
    same site collapse into one representative.

    Law 9: a lossy collapse (keep_residuals=False) REQUIRES member
    addresses. If member_ids is None they are auto-derived as row
    hashes, so the debt is always addressable.

    Conserved mass = sum of squared row norms, transferred to sites and
    checked on reconstruction (the KMind's foreground-mass conservation).
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[1] != 24:
        raise ValueError("consolidate expects (N, 24) rows")
    n = rows.shape[0]

    src_bytes = rows.astype(np.float32).tobytes()
    debt = DebtRecord(_sha(src_bytes), len(src_bytes))
    debt.conserved_mass = float(np.sum(rows**2))
    debt.member_count = n

    if member_ids is None:
        member_ids = [
            hashlib.sha256(rows[i].astype(np.float32).tobytes()).hexdigest()[:16]
            for i in range(n)
        ]
    if not keep_residuals and not member_ids:
        raise Law9Error("compression lacks source addresses")

    # Block-spin: scale then snap
    points, _, _ = batch_nearest_leech_point(rows * rg_scale)
    g_idx, case, z = leech_index_decompose(points)

    # Group rows by site
    site_keys: Dict[Tuple[int, int, bytes], int] = {}
    assignments = np.empty(n, dtype=np.int64)
    sites: List[Tuple[int, int, np.ndarray]] = []
    for i in range(n):
        key = (int(g_idx[i]), int(case[i]), z[i].tobytes())
        if key not in site_keys:
            site_keys[key] = len(sites)
            sites.append((int(g_idx[i]), int(case[i]), z[i]))
        assignments[i] = site_keys[key]
    n_sites = len(sites)

    # Per-site conserved mass (the KMind's foreground-mass transfer):
    # reconstruction rescales each site so its group's energy matches.
    site_mass = np.zeros(n_sites, dtype=np.float64)
    site_count = np.zeros(n_sites, dtype=np.int64)
    row_energy = np.sum(rows**2, axis=1)
    for i in range(n):
        site_mass[assignments[i]] += row_energy[i]
        site_count[assignments[i]] += 1

    # Encode: site table + assignments + optional residuals
    enc = RangeEncoder()
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    zz_all = _zigzag(np.array([s[2] for s in sites]).ravel())
    z_alpha = int(zz_all.max()) + 1 if zz_all.size else 1
    m_z = AdaptiveModel(max(z_alpha, 2))
    for gi, ci, _zi in sites:
        m_g.encode(enc, gi)
        m_c.encode(enc, ci)
    for u in zz_all:
        m_z.encode(enc, int(u))
    m_a = AdaptiveModel(max(n_sites, 2))
    for a in assignments:
        m_a.encode(enc, int(a))
    coded = enc.finish()

    site_points = leech_index_reconstruct(
        np.array([s[0] for s in sites]),
        np.array([s[1] for s in sites]),
        np.array([s[2] for s in sites]),
    )
    # Round masses through float32 so the encoder reconstructs against
    # exactly what the decoder will read back from the archive.
    site_mass32 = site_mass.astype(np.float32).astype(np.float64)
    recon32 = _mass_scaled_recon(
        site_points, site_mass32, site_count, assignments, rg_scale
    )

    residual_blob = b""
    if keep_residuals:
        # Bitwise-exact residuals: XOR of the IEEE-754 payloads. Immune
        # to double-rounding, and near-zero residuals have mostly-zero
        # high bytes, which zlib exploits.
        rows32 = rows.astype(np.float32)
        xor = rows32.view(np.uint32) ^ recon32.view(np.uint32)
        residual_blob = zlib.compress(xor.tobytes(), 6)

    addr_blob = zlib.compress("\n".join(member_ids).encode(), 6)
    mass_blob = site_mass32.astype(np.float32).tobytes()

    params = struct.pack(
        "<IIdIB", n, n_sites, rg_scale, max(z_alpha, 2), int(keep_residuals)
    )
    truth = EXACT_RECOVERY if keep_residuals else RECONSTRUCTION
    return (
        _header(REGIME_CONSOLIDATE, truth, debt)
        + params
        + _pack_stream(coded)
        + _pack_stream(mass_blob)
        + _pack_stream(residual_blob)
        + _pack_stream(addr_blob)
    )


def _mass_scaled_recon(
    site_points: np.ndarray,
    site_mass: np.ndarray,
    site_count: np.ndarray,
    assignments: np.ndarray,
    rg_scale: float,
) -> np.ndarray:
    """
    Deterministic mass-conserving reconstruction, shared verbatim by
    encoder and decoder so bitwise residuals line up.

    Each site vector is rescaled so the energy of its reconstructed
    group equals the recorded conserved mass:
        count * ||alpha * site/rg_scale||^2 = mass
    """
    base = site_points / rg_scale  # (S, 24)
    base_energy = np.sum(base**2, axis=1)  # (S,)
    denom = np.maximum(site_count * base_energy, 1e-30)
    alpha = np.sqrt(np.maximum(site_mass, 0.0) / denom)  # (S,)
    scaled = (base * alpha[:, None]).astype(np.float32)
    return scaled[assignments]


def deconsolidate(data: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
    regime, truth, debt, off = _parse_header(data)
    if regime != REGIME_CONSOLIDATE:
        raise ValueError("archive is not consolidate-regime")

    n, n_sites, rg_scale, z_alpha, kept = struct.unpack_from("<IIdIB", data, off)
    off += struct.calcsize("<IIdIB")
    coded, off = _unpack_stream(data, off)
    mass_blob, off = _unpack_stream(data, off)
    residual_blob, off = _unpack_stream(data, off)
    addr_blob, off = _unpack_stream(data, off)

    dec = RangeDecoder(coded)
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    m_z = AdaptiveModel(z_alpha)
    g_idx = np.empty(n_sites, dtype=np.int64)
    case = np.empty(n_sites, dtype=np.int64)
    for i in range(n_sites):
        g_idx[i] = m_g.decode(dec)
        case[i] = m_c.decode(dec)
    zz = np.empty(n_sites * 24, dtype=np.int64)
    for i in range(n_sites * 24):
        zz[i] = m_z.decode(dec)
    z = _unzigzag(zz).reshape(n_sites, 24)
    m_a = AdaptiveModel(max(n_sites, 2))
    assignments = np.array([m_a.decode(dec) for _ in range(n)], dtype=np.int64)

    site_mass = np.frombuffer(mass_blob, dtype=np.float32).astype(np.float64)
    site_count = np.bincount(assignments, minlength=n_sites).astype(np.int64)

    site_points = leech_index_reconstruct(g_idx, case, z)
    rows32 = _mass_scaled_recon(
        site_points, site_mass, site_count, assignments, rg_scale
    )

    if kept:
        xor = np.frombuffer(zlib.decompress(residual_blob), dtype=np.uint32).reshape(
            n, 24
        )
        rows32 = (rows32.view(np.uint32) ^ xor).view(np.float32)

    member_ids = zlib.decompress(addr_blob).decode().split("\n") if addr_blob else []

    verified = _sha(rows32.tobytes()) == debt.source_sha256
    mass = float(np.sum(rows32.astype(np.float64) ** 2))

    return rows32, {
        "truth_status": TRUTH_NAMES[truth],
        "verified": verified,
        "n_sites": int(n_sites),
        "collapse_ratio": n / max(n_sites, 1),
        "conserved_mass_in": debt.conserved_mass,
        "conserved_mass_out": mass,
        "member_ids": member_ids,
    }


# ======================================================================
# Unified facade
# ======================================================================


class KGCCompressor:
    """
    The unified Kaleidoscope Geometric Codec.

    >>> kgc = KGCCompressor()
    >>> blob = kgc.compress(b"some data")          # bytes -> lossless
    >>> data, info = kgc.decompress(blob)
    >>> blob = kgc.compress_tensor(weight_matrix)  # tensors -> lattice VQ
    >>> blob = kgc.consolidate(embedding_rows)     # rows -> RG collapse
    """

    def compress(self, data: bytes, mode: str = "geometric") -> bytes:
        return compress_bytes(data, mode)

    def decompress(self, blob: bytes) -> Tuple[bytes, Dict[str, Any]]:
        return decompress_bytes(blob)

    def compress_tensor(self, w: np.ndarray, scale: float = 4.0, **kw) -> bytes:
        return compress_tensor(w, scale=scale, **kw)

    def decompress_tensor(self, blob: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        return decompress_tensor(blob)

    def compress_tensor_progressive(self, w: np.ndarray, **kw) -> bytes:
        """
        Gain/shape progressive lattice encoding (see core/klc.py).

        Separates per-block magnitude from direction before lattice-
        quantizing, then stores truncatable residual refinement layers.
        Prefer this over compress_tensor on data with block-to-block
        magnitude variation (real weight tensors, embeddings) - see
        benchmarks/benchmark_klc.py for the measured trade-off.
        """
        from e8zip.core import klc

        return klc.encode(w, **kw)

    def decompress_tensor_progressive(
        self, blob: bytes, up_to_layer: Optional[int] = None
    ) -> np.ndarray:
        """Decode a compress_tensor_progressive archive, optionally truncated."""
        from e8zip.core import klc

        return klc.decode(blob, up_to_layer=up_to_layer)

    @staticmethod
    def tensor_progressive_layer_sizes(blob: bytes) -> Dict[str, int]:
        """Bytes needed to decode at each progressive truncation point."""
        from e8zip.core import klc

        return klc.layer_sizes(blob)

    def consolidate(self, rows: np.ndarray, **kw) -> bytes:
        return consolidate(rows, **kw)

    def deconsolidate(self, blob: bytes) -> Tuple[np.ndarray, Dict[str, Any]]:
        return deconsolidate(blob)

    @staticmethod
    def inspect(blob: bytes) -> Dict[str, Any]:
        """Read an archive's debt record without decoding the payload."""
        regime, truth, debt, _ = _parse_header(blob)
        return {
            "regime": {1: "bytes", 2: "tensor", 3: "consolidate"}.get(regime),
            "truth_status": TRUTH_NAMES.get(truth),
            "source_length": debt.source_length,
            "source_sha256": debt.source_sha256.hex(),
            "conserved_mass": debt.conserved_mass,
            "member_count": debt.member_count,
            "archive_length": len(blob),
        }
