"""
KLC - Kaleidoscope Layered Codec (gain/shape progressive lattice VQ)

Ported and extended from the Glass Network KMind's KLC2
(packages/kdot_v2/lattice_codec.py, glass_windows branch), which
separates a block's magnitude (gain) from its direction (shape) before
lattice-quantizing the shape, then refines the residual in progressive
int8 layers - classic gain-shape / successive-refinement VQ, the same
idea behind speech codecs' multi-stage residual VQ.

Why gain/shape beats the flat pipeline in core/kgc.py on real weights:
a global scale (one sigma over a whole 128-wide Hadamard row) can't
track block-to-block magnitude variation. Real weight tensors have it
in spades - the Glass Network's own real-model benchmark shows shell
occupancy (== magnitude) varying by >15% between tensors in the same
model (word_embeddings mean_shell 92.95 vs token_type_embeddings
80.47, turbo_leech_rate_distortion.json). A per-24D-block gain removes
that variance from the direction-coding problem before the lattice
ever sees it.

Two upgrades over the original KLC2:

  1. The base layer uses the EXACT Leech nearest-point decoder
     (core/leech_lattice.py) instead of KLC2's cheap sign-only Golay
     coset - a strictly better zeroth-order fit before any residual
     refinement even starts.
  2. Every stream (gain, base index, each residual layer) is entropy
     coded with an adaptive range-coder model instead of stored as raw
     bytes - KLC2 never entropy-coded anything. Layers stay
     independently truncatable because each is its own length-prefixed
     coded segment.

Progressive decode: reconstruction may stop at any residual layer -
base only (coarsest, smallest), or +k layers (successive refinement
toward the archive's full fidelity). This is a genuinely new
capability versus the flat Hadamard->Leech path in core/kgc.py, which
is all-or-nothing.
"""

import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from e8zip.core import transforms
from e8zip.core.entropy import AdaptiveModel, RangeDecoder, RangeEncoder
from e8zip.core.leech_lattice import (
    batch_nearest_leech_point,
    leech_index_decompose,
    leech_index_reconstruct,
)

MAGIC = b"KLC3"
VERSION = 1
BLOCK_DIM = 24

_STEP0 = 0.0625  # layer-0 residual quantizer step, halved each layer
_REFINE = 0.5
_EPS = 1e-12
_ZIGZAG_RANGE = 255  # int8 residual, zigzag-mapped to [0, 254]


def _zigzag8(z: np.ndarray) -> np.ndarray:
    return np.where(z >= 0, 2 * z, -2 * z - 1).astype(np.int64)


def _unzigzag8(u: np.ndarray) -> np.ndarray:
    return np.where(u % 2 == 0, u // 2, -(u + 1) // 2).astype(np.int64)


def _gain_quantize(g: np.ndarray, gain_k: float) -> np.ndarray:
    # Byte-resolution, matching KLC2: the range coder's per-model alphabet
    # is capped by BOT (core/entropy.py) with headroom assumptions that
    # break down near that ceiling, so gain stays one byte like every
    # other stream here.
    return np.clip(np.round(np.log1p(np.maximum(g, 0.0)) * gain_k), 0, 255).astype(
        np.int64
    )


def _gain_dequantize(q: np.ndarray, gain_k: float) -> np.ndarray:
    return np.expm1(q.astype(np.float64) / gain_k)


@dataclass
class KLCInfo:
    n_blocks: int
    n_layers: int
    bytes_per_layer: List[int]
    truth_status: str = "reconstruction"


def encode(
    w: np.ndarray,
    n_layers: int = 2,
    shape_scale: float = 48.0,
    gain_k: float = 128.0,
    transform: str = "hadamard",
    transform_dim: int = 128,
    seed: int = 1729,
) -> bytes:
    """
    Gain/shape progressive lattice encoding of a tensor.

    Pipeline: flatten -> incoherence transform (over transform_dim-wide
    rows) -> split into 24-D blocks -> per-block gain/shape separation
    -> exact Leech CVP of the (scaled) shape -> progressive int8
    residual layers, each entropy-coded as its own truncatable segment.

    The transform padding (to a multiple of transform_dim) and the
    block padding (to a multiple of 24) are independent - transform_dim
    need not be a multiple of 24.

    Args:
        n_layers: number of progressive residual layers to store.
        shape_scale: internal fixed-point scale applied to the
            unit-norm shape vector before lattice-snapping (shape
            vectors have norm 1 by construction; this brings them into
            the Leech lattice's natural spacing). Tune per data - see
            benchmarks/benchmark_klc.py.
        transform_dim: row width for the incoherence transform. The
            'hadamard' transform requires this to be a power of 2;
            'rotation' and 'none' accept any positive value.
    """
    if transform == "hadamard" and transform_dim & (transform_dim - 1) != 0:
        raise ValueError("hadamard transform requires transform_dim to be a power of 2")
    w = np.asarray(w, dtype=np.float64)
    orig_shape = w.shape
    flat = w.ravel()
    n = flat.size

    pad_t = (-n) % transform_dim
    padded = np.concatenate([flat, np.zeros(pad_t)])
    rows = padded.reshape(-1, transform_dim)
    inc = transforms.forward(rows, transform, seed).ravel()  # length n + pad_t

    pad_b = (-inc.size) % BLOCK_DIM
    inc_padded = np.concatenate([inc, np.zeros(pad_b)])
    n_blocks = inc_padded.size // BLOCK_DIM
    blocks = inc_padded.reshape(n_blocks, BLOCK_DIM)

    gain = np.linalg.norm(blocks, axis=1)  # (n_blocks,)
    safe_gain = np.where(gain > _EPS, gain, 1.0)
    shape = blocks / safe_gain[:, None]
    shape[gain <= _EPS] = 0.0

    gain_q = _gain_quantize(gain, gain_k)

    # Base layer: exact Leech CVP of the scaled shape.
    scaled_shape = shape * shape_scale
    points, _, _ = batch_nearest_leech_point(scaled_shape)
    g_idx, case, z = leech_index_decompose(points)
    coarse = points / shape_scale  # back in shape-space

    # Entropy-code gain + base indices.
    enc = RangeEncoder()
    m_gain = AdaptiveModel(256)
    for q in gain_q:
        m_gain.encode(enc, int(q))
    gain_stream = enc.finish()

    enc = RangeEncoder()
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    zz_base = _zigzag8(z.ravel())
    z_alpha = int(zz_base.max()) + 1 if zz_base.size else 1
    m_z = AdaptiveModel(max(z_alpha, 2))
    for i in range(n_blocks):
        m_g.encode(enc, int(g_idx[i]))
        m_c.encode(enc, int(case[i]))
    for u in zz_base:
        m_z.encode(enc, int(u))
    base_stream = enc.finish()

    # Progressive residual layers, each its own truncatable segment.
    residual = shape - coarse
    layer_streams: List[bytes] = []
    step = _STEP0
    for _ in range(n_layers):
        q = np.clip(np.round(residual / step), -127, 127).astype(np.int64)
        zz = _zigzag8(q.ravel())
        enc = RangeEncoder()
        m_r = AdaptiveModel(_ZIGZAG_RANGE)
        for u in zz:
            m_r.encode(enc, int(u))
        layer_streams.append(enc.finish())
        residual = residual - q.astype(np.float64) * step
        step *= _REFINE

    transform_kinds = ("hadamard", "rotation", "none")
    header = (
        MAGIC
        + struct.pack("<B", VERSION)
        + struct.pack(
            "<IIddIIBB",
            n,
            n_blocks,
            shape_scale,
            gain_k,
            transform_dim,
            seed,
            len(orig_shape),
            transform_kinds.index(transform),
        )
        + struct.pack(f"<{len(orig_shape)}I", *orig_shape)
        + struct.pack("<BI", n_layers, max(z_alpha, 2))
    )

    out = bytearray(header)
    for stream in (gain_stream, base_stream, *layer_streams):
        out += struct.pack("<I", len(stream)) + stream
    return bytes(out)


def _read_header(data: bytes) -> Tuple[Dict[str, Any], int]:
    if data[:4] != MAGIC:
        raise ValueError("not a KLC3 stream")
    off = 4
    (version,) = struct.unpack_from("<B", data, off)
    off += 1
    if version != VERSION:
        raise ValueError(f"unsupported KLC version {version}")
    n, n_blocks, shape_scale, gain_k, transform_dim, seed, ndim, transform_id = (
        struct.unpack_from("<IIddIIBB", data, off)
    )
    off += struct.calcsize("<IIddIIBB")
    orig_shape = struct.unpack_from(f"<{ndim}I", data, off)
    off += 4 * ndim
    n_layers, z_alpha = struct.unpack_from("<BI", data, off)
    off += struct.calcsize("<BI")
    transform_kinds = ("hadamard", "rotation", "none")
    meta = {
        "n": n,
        "n_blocks": n_blocks,
        "shape_scale": shape_scale,
        "gain_k": gain_k,
        "transform_dim": transform_dim,
        "seed": seed,
        "orig_shape": orig_shape,
        "n_layers": n_layers,
        "z_alpha": z_alpha,
        "transform": transform_kinds[transform_id],
    }
    return meta, off


def _read_stream(data: bytes, off: int) -> Tuple[bytes, int]:
    (length,) = struct.unpack_from("<I", data, off)
    off += 4
    return data[off : off + length], off + length


def decode(data: bytes, up_to_layer: Optional[int] = None) -> np.ndarray:
    """
    Reconstruct a tensor from a KLC stream.

    up_to_layer caps how many progressive residual layers are applied
    (None = all stored layers -> full stream fidelity; 0 = base only).
    """
    meta, off = _read_header(data)
    n_blocks = meta["n_blocks"]
    n_layers = meta["n_layers"]
    k = n_layers if up_to_layer is None else max(0, min(int(up_to_layer), n_layers))

    gain_stream, off = _read_stream(data, off)
    dec = RangeDecoder(gain_stream)
    m_gain = AdaptiveModel(256)
    gain_q = np.array([m_gain.decode(dec) for _ in range(n_blocks)], dtype=np.int64)
    gain = _gain_dequantize(gain_q, meta["gain_k"])

    base_stream, off = _read_stream(data, off)
    dec = RangeDecoder(base_stream)
    m_g = AdaptiveModel(4096)
    m_c = AdaptiveModel(2)
    m_z = AdaptiveModel(meta["z_alpha"])
    g_idx = np.empty(n_blocks, dtype=np.int64)
    case = np.empty(n_blocks, dtype=np.int64)
    for i in range(n_blocks):
        g_idx[i] = m_g.decode(dec)
        case[i] = m_c.decode(dec)
    zz = np.empty(n_blocks * BLOCK_DIM, dtype=np.int64)
    for i in range(n_blocks * BLOCK_DIM):
        zz[i] = m_z.decode(dec)
    z = _unzigzag8(zz).reshape(n_blocks, BLOCK_DIM)

    points = leech_index_reconstruct(g_idx, case, z)
    shape = points / meta["shape_scale"]

    step = _STEP0
    for layer in range(n_layers):
        layer_stream, off = _read_stream(data, off)
        if layer < k:
            dec = RangeDecoder(layer_stream)
            m_r = AdaptiveModel(_ZIGZAG_RANGE)
            zzr = np.array(
                [m_r.decode(dec) for _ in range(n_blocks * BLOCK_DIM)], dtype=np.int64
            )
            q = _unzigzag8(zzr).reshape(n_blocks, BLOCK_DIM)
            shape = shape + q.astype(np.float64) * step
        step *= _REFINE

    blocks = shape * gain[:, None]
    inc_padded = blocks.ravel()  # length n_blocks * BLOCK_DIM (block-padded)

    # Trim the block-padding tail back to the transform-padded length
    # (n + pad_t) before inverting the transform - the two paddings are
    # independent, mirroring encode()'s two-stage pad.
    transform_dim = meta["transform_dim"]
    n = meta["n"]
    pad_t = (-n) % transform_dim
    inc = inc_padded[: n + pad_t]
    rows = inc.reshape(-1, transform_dim)
    flat = transforms.inverse(rows, meta["transform"], meta["seed"]).ravel()[:n]

    return flat.reshape(meta["orig_shape"]).astype(np.float32)


def layer_sizes(data: bytes) -> Dict[str, int]:
    """Cumulative byte offset at each truncation point (base, +1, +2, ...)."""
    meta, off = _read_header(data)
    sizes = {}
    _, off = _read_stream(data, off)  # gain
    _, off = _read_stream(data, off)  # base
    sizes["base"] = off
    for layer in range(meta["n_layers"]):
        _, off = _read_stream(data, off)
        sizes[f"layer_{layer + 1}"] = off
    sizes["full"] = len(data)
    return sizes


def reconstruction_snr_db(orig: np.ndarray, recon: np.ndarray) -> float:
    orig = np.asarray(orig, dtype=np.float64)
    recon = np.asarray(recon, dtype=np.float64)
    err = recon - orig
    mse = float(np.mean(err * err))
    sig = float(np.mean(orig**2))
    if sig <= _EPS or mse <= _EPS:
        return 0.0
    import math

    return 10.0 * math.log10(sig / mse)
