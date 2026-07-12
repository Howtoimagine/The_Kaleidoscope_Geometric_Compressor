"""
Extended Binary Golay Code [24,12,8]

The combinatorial heart of the Leech lattice — and therefore of the
Kaleidoscope Geometric Codec. Ported from the Glass Network KMind
(packages/kmind/golay.py, glass_windows branch) where it seals memory
boundaries; here it doubles as:

  * the coset labeling of the Leech lattice (Construction A):
    every Leech point decomposes as  y = 2g + i*1 + 4z  with g a Golay
    codeword — so 12 bits of every 24-D quantized block are literally
    a Golay message, and

  * an error-correcting layer: decode_24 heals up to 3 bit flips in any
    stored coset index, making archives partially self-repairing.

Structure: systematic G = [I_12 | B] with B the bordered QR(11) parity
matrix. B is symmetric and B^2 = I over GF(2), which the Pless
arithmetic decoder below exploits.
"""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

# Bordered quadratic-residue QR(11) parity matrix (12x12, symmetric).
_P = [
    [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
    [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1],
    [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1],
    [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1],
    [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
]

# Rows of B packed as 12-bit ints (B symmetric: rows == columns).
_B_ROWS: Tuple[int, ...] = tuple(
    sum((_P[i][j] & 1) << j for j in range(12)) for i in range(12)
)


def encode_12_to_24(message: int) -> int:
    """Encode a 12-bit message into a 24-bit extended Golay codeword."""
    m_bits = [(message >> i) & 1 for i in range(12)]

    p_bits = [0] * 12
    for i in range(12):
        s = 0
        for j in range(12):
            s ^= m_bits[j] & _P[j][i]
        p_bits[i] = s

    codeword = message
    for i in range(12):
        if p_bits[i]:
            codeword |= 1 << (i + 12)

    return codeword


def compute_syndrome(word: int) -> int:
    """Compute the 12-bit syndrome of a 24-bit word (H = [B^T | I_12])."""
    w_bits = [(word >> i) & 1 for i in range(24)]

    s_bits = [0] * 12
    for i in range(12):
        s = w_bits[i + 12]
        for j in range(12):
            s ^= w_bits[j] & _P[j][i]
        s_bits[i] = s

    syndrome = 0
    for i in range(12):
        if s_bits[i]:
            syndrome |= 1 << i

    return syndrome


def _b_times(vec: int) -> int:
    """Multiply a 12-bit vector by B over GF(2)."""
    out = 0
    for i in range(12):
        if (vec >> i) & 1:
            out ^= _B_ROWS[i]
    return out


def decode_24(received: int) -> Optional[Tuple[int, int, int]]:
    """
    Syndrome-decode a (possibly corrupted) 24-bit word.

    Returns (codeword, message, errors_corrected) when the received word
    is within Hamming distance 3 of a unique codeword, or None when the
    error pattern is uncorrectable (weight >= 4 detected). The code is
    systematic, so `message` is simply the low 12 bits of the codeword.
    """
    received &= (1 << 24) - 1
    s = compute_syndrome(received)  # s = B*e_info XOR e_parity

    error: Optional[int] = None

    # Case 1: errors confined to the parity block.
    if s.bit_count() <= 3:
        error = s << 12
    else:
        # Case 2: one info error at position i, <=2 parity errors.
        for i in range(12):
            if (s ^ _B_ROWS[i]).bit_count() <= 2:
                error = (1 << i) | ((s ^ _B_ROWS[i]) << 12)
                break

    if error is None:
        sp = _b_times(s)  # B^2 = I: if e_parity = 0 then B*s = e_info
        # Case 3: errors confined to the information block.
        if sp.bit_count() <= 3:
            error = sp
        else:
            # Case 4: one parity error at position i, <=2 info errors.
            for i in range(12):
                if (sp ^ _B_ROWS[i]).bit_count() <= 2:
                    error = (sp ^ _B_ROWS[i]) | (1 << (i + 12))
                    break

    if error is None:
        return None  # >= 4 errors: detected, not correctable

    codeword = received ^ error
    message = codeword & ((1 << 12) - 1)
    return codeword, message, error.bit_count()


@lru_cache(maxsize=1)
def all_codewords() -> np.ndarray:
    """
    All 4096 Golay codewords as a (4096, 24) int32 bit matrix.

    Row index == 12-bit message (the code is systematic), so this array
    is simultaneously the codebook and the index map: codeword_bits ->
    message is just `codeword & 0xFFF`.
    """
    out = np.zeros((4096, 24), dtype=np.int32)
    for msg in range(4096):
        cw = encode_12_to_24(msg)
        for i in range(24):
            out[msg, i] = (cw >> i) & 1
    return out


def bits_to_index(bits: np.ndarray) -> int:
    """Map a 24-length 0/1 codeword vector to its 12-bit message index."""
    word = 0
    for i in range(24):
        if bits[i]:
            word |= 1 << i
    return word & 0xFFF


# Weight enumerator of the code: {weight: count}. Used as the canonical
# distribution in the moonshine RG diagnostics and as a static prior.
WEIGHT_ENUMERATOR = {0: 1, 8: 759, 12: 2576, 16: 759, 24: 1}
