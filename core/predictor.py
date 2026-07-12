"""
Predictor Front-End - Neural/LLM next-byte prediction for the KGC BYTES regime

"Compression is prediction": any model that emits a probability
distribution over the next byte can drive the range coder. This module
is the socket that lets a language model (or any learned predictor) be
that model, while staying dependency-free by default.

Contract: the predictor is DETERMINISTIC and causal. Encoder and decoder
run the identical predictor over the identical byte history, so the
probability streams match exactly and no side information is stored
beyond the predictor's registry name. (Same machine, same weights, same
library versions — cross-platform float drift breaks the stream, which
the sha256 debt check will catch loudly rather than silently.)

    class MyLLM:
        def reset(self): ...
        def predict(self) -> np.ndarray:   # (256,) probs, sum ~ 1
        def update(self, byte: int): ...   # advance context by one byte

    register_predictor("my-llm", MyLLM)
    blob = kgc.compress(data, mode="predictor:my-llm")
    data, info = kgc.decompress(blob)   # instantiates "my-llm" again

The built-in reference predictor ("ngram-mix") is a dependency-free
byte-level context mixer — the architectural stand-in an actual LLM
front-end plugs in over.
"""

from typing import Callable, Dict, Protocol

import numpy as np

from e8zip.core.entropy import RangeDecoder, RangeEncoder

CDF_TOTAL = 1 << 16  # must not exceed entropy.BOT


class Predictor(Protocol):
    def reset(self) -> None: ...

    def predict(self) -> np.ndarray: ...

    def update(self, byte: int) -> None: ...


def _quantize_pmf(p: np.ndarray) -> np.ndarray:
    """
    Deterministically quantize a pmf to integer frequencies.

    Every symbol keeps freq >= 1 (stays codeable however wrong the
    predictor is) and the total stays <= CDF_TOTAL.
    """
    p = np.asarray(p, dtype=np.float64)
    p = np.maximum(p, 0.0)
    total = p.sum()
    if not np.isfinite(total) or total <= 0:
        return np.ones(256, dtype=np.int64)
    avail = CDF_TOTAL - 256
    return np.floor(p / total * avail).astype(np.int64) + 1


class PredictorByteModel:
    """Drives the range coder from any Predictor's next-byte pmf."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def compress(self, data: bytes) -> bytes:
        self.predictor.reset()
        enc = RangeEncoder()
        for b in data:
            f = _quantize_pmf(self.predictor.predict())
            cum = int(f[:b].sum())
            enc.encode(cum, int(f[b]), int(f.sum()))
            self.predictor.update(b)
        return enc.finish()

    def decompress(self, blob: bytes, n: int) -> bytes:
        self.predictor.reset()
        dec = RangeDecoder(blob)
        out = bytearray()
        for _ in range(n):
            f = _quantize_pmf(self.predictor.predict())
            cdf = np.concatenate([[0], np.cumsum(f)])
            tot = int(cdf[-1])
            slot = dec.decode_freq(tot)
            b = int(np.searchsorted(cdf, slot, side="right") - 1)
            dec.decode_update(int(cdf[b]), int(f[b]), tot)
            out.append(b)
            self.predictor.update(b)
        return bytes(out)


class NGramMixPredictor:
    """
    Reference predictor: adaptive mixture of order-0/1/2 byte statistics.

    Not a neural model — it is the deterministic, dependency-free
    stand-in that exercises the exact same socket an LLM front-end uses.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.o0 = np.ones(256, dtype=np.float64)
        self.o1: Dict[int, np.ndarray] = {}
        self.o2: Dict[int, np.ndarray] = {}
        self.c1 = 0
        self.c2 = 0

    def predict(self) -> np.ndarray:
        p = self.o0 / self.o0.sum()
        t1 = self.o1.get(self.c1)
        if t1 is not None:
            w = t1.sum() / (t1.sum() + 4.0)
            p = (1 - w) * p + w * (t1 / t1.sum())
        t2 = self.o2.get(self.c2)
        if t2 is not None:
            w = t2.sum() / (t2.sum() + 2.0)
            p = (1 - w) * p + w * (t2 / t2.sum())
        return p

    def update(self, byte: int) -> None:
        self.o0[byte] += 1
        t1 = self.o1.get(self.c1)
        if t1 is None:
            t1 = self.o1[self.c1] = np.zeros(256, dtype=np.float64)
        t1[byte] += 1
        t2 = self.o2.get(self.c2)
        if t2 is None:
            t2 = self.o2[self.c2] = np.zeros(256, dtype=np.float64)
        t2[byte] += 1
        self.c2 = ((self.c1 & 0xFF) << 8) | byte
        self.c1 = byte


class CallablePredictor:
    """
    Adapter for external models (torch, llama.cpp, ONNX, ...).

    logits_fn receives the byte history (bytes, up to context_len long)
    and returns 256 unnormalized logits or probabilities for the next
    byte. Register a factory that closes over your loaded model:

        register_predictor(
            "llama-байт", lambda: CallablePredictor(my_model_fn, 512)
        )
    """

    def __init__(self, logits_fn: Callable[[bytes], np.ndarray], context_len: int = 256):
        self.fn = logits_fn
        self.context_len = context_len
        self.history = bytearray()

    def reset(self) -> None:
        self.history = bytearray()

    def predict(self) -> np.ndarray:
        raw = np.asarray(
            self.fn(bytes(self.history[-self.context_len :])), dtype=np.float64
        )
        if raw.min() < 0 or raw.sum() <= 0:  # treat as logits
            raw = np.exp(raw - raw.max())
        return raw / raw.sum()

    def update(self, byte: int) -> None:
        self.history.append(byte)


class OnlineGRUPredictor:
    """
    NNCP-style neural predictor: a byte-level GRU trained ONLINE, during
    both compression and decompression, from a fixed seed.

    No pretrained weights ride in the archive — encoder and decoder each
    start from the same seeded initialization and take identical
    gradient steps as bytes stream through, so their probability
    streams match. This is the "learned model as program, not payload"
    version of Law 9.

    Determinism contract: torch on CPU, same machine/library versions.
    Truncated BPTT of length 1 (state detached each step). Slow —
    roughly a millisecond per byte — so this is the demonstrator for
    the LLM socket, not the fast path.
    """

    def __init__(
        self, hidden: int = 64, emb: int = 32, lr: float = 3e-3, seed: int = 0
    ) -> None:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "OnlineGRUPredictor needs PyTorch: pip install torch"
            ) from e
        self.torch = torch
        self.hidden = hidden
        self.emb_dim = emb
        self.lr = lr
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        torch = self.torch
        torch.manual_seed(self.seed)
        self.emb = torch.nn.Embedding(256, self.emb_dim)
        self.cell = torch.nn.GRUCell(self.emb_dim, self.hidden)
        self.head = torch.nn.Linear(self.hidden, 256)
        self.opt = torch.optim.Adam(
            [*self.emb.parameters(), *self.cell.parameters(), *self.head.parameters()],
            lr=self.lr,
        )
        self.h = torch.zeros(1, self.hidden)

    def predict(self) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            logits = self.head(self.h)
            p = torch.softmax(logits, dim=1)[0].double().numpy()
        # floor the pmf so one confident wrong prediction can't blow up
        p = 0.999 * p + 0.001 / 256.0
        return p

    def update(self, byte: int) -> None:
        torch = self.torch
        target = torch.tensor([byte])
        # self.h carries a one-step graph (built at the END of the previous
        # update, after that step's optimizer step), so this loss reaches
        # the head directly and the cell + embedding through h. Backward
        # must run BEFORE opt.step() mutates the saved tensors.
        logits = self.head(self.h)
        loss = torch.nn.functional.cross_entropy(logits, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        # advance state with a fresh graph over the updated parameters
        self.h = self.cell(self.emb(target), self.h.detach())


def make_llama_predictor() -> Predictor:
    """
    Creates a Predictor using a GGUF model via llama-cpp-python.
    Defaults to the Qwen 2.5 1.5B model on the F drive.
    """
    try:
        import llama_cpp
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "LlamaPredictor needs llama-cpp-python: pip install llama-cpp-python"
        ) from e

    import os
    model_path = os.environ.get(
        "KGC_LLM_PATH", r"F:\GlassNetwork_Models\qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM not found at {model_path}. Set KGC_LLM_PATH.")

    context_len = 128
    llm = llama_cpp.Llama(
        model_path=model_path,
        n_ctx=context_len,
        logits_all=True,
        verbose=False,
    )
    vocab_size = llm.n_vocab()
    first_byte_map = np.zeros(vocab_size, dtype=np.int32)
    for i in range(vocab_size):
        try:
            b = llm.detokenize([i])
            first_byte_map[i] = b[0] if len(b) > 0 else -1
        except Exception:
            first_byte_map[i] = -1

    valid_mask = first_byte_map >= 0
    valid_bytes = first_byte_map[valid_mask]

    def logits_fn(history: bytes) -> np.ndarray:
        tokens = llm.tokenize(history, add_bos=False)
        llm.reset()
        if len(tokens) == 0:
            return np.ones(256)
        llm.eval(tokens)
        next_token_logits = llm.scores[len(tokens) - 1, :]
        
        shifted = next_token_logits - np.max(next_token_logits)
        probs = np.exp(shifted)
        probs /= np.sum(probs)
        
        byte_probs = np.zeros(256, dtype=np.float64)
        np.add.at(byte_probs, valid_bytes, probs[valid_mask])
        return 0.999 * byte_probs + 0.001 / 256.0

    return CallablePredictor(logits_fn, context_len=context_len)


# --- registry: archives store the predictor NAME, both sides must have it ---

_REGISTRY: Dict[str, Callable[[], Predictor]] = {}


def register_predictor(name: str, factory: Callable[[], Predictor]) -> None:
    if not name or len(name.encode()) > 255:
        raise ValueError("predictor name must be 1..255 bytes")
    _REGISTRY[name] = factory


def get_predictor(name: str) -> Predictor:
    factory = _REGISTRY.get(name)
    if factory is None:
        raise KeyError(
            f"predictor '{name}' not registered on this side; "
            f"available: {sorted(_REGISTRY)}"
        )
    return factory()


register_predictor("ngram-mix", NGramMixPredictor)
register_predictor("gru-online", OnlineGRUPredictor)  # needs torch
register_predictor("llama-qwen", make_llama_predictor)
