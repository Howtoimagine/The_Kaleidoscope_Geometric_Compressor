import sys
from pathlib import Path
import importlib.util

_root = Path(__file__).resolve().parent
if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)

import e8zip.core.kgc as kgc
import time

def test():
    data = b"hello world from qwen! this is a test of the llm compressor socket."
    print("Compressing...")
    t0 = time.time()
    comp = kgc.KGCCompressor()
    blob = comp.compress(data, mode="predictor:llama-qwen")
    t1 = time.time()
    print("Compressed length:", len(blob), "Time:", t1 - t0)
    print("Decompressing...")
    out, info = comp.decompress(blob)
    t2 = time.time()
    print("Output:", out)
    print("Time:", t2 - t1)
    assert out == data
    print("Success!")

if __name__ == "__main__":
    test()
