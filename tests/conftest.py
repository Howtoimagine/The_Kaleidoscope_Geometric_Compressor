import sys
from pathlib import Path
import importlib.util

_root = Path(__file__).resolve().parent.parent
if "e8zip" not in sys.modules:
    _spec = importlib.util.spec_from_file_location(
        "e8zip", _root / "__init__.py", submodule_search_locations=[str(_root)]
    )
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules["e8zip"] = _mod
    _spec.loader.exec_module(_mod)
