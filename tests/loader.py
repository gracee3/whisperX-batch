from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))


def load_entrypoint(filename: str, module_name: str) -> ModuleType:
  existing = sys.modules.get(module_name)
  if existing is not None:
    return existing

  loader = importlib.machinery.SourceFileLoader(module_name, str(ROOT / filename))
  spec = importlib.util.spec_from_loader(module_name, loader)
  if spec is None:
    raise RuntimeError(f"could not create module spec for {filename}")
  module = importlib.util.module_from_spec(spec)
  sys.modules[module_name] = module
  loader.exec_module(module)
  return module
