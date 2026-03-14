"""Shared helpers for benchmark CLI option parsing."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Sequence


def parse_kv_items(items: Sequence[str], *, allow_multiple: bool = False) -> OrderedDict[str, List[str]]:
  parsed: OrderedDict[str, List[str]] = OrderedDict()

  for item in items:
    if "=" not in item:
      raise ValueError(f"expected key=value format: {item}")
    raw_key, raw_value = item.split("=", 1)
    key = raw_key.strip().replace("_", "-")
    if not key:
      raise ValueError(f"invalid empty key in '{item}'")
    if key in {"cuda-devices"}:
      values = [raw_value.strip()]
    else:
      values = [v.strip() for v in raw_value.split(",") if v.strip()]
    if not values:
      raise ValueError(f"empty value list for '{item}'")
    if (not allow_multiple) and key in parsed:
      raise ValueError(
        f"duplicate key '{key}' supplied to --set; pass multiple values in comma list",
      )
    parsed.setdefault(key, [])
    parsed[key].extend(values)
    if (not allow_multiple) and len(parsed[key]) != 1:
      raise ValueError(f"multiple values for '{key}' in single-valued option")

  return parsed


def whisper_arg_for_sweep(token: str, value: str) -> List[str]:
  bool_true = {"true", "1", "yes", "on"}
  bool_false = {"false", "0", "no", "off"}
  low = value.lower()
  toggles = {
    "no-align",
    "return-char-alignments",
    "diarize",
    "no-diarize",
    "suppress-numerals",
    "speaker-embeddings",
    "highlight-words",
  }
  if token in toggles:
    if low in bool_true:
      return [f"--{token}"]
    if low in bool_false:
      return []
    return [f"--{token}", value]
  if low in bool_true.union(bool_false):
    return [f"--{token}", value]
  return [f"--{token}", value]
