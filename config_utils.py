"""Shared helper utilities for config loading and common parsing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional

import tomllib


def parse_bool(value: object) -> Optional[bool]:
  if isinstance(value, bool):
    return value
  if value is None:
    return None
  if isinstance(value, int):
    return bool(value)
  if isinstance(value, str):
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on", "enabled"}:
      return True
    if lowered in {"0", "false", "no", "off", "disabled"}:
      return False
  return None


def load_toml_config(path: str) -> Dict[str, object]:
  config_path = Path(path).expanduser()
  if not config_path.exists():
    return {}

  with config_path.open("rb") as f:
    config = tomllib.load(f)

  return config if isinstance(config, dict) else {}


def choose_value(
  section: Mapping[str, object],
  cli_value: object,
  cfg_key: str,
  default: object,
) -> object:
  if cli_value is not None and cli_value != "":
    return cli_value
  if not isinstance(section, Mapping):
    return default

  cfg_value = section.get(cfg_key)
  if cfg_value is not None:
    return cfg_value
  return default


def choose_bool(
  section: Mapping[str, object],
  cli_value: Optional[bool],
  cfg_key: str,
  default: bool,
) -> bool:
  if cli_value is not None:
    return bool(cli_value)
  if not isinstance(section, Mapping):
    return default

  parsed_cfg = parse_bool(section.get(cfg_key))
  return parsed_cfg if parsed_cfg is not None else default
