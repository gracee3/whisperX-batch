#!/usr/bin/env python3
"""
Prepare LibriSpeech data for local benchmarking:
download archive, extract, and optionally convert FLAC audio to WAV.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tarfile
import urllib.request
import urllib.error
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
  import tomllib
except ModuleNotFoundError:  # pragma: no cover - py3.11 fallback compatibility path
  import tomli as tomllib  # type: ignore


OPENSLR_ROOT = "https://www.openslr.org/resources/12"
DEFAULT_SUBSET = "dev-clean"
DEFAULT_ROOT = "~/Downloads/LibriSpeech"
DEFAULT_CONFIG_PATH = str(Path(__file__).resolve().parent.parent / "config.toml")


@dataclass(frozen=True)
class Settings:
  root: Path
  subset: str
  url: str
  archive: str
  convert_to_wav: bool
  convert_num_proc: int
  ffmpeg_threads: int
  keep_archive: bool
  overwrite_wav: bool
  convert_if_missing_only: bool
  skip_download: bool
  skip_extract: bool


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Prepare a local LibriSpeech subset by downloading, unpacking, and converting FLAC->WAV.",
  )
  parser.add_argument(
    "-c",
    "--config",
    default=DEFAULT_CONFIG_PATH,
    help="Path to config.toml (default: repo-root/config.toml).",
  )
  parser.add_argument(
    "--subset",
    default=None,
    help="LibriSpeech subset (for example: dev-clean, test-clean, train-clean-100).",
  )
  parser.add_argument("--root", default=None, help="Base directory for LibriSpeech data.")
  parser.add_argument("--url", default=None, help="Archive URL for the selected subset.")
  parser.add_argument(
    "--archive",
    default=None,
    help="Archive filename to write under root (defaults to URL basename).",
  )
  parser.add_argument(
    "--convert",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Convert FLAC files to WAV in-place.",
  )
  parser.add_argument(
    "--num-proc",
    type=int,
    default=None,
    help="FFmpeg workers used for FLAC->WAV conversion.",
  )
  parser.add_argument(
    "--ffmpeg-threads",
    type=int,
    default=None,
    help="Value passed as ffmpeg -threads (0 to let ffmpeg decide).",
  )
  parser.add_argument(
    "--overwrite-wav",
    action="store_true",
    help="Overwrite existing WAV files during conversion.",
  )
  parser.add_argument(
    "--keep-archive",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Keep downloaded tar archive after extraction.",
  )
  parser.add_argument(
    "--skip-download",
    action="store_true",
    help="Do not download; use existing archive if present.",
  )
  parser.add_argument(
    "--skip-extract",
    action="store_true",
    help="Skip extraction; assume files already exist.",
  )
  parser.add_argument(
    "--convert-if-missing-only",
    action="store_true",
    help="Only run conversion when no WAV files are present for the dataset.",
  )
  parser.add_argument(
    "--only-check",
    action="store_true",
    help="Only validate and print current dataset state without downloading/extracting.",
  )
  return parser.parse_args()


def read_config(path: Path) -> dict:
  if not path.exists():
    return {}
  try:
    with path.open("rb") as f:
      return tomllib.load(f)
  except Exception as exc:
    raise SystemExit(f"ERROR: failed to read config {path}: {exc}") from exc


def bool_from_config(raw: object, default: bool) -> bool:
  if raw is None:
    return default
  if isinstance(raw, bool):
    return raw
  return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def resolve_settings(args: argparse.Namespace) -> Settings:
  cfg = read_config(Path(args.config).expanduser())
  section = cfg.get("librispeech", {}) if isinstance(cfg.get("librispeech", {}), dict) else {}

  root = Path(
    args.root or section.get("root", DEFAULT_ROOT)
  ).expanduser()
  subset = args.subset or section.get("subset", DEFAULT_SUBSET)
  url = args.url
  if not url:
    url = section.get("url", f"{OPENSLR_ROOT}/{subset}.tar.gz")
  archive = args.archive or section.get("archive", Path(url).name)

  convert_to_wav = bool_from_config(
    section.get("convert_to_wav"),
    args.convert if args.convert is not None else True,
  )
  if args.convert is not None:
    convert_to_wav = args.convert

  convert_num_proc = args.num_proc if args.num_proc is not None else int(section.get("convert_num_proc", 8))
  ffmpeg_threads = args.ffmpeg_threads if args.ffmpeg_threads is not None else int(section.get("ffmpeg_threads", 0))
  keep_archive = bool_from_config(section.get("keep_archive"), True)
  if args.keep_archive is not None:
    keep_archive = args.keep_archive
  overwrite_wav = bool(args.overwrite_wav)
  convert_if_missing_only = bool(args.convert_if_missing_only)
  skip_download = bool(args.skip_download)
  skip_extract = bool(args.skip_extract)

  if convert_num_proc < 1:
    raise SystemExit("ERROR: --num-proc must be >= 1")
  if ffmpeg_threads < 0:
    raise SystemExit("ERROR: --ffmpeg-threads must be >= 0")
  if " " in subset:
    raise SystemExit("ERROR: subset should not contain spaces.")

  return Settings(
    root=root,
    subset=subset,
    url=url,
    archive=archive,
    convert_to_wav=convert_to_wav,
    convert_num_proc=convert_num_proc,
    ffmpeg_threads=ffmpeg_threads,
    keep_archive=keep_archive,
    overwrite_wav=overwrite_wav,
    convert_if_missing_only=convert_if_missing_only,
    skip_download=skip_download,
    skip_extract=skip_extract,
  )


def resolve_dataset_dir(root: Path, subset: str) -> Optional[Path]:
  candidate = root / subset
  if candidate.is_dir():
    return candidate
  candidate = root / "LibriSpeech" / subset
  if candidate.is_dir():
    return candidate
  return None


def download(url: str, destination: Path, skip_if_present: bool) -> Path:
  if destination.exists() and destination.stat().st_size > 0:
    if skip_if_present:
      print(f"[download] archive already present: {destination}")
      return destination
    print(f"[download] skipping because file exists (use --overwrite behavior by deleting first): {destination}")
  print(f"[download] fetching {url}")
  destination.parent.mkdir(parents=True, exist_ok=True)

  try:
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
      while True:
        chunk = response.read(1024 * 1024)
        if not chunk:
          break
        out.write(chunk)
  except urllib.error.URLError as exc:
    raise SystemExit(f"ERROR: failed to download {url}: {exc}") from exc

  return destination


def extract(archive: Path, root: Path, skip_if_present: bool) -> None:
  if skip_if_present:
    print(f"[extract] skipping extraction due to --skip-extract: {archive}")
    return
  try:
    with tarfile.open(archive, mode="r:gz") as tar:
      tar.extractall(path=str(root), filter="data")
  except Exception as exc:
    raise SystemExit(f"ERROR: failed to extract {archive}: {exc}") from exc
  print(f"[extract] complete: {archive} -> {root}")


def run_conversion(dataset_dir: Path, settings: Settings) -> int:
  converter_script = Path(__file__).resolve().parent / "convert_flac_to_wav.py"
  if not converter_script.exists():
    print(
      "ERROR: converter script missing: scripts/convert_flac_to_wav.py",
      file=sys.stderr,
    )
    return 2

  cmd = [
    sys.executable,
    str(converter_script),
    "--root",
    str(dataset_dir),
    "--ext",
    "flac",
    "--num-proc",
    str(settings.convert_num_proc),
    "--ffmpeg-threads",
    str(settings.ffmpeg_threads),
  ]
  if settings.overwrite_wav:
    cmd.append("--overwrite")
  proc = subprocess.run(cmd, text=True)
  return proc.returncode


def has_files(path: Path, suffix: str) -> int:
  return sum(1 for _ in path.rglob(f"*.{suffix}"))


def main() -> int:
  args = parse_args()
  settings = resolve_settings(args)
  settings.root.mkdir(parents=True, exist_ok=True)

  dataset_dir = resolve_dataset_dir(settings.root, settings.subset)
  flac_count = has_files(dataset_dir, "flac") if dataset_dir else 0
  wav_count = has_files(dataset_dir, "wav") if dataset_dir else 0
  print(f"[state] dataset root: {settings.root}")
  print(f"[state] subset: {settings.subset}")

  if args.only_check:
    if not dataset_dir:
      print(f"[state] NOT READY: subset directory not found under {settings.root}.")
      print(f"[state] expected either: {settings.root / settings.subset} or {settings.root / 'LibriSpeech' / settings.subset}")
      return 2
    print(f"[state] dataset dir: {dataset_dir}")
    print(f"[state] flac files: {flac_count}")
    print(f"[state] wav files: {wav_count}")
    return 0

  if dataset_dir:
    print(f"[info] found existing dataset directory: {dataset_dir}")
  else:
    print(f"[info] dataset directory not found for subset '{settings.subset}'")

  if not dataset_dir and not settings.skip_download:
    archive_path = settings.root / settings.archive
    archive = download(settings.url, archive_path, skip_if_present=True)
    if settings.skip_extract:
      print(f"[info] skipping extract due to --skip-extract; using pre-existing layout if available.")
    else:
      extract(archive, settings.root, skip_if_present=False)
  elif not dataset_dir and settings.skip_download:
    print("[error] dataset missing and --skip-download was set; cannot proceed.")
    return 2
  elif settings.skip_extract:
    print("[info] --skip-extract set; assuming dataset exists as-is.")

  dataset_dir = resolve_dataset_dir(settings.root, settings.subset)
  if not dataset_dir:
    return 2

  flac_count = has_files(dataset_dir, "flac")
  wav_count = has_files(dataset_dir, "wav")
  print(f"[state] dataset dir: {dataset_dir}")
  print(f"[state] flac files: {flac_count}")
  print(f"[state] wav files: {wav_count}")

  if settings.convert_to_wav and flac_count > 0:
    if settings.convert_if_missing_only and wav_count > 0:
      print("[convert] wav files already present and --convert-if-missing-only is set; skipping conversion.")
    else:
      print(f"[convert] converting {flac_count} files to wav with {settings.convert_num_proc} process(es)")
      convert_code = run_conversion(dataset_dir, settings)
      if convert_code != 0:
        return convert_code
      wav_count = has_files(dataset_dir, "wav")
      print(f"[state] wav files after convert: {wav_count}")
  elif settings.convert_to_wav and flac_count == 0:
    print("[info] no flac files found; nothing to convert.")

  if not settings.keep_archive and not settings.skip_download:
    archive_path = settings.root / settings.archive
    if archive_path.exists():
      archive_path.unlink()
      print(f"[cleanup] removed archive: {archive_path}")

  print("[done] LibriSpeech dataset is prepared.")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
