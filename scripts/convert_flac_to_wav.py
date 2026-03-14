#!/usr/bin/env python3
"""
Parallel in-place FLAC->WAV converter using multiple ffmpeg processes.

Examples:
  python scripts/convert_flac_to_wav.py \
    --root /home/emmy/Downloads/LibriSpeech/dev-clean \
    --num-proc 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


@dataclass
class Stats:
  converted: int = 0
  skipped: int = 0
  failed: int = 0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Convert FLAC files to WAV in-place using parallel ffmpeg processes."
  )
  parser.add_argument("--root", default=".", help="Root directory to scan for audio files.")
  parser.add_argument(
    "--num-proc",
    type=int,
    default=max(1, os.cpu_count() or 1),
    help="Number of concurrent ffmpeg processes.",
  )
  parser.add_argument(
    "--ffmpeg-threads",
    type=int,
    default=0,
    help="ffmpeg -threads value for each process (0 = use ffmpeg default).",
  )
  parser.add_argument("--sample-rate", type=int, default=16000, help="Output sample rate.")
  parser.add_argument("--channels", type=int, default=1, help="Output channel count.")
  parser.add_argument("--sample-fmt", default="s16", help="Output sample format.")
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing WAV files.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show what would be converted, but do not run ffmpeg.",
  )
  parser.add_argument(
    "--ext",
    default="flac",
    help="Input audio extension to convert (default: flac).",
  )
  return parser.parse_args()


OK = 0
SKIP = 1
DRY = 2
FAIL = -1


def iter_inputs(root: Path, ext: str) -> Iterable[Path]:
  pattern = f"*.{ext.lstrip('.')}"
  yield from (p for p in root.rglob(pattern) if p.is_file())


def build_wav_path(src: Path) -> Path:
  return src.with_suffix(".wav")


def convert_one(src: str, args: argparse.Namespace) -> tuple[str, str, int]:
  src_path = Path(src)
  dst = build_wav_path(src_path)
  src_name = src_path.name

  if args.dry_run:
    if dst.exists() and not args.overwrite:
      return (
        "skip",
        f"{src_name} (already has {dst.name}; use --overwrite)",
        SKIP,
      )
    return ("planned", f"{src_name} -> {dst.name}", DRY)

  if dst.exists() and not args.overwrite:
    return ("skip", f"{src_name} (already has {dst.name})", SKIP)

  cmd = [
    "ffmpeg",
    "-y",
    "-v",
    "error",
    "-i",
    str(src_path),
    "-ac",
    str(args.channels),
    "-ar",
    str(args.sample_rate),
    "-sample_fmt",
    args.sample_fmt,
    str(dst),
  ]

  if args.ffmpeg_threads > 0:
    cmd.insert(4, str(args.ffmpeg_threads))
    cmd.insert(4, "-threads")

  proc = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    check=False,
  )

  if proc.returncode != 0:
    err = proc.stderr.strip() or proc.stdout.strip() or "ffmpeg failed"
    return ("failed", f"{src_name}: {err}", FAIL)
  return ("done", src_name, OK)


def run(args: argparse.Namespace) -> int:
  root = Path(args.root).resolve()
  if not root.is_dir():
    raise SystemExit(f"ERROR: root is not a directory: {root}")
  if args.num_proc < 1:
    raise SystemExit("ERROR: --num-proc must be >= 1")
  if args.ffmpeg_threads < 0:
    raise SystemExit("ERROR: --ffmpeg-threads must be >= 0")

  files: List[Path] = list(iter_inputs(root, args.ext))
  if not files:
    print(f"No *.{args.ext} files found under: {root}")
    return 0

  print(
    f"Converting {len(files)} files from .{args.ext} to .wav under {root} "
    f"using {args.num_proc} ffmpeg processes."
  )

  stats = Stats()
  total = len(files)
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_proc) as pool:
    futures = [pool.submit(convert_one, str(src), args) for src in files]
    for fut in concurrent.futures.as_completed(futures):
      status, msg, code = fut.result()
      done = stats.converted + stats.skipped + stats.failed
      done += 1
      if code == SKIP:
        stats.skipped += 1
      elif code == OK:
        stats.converted += 1
      elif code == DRY:
        stats.converted += 1
      else:
        # include failure count for non-zero returns
        stats.failed += 1
      if status == "done":
        print(f"[{done}/{total}] converted: {msg}")
      elif status == "planned":
        print(f"[{done}/{total}] would convert: {msg}")
      elif status == "skip":
        print(f"[{done}/{total}] skip: {msg}")
      else:
        print(f"[{done}/{total}] fail: {msg}")

  print(
    "Done: "
    f"converted={stats.converted} skipped={stats.skipped} failed={stats.failed}"
  )

  return 1 if stats.failed else 0


if __name__ == "__main__":
  raise SystemExit(run(parse_args()))
