#!/usr/bin/env python3
"""Clean one or more audio files for WhisperX ingestion.

This script applies the same ffmpeg processing chain used by the transcribe
cleaning step, writing a cleaned WAV file next to each input unless an output
directory is provided.

Examples:
  python3 scripts/clean_audio.py /home/emmy/Music/260312_0906.wav
  python3 scripts/clean_audio.py /tmp/a.wav /tmp/b.flac --overwrite
  python3 scripts/clean_audio.py /tmp/meeting1.mp3 /tmp/meeting2.wav \
    --output-dir /tmp/clean-audio --jobs 4
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DEFAULT_AUDIO_FILTER = "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,loudnorm=I=-16:LRA=11:TP=-2"


@dataclass
class Stats:
  converted: int = 0
  skipped: int = 0
  failed: int = 0


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Clean one or more source files with the whisperx ffmpeg chain."
  )
  parser.add_argument(
    "files",
    nargs="+",
    help="One or more input files to clean.",
  )
  parser.add_argument(
    "--audio-filter",
    dest="audio_filter",
    default=DEFAULT_AUDIO_FILTER,
    help="ffmpeg -af filter chain to apply.",
  )
  parser.add_argument(
    "--sample-rate",
    dest="sample_rate",
    type=int,
    default=16000,
    help="Output sample rate.",
  )
  parser.add_argument(
    "--channels",
    type=int,
    default=1,
    help="Output channel count.",
  )
  parser.add_argument(
    "--sample-fmt",
    default="s16",
    help="Output sample format.",
  )
  parser.add_argument(
    "--jobs",
    type=int,
    default=max(1, (os.cpu_count() or 1)),
    help="Number of ffmpeg processes to run in parallel.",
  )
  parser.add_argument(
    "--suffix",
    default="_clean",
    help="Suffix for output filename before .wav extension.",
  )
  parser.add_argument(
    "--output-dir",
    default="",
    help="Optional output directory (default: alongside input files).",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite output files if they already exist.",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Show planned output paths without running ffmpeg.",
  )
  return parser.parse_args()


def iter_inputs(files: Iterable[str]) -> List[Path]:
  paths: List[Path] = []
  for entry in files:
    path = Path(entry).expanduser().resolve()
    if not path.is_file():
      raise SystemExit(f"ERROR: input file not found: {entry}")
    paths.append(path)
  return paths


def build_output_path(
  source: Path,
  suffix: str,
  output_dir: str,
) -> Path:
  if output_dir:
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
  else:
    out_dir = source.parent

  return out_dir / f"{source.stem}{suffix}.wav"


def convert_one(
  source: str,
  args: argparse.Namespace,
) -> tuple[str, str, str, int]:
  source_path = Path(source)
  output_path = build_output_path(source_path, args.suffix, args.output_dir)

  if output_path.exists() and not args.overwrite:
    return ("skip", str(source_path), str(output_path), 0)

  if args.dry_run:
    return ("planned", str(source_path), str(output_path), 0)

  cmd = [
    "ffmpeg",
    "-hide_banner",
    "-loglevel",
    "error",
    "-y",
    "-i",
    str(source_path),
    "-ac",
    str(args.channels),
    "-ar",
    str(args.sample_rate),
    "-sample_fmt",
    args.sample_fmt,
  ]
  if args.audio_filter:
    cmd += ["-af", args.audio_filter]
  cmd.append(str(output_path))

  proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
  if proc.returncode != 0:
    msg = (proc.stderr.strip() or proc.stdout.strip() or "ffmpeg failed").split("\n")[0]
    return ("failed", str(source_path), msg[:300], proc.returncode)

  return ("done", str(source_path), str(output_path), 0)


def run(args: argparse.Namespace) -> int:
  sources = iter_inputs(args.files)
  if args.jobs < 1:
    raise SystemExit("ERROR: --jobs must be >= 1")
  if args.channels < 1:
    raise SystemExit("ERROR: --channels must be >= 1")
  if args.sample_rate < 1:
    raise SystemExit("ERROR: --sample-rate must be >= 1")

  print(f"Cleaning {len(sources)} file(s) using {args.jobs} ffmpeg process(es).")

  stats = Stats()
  total = len(sources)
  with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as pool:
    futures = [pool.submit(convert_one, str(source), args) for source in sources]
    for completed, future in enumerate(concurrent.futures.as_completed(futures), start=1):
      status, source_path, detail, rc = future.result()
      if status == "done":
        stats.converted += 1
        print(f"[{completed}/{total}] cleaned: {source_path} -> {detail}")
      elif status == "skip":
        stats.skipped += 1
        print(f"[{completed}/{total}] skip: {source_path} (exists: {detail})")
      elif status == "planned":
        stats.converted += 1
        print(f"[{completed}/{total}] planned: {source_path} -> {detail}")
      else:
        stats.failed += 1
        print(f"[{completed}/{total}] fail: {source_path}: {detail}")

  print(f"Done: converted={stats.converted} skipped={stats.skipped} failed={stats.failed}")
  return 1 if stats.failed else 0


if __name__ == "__main__":
  raise SystemExit(run(parse_args()))
