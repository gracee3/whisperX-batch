"""Shared default-resolution helpers for repository CLI entrypoints."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

from config_utils import choose_bool, choose_value, load_toml_config


def resolve_transcribe_defaults(args: argparse.Namespace) -> Dict[str, object]:
  config: Dict[str, object] = {}
  if not args.no_config:
    config = load_toml_config(args.config)
    if not isinstance(config, dict):
      config = {}

  trans_cfg = config.get("transcribe", {}) if isinstance(config, dict) else {}
  whisper_cfg = config.get("whisper", {}) if isinstance(config, dict) else {}

  return {
    "input_dir": str(Path(choose_value(trans_cfg, args.input_dir, "input_dir", ".")).expanduser()),
    "output_dir": str(Path(choose_value(trans_cfg, args.output_dir, "output_dir", "~/transcribe")).expanduser()),
    "cuda_devices": choose_value(trans_cfg, args.cuda_devices, "cuda_devices", "0"),
    "recursive": choose_bool(trans_cfg, args.recursive, "recursive", False),
    "model": choose_value(trans_cfg, args.model, "model", "/data/models/Systran/faster-whisper-large-v3"),
    "task": choose_value(trans_cfg, args.task, "task", "transcribe"),
    "language": choose_value(trans_cfg, args.language, "language", "en"),
    "output_format": choose_value(trans_cfg, args.output_format, "output_format", "all"),
    "device": choose_value(trans_cfg, args.device, "device", "cuda"),
    "compute_type": choose_value(trans_cfg, args.compute_type, "compute_type", "float16"),
    "batch_size": choose_value(trans_cfg, args.batch_size, "batch_size", "16"),
    "max_speakers": choose_value(trans_cfg, args.max_speakers, "max_speakers", ""),
    "skip_transcribe_existing": choose_bool(
      trans_cfg,
      args.skip_transcribe_existing,
      "skip_transcribe_existing",
      False,
    ),
    "diarize": choose_bool(trans_cfg, args.diarize, "diarize", True),
    "docker_image": choose_value(trans_cfg, args.docker_image, "docker_image", "whisperx:torch280-cu128"),
    "docker_pull_policy": str(
      choose_value(trans_cfg, args.docker_pull_policy, "docker_pull_policy", "missing"),
    ).strip(),
    "docker_cache": choose_value(trans_cfg, args.docker_cache, "docker_cache", "/data/models/.hf-cache"),
    "align_model": choose_value(whisper_cfg, args.align_model, "align_model", "WAV2VEC2_ASR_BASE_960H"),
    "interpolate_method": choose_value(whisper_cfg, args.interpolate_method, "interpolate_method", ""),
    "no_align": choose_bool(whisper_cfg, args.no_align, "no_align", False),
    "return_char_alignments": choose_bool(whisper_cfg, args.return_char_alignments, "return_char_alignments", False),
    "vad_method": choose_value(whisper_cfg, args.vad_method, "vad_method", ""),
    "vad_onset": choose_value(whisper_cfg, args.vad_onset, "vad_onset", ""),
    "vad_offset": choose_value(whisper_cfg, args.vad_offset, "vad_offset", ""),
    "chunk_size": choose_value(whisper_cfg, args.chunk_size, "chunk_size", ""),
    "diarize_model": choose_value(whisper_cfg, args.diarize_model, "diarize_model", "/data/models/pyannote/speaker-diarization-community-1"),
    "speaker_embeddings": choose_bool(whisper_cfg, args.speaker_embeddings, "speaker_embeddings", False),
    "temperature": str(choose_value(whisper_cfg, args.temperature, "temperature", "")),
    "best_of": str(choose_value(whisper_cfg, args.best_of, "best_of", "")),
    "beam_size": str(choose_value(whisper_cfg, args.beam_size, "beam_size", "")),
    "patience": str(choose_value(whisper_cfg, args.patience, "patience", "")),
    "length_penalty": str(choose_value(whisper_cfg, args.length_penalty, "length_penalty", "")),
    "suppress_tokens": str(choose_value(whisper_cfg, args.suppress_tokens, "suppress_tokens", "")),
    "suppress_numerals": choose_bool(whisper_cfg, args.suppress_numerals, "suppress_numerals", False),
    "initial_prompt": choose_value(whisper_cfg, args.initial_prompt, "initial_prompt", ""),
    "condition_on_previous_text": str(
      choose_value(whisper_cfg, args.condition_on_previous_text, "condition_on_previous_text", ""),
    ),
    "fp16": str(choose_value(whisper_cfg, args.fp16, "fp16", "")),
    "temperature_increment_on_fallback": str(
      choose_value(
        whisper_cfg,
        args.temperature_increment_on_fallback,
        "temperature_increment_on_fallback",
        "",
      ),
    ),
    "compression_ratio_threshold": str(
      choose_value(
        whisper_cfg,
        args.compression_ratio_threshold,
        "compression_ratio_threshold",
        "",
      ),
    ),
    "logprob_threshold": str(choose_value(whisper_cfg, args.logprob_threshold, "logprob_threshold", "")),
    "no_speech_threshold": str(choose_value(whisper_cfg, args.no_speech_threshold, "no_speech_threshold", "")),
    "max_line_width": str(choose_value(whisper_cfg, args.max_line_width, "max_line_width", "")),
    "max_line_count": str(choose_value(whisper_cfg, args.max_line_count, "max_line_count", "")),
    "highlight_words": choose_bool(whisper_cfg, args.highlight_words, "highlight_words", False),
    "segment_resolution": choose_value(whisper_cfg, args.segment_resolution, "segment_resolution", ""),
    "threads": str(choose_value(whisper_cfg, args.threads, "threads", "")),
    "print_progress": str(choose_value(whisper_cfg, args.print_progress, "print_progress", "")),
    "verbose": str(choose_value(whisper_cfg, args.verbose, "verbose", "")),
    "log_level": choose_value(whisper_cfg, args.log_level, "log_level", ""),
    "hotwords": choose_value(whisper_cfg, args.hotwords, "hotwords", ""),
    "min_speakers": choose_value(whisper_cfg, args.min_speakers, "min_speakers", ""),
    "extra_args": list(args.whisper_arg) + list(args.remaining),
  }


def resolve_benchmark_defaults(args: argparse.Namespace) -> argparse.Namespace:
  config: Dict[str, object] = {}
  if args.config:
    config = load_toml_config(args.config)
    if not isinstance(config, dict):
      config = {}

  bench_cfg = config.get("benchmark", {}) if isinstance(config, dict) else {}

  def to_non_negative_int(value: object, cfg_key: str, cli_default: object) -> int:
    raw = choose_value(bench_cfg, value, cfg_key, cli_default)
    try:
      parsed = int(raw)
    except (TypeError, ValueError):
      raise SystemExit(f"ERROR: [benchmark].{cfg_key} must be an integer: {raw!r}")
    if parsed < 0:
      raise SystemExit(f"ERROR: [benchmark].{cfg_key} must be >= 0: {parsed}")
    return parsed

  defaults = dict(args.__dict__)
  defaults["dataset"] = choose_value(bench_cfg, args.dataset, "dataset", "")
  if defaults["dataset"]:
    defaults["dataset"] = str(Path(defaults["dataset"]).expanduser())
  defaults["ext"] = choose_value(bench_cfg, args.ext, "ext", "wav")
  defaults["transcribe"] = choose_value(bench_cfg, args.transcribe, "transcribe", "transcribe")
  defaults["output_root"] = str(
    Path(
      choose_value(bench_cfg, args.output_root, "output_root", "~/transcribe/temp/whisperx-benchmark"),
    ).expanduser(),
  )
  defaults["output_format"] = choose_value(bench_cfg, args.output_format, "output_format", "all")
  defaults["score_format"] = choose_value(bench_cfg, args.score_format, "score_format", "txt")
  defaults["max_files"] = to_non_negative_int(args.max_files, "max_files", 0)
  defaults["batch_size"] = to_non_negative_int(args.batch_size, "batch_size", 16)
  defaults["transcribe_config"] = str(
    choose_value(
      bench_cfg,
      args.transcribe_config,
      "transcribe_config",
      str(Path(__file__).resolve().parent / "config.toml"),
    ),
  )
  defaults["skip_transcribe_existing"] = choose_bool(
    bench_cfg,
    args.skip_transcribe_existing,
    "skip_transcribe_existing",
    False,
  )
  defaults["results_csv"] = choose_value(bench_cfg, args.results_csv, "results_csv", "")
  defaults["limit"] = to_non_negative_int(args.limit, "limit", 0)
  defaults["no_diarize"] = choose_bool(bench_cfg, args.no_diarize, "no_diarize", True)

  if not defaults["dataset"]:
    raise SystemExit(
      "ERROR: dataset is required. Set [benchmark].dataset in config.toml or pass --dataset.",
    )

  transcribe_config = str(defaults["transcribe_config"]).strip()
  if transcribe_config:
    tc_path = Path(transcribe_config).expanduser()
    if not tc_path.is_absolute():
      tc_path = Path(args.config).expanduser().resolve().parent / tc_path
    defaults["transcribe_config"] = str(tc_path)

  transcribe = str(defaults["transcribe"]).strip()
  if transcribe:
    transcribe_path = Path(transcribe).expanduser()
    if not transcribe_path.is_absolute():
      if shutil.which(transcribe):
        transcribe_path = Path(shutil.which(transcribe))
      else:
        fallback = Path(__file__).resolve().parent / transcribe
        if fallback.exists():
          transcribe_path = fallback
    defaults["transcribe"] = str(transcribe_path)

  return argparse.Namespace(**defaults)
