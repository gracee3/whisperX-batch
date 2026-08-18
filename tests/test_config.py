from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config_utils
from loader import load_entrypoint


TRANSCRIBE = load_entrypoint("transcribe", "whisperx_batch_transcribe_config_test")
BENCHMARK = load_entrypoint(
  "whisperx-benchmark",
  "whisperx_batch_benchmark_config_test",
)


class ConfigUtilsTests(unittest.TestCase):
  def test_parse_bool_accepts_explicit_spellings(self) -> None:
    for value in (True, 1, "true", "YES", " enabled "):
      with self.subTest(value=value):
        self.assertIs(config_utils.parse_bool(value), True)
    for value in (False, 0, "false", "NO", " disabled "):
      with self.subTest(value=value):
        self.assertIs(config_utils.parse_bool(value), False)
    for value in (None, "maybe", object()):
      with self.subTest(value=value):
        self.assertIsNone(config_utils.parse_bool(value))

  def test_choose_value_precedence(self) -> None:
    section = {"value": "config"}
    self.assertEqual(config_utils.choose_value(section, "cli", "value", "default"), "cli")
    self.assertEqual(config_utils.choose_value(section, None, "value", "default"), "config")
    self.assertEqual(config_utils.choose_value({}, None, "value", "default"), "default")

  def test_choose_bool_precedence_and_invalid_config(self) -> None:
    self.assertFalse(config_utils.choose_bool({"flag": True}, False, "flag", True))
    self.assertTrue(config_utils.choose_bool({"flag": "yes"}, None, "flag", False))
    self.assertFalse(config_utils.choose_bool({"flag": "unknown"}, None, "flag", False))

  def test_load_toml_config_and_missing_file(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      self.assertEqual(config_utils.load_toml_config(str(root / "missing.toml")), {})
      path = root / "config.toml"
      path.write_text('[transcribe]\nlanguage = "en"\n', encoding="utf-8")
      self.assertEqual(
        config_utils.load_toml_config(str(path))["transcribe"]["language"],
        "en",
      )


class DefaultResolutionTests(unittest.TestCase):
  def parse_transcribe(self, *args: str):
    with mock.patch.object(sys, "argv", ["transcribe", *args]):
      return TRANSCRIBE.parse_args()

  def parse_benchmark(self, *args: str):
    with mock.patch.object(sys, "argv", ["whisperx-benchmark", *args]):
      return BENCHMARK.parse_args()

  def test_transcribe_no_config_uses_portable_safe_defaults(self) -> None:
    values = TRANSCRIBE.apply_defaults(self.parse_transcribe("--no-config"))
    self.assertEqual(values["output_dir"], "output")
    self.assertEqual(values["model"], "/models/faster-whisper-large-v3")
    self.assertEqual(values["docker_cache"], "~/.cache/whisperx-batch")
    self.assertFalse(values["diarize"])
    self.assertEqual(values["diarize_model"], "")

  def test_transcribe_cli_overrides_config(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      config = Path(temp_dir) / "config.toml"
      config.write_text(
        "[transcribe]\n"
        'language = "fr"\n'
        "batch_size = 4\n"
        "diarize = true\n",
        encoding="utf-8",
      )
      args = self.parse_transcribe(
        "--config",
        str(config),
        "--language",
        "en",
        "--batch-size",
        "8",
        "--no-diarize",
      )
      values = TRANSCRIBE.apply_defaults(args)
      self.assertEqual(values["language"], "en")
      self.assertEqual(values["batch_size"], "8")
      self.assertFalse(values["diarize"])

  def test_benchmark_resolves_relative_transcribe_config(self) -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
      root = Path(temp_dir)
      config = root / "benchmark.toml"
      config.write_text(
        "[benchmark]\n"
        'dataset = "dataset"\n'
        'output_root = "results"\n'
        'transcribe_config = "transcribe.toml"\n'
        "batch_size = 7\n",
        encoding="utf-8",
      )
      values = BENCHMARK.apply_defaults(self.parse_benchmark("--config", str(config)))
      self.assertEqual(values.batch_size, 7)
      self.assertEqual(values.transcribe_config, str(root / "transcribe.toml"))
      self.assertEqual(values.output_root, "results")

  def test_benchmark_rejects_missing_dataset(self) -> None:
    args = self.parse_benchmark("--config", "/definitely/missing/config.local.toml")
    with self.assertRaisesRegex(SystemExit, "dataset is required"):
      BENCHMARK.apply_defaults(args)

  def test_benchmark_rejects_negative_integer(self) -> None:
    args = self.parse_benchmark(
      "--config",
      "/definitely/missing/config.local.toml",
      "--dataset",
      "/data",
      "--limit",
      "-1",
    )
    with self.assertRaisesRegex(SystemExit, "must be >= 0"):
      BENCHMARK.apply_defaults(args)


if __name__ == "__main__":
  unittest.main()
