# whisperx-batch

whisperx-batch is a Docker-first batch ASR stack for local environments.
It runs WhisperX transcription with optional diarization over directories of files, supports reproducible benchmark sweeps, and is tuned for deterministic local/offline execution.

## Stack

Key versions and base images are pinned in `Dockerfile.whisperx-torch280-cu128`:

- Ubuntu: `ubuntu22.04`
- Python: `3.11`
- CUDA runtime: `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04`
- torch: `2.8.0+cu128`
- torchaudio: `2.8.0+cu128`
- triton: `>=3.3.0`
- [CTranslate2](https://github.com/OpenNMT/CTranslate2): `4.7.1`
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper): `1.2.1`
- [openai/whisper](https://github.com/openai/whisper)
- [WhisperX](https://github.com/m-bain/whisperx): `3.8.2`
- pyannote-audio: `4.0.4`
- transformers: `>=4.48.0`

## What it does

- Transcribe directories or file trees with `transcribe` (Docker-only).
- Run parameter sweeps and WER scoring with `whisperx-benchmark`.
- Cache outputs so reruns skip completed files.
- Keep model and cache directories on disk and run without remote model downloads.

## Install / Setup

```bash
make install
make preflight
```

Install command is created in:

- `~/.local/bin` (default)
- custom via `make install BINDIR=/usr/local/bin`

## CLI defaults and config

Both CLIs default to repo `config.toml`.

```text
-c, --config PATH         TOML config override
--no-config                ignore config defaults
```

Benchmark and transcribe configs are now centralized in the same file.

Run benchmark with an explicit config path:

```bash
whisperx-benchmark --dataset ./dataset --config /path/to/config.toml
```

## Required local assets

- Whisper model: `/data/models/Systran/faster-whisper-large-v3`
- Diarization model: `/data/models/pyannote/speaker-diarization-community-1`
- HF/torch cache: `/data/models/.hf-cache`

Optional alignment artifacts:

- `/data/models/.hf-cache/nltk_data/tokenizers/punkt_tab`
- `/data/models/.hf-cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth`

Preseed and validate caches:

```bash
python scripts/preseed_whisperx_cache.py --cache-dir /data/models/.hf-cache
test -f /data/models/.hf-cache/nltk_data/tokenizers/punkt_tab/english/README
```

## Recommended run pattern

Transcribe:

```bash
transcribe \
  --input-dir /path/to/audio \
  --output-dir /path/to/out \
  --cuda-devices 1 \
  --batch-size 16 \
  --no-diarize \
  --skip-transcribe-existing
```

Benchmark:

```bash
whisperx-benchmark \
  --dataset /path/to/dataset \
  --ext wav \
  --output-root /path/to/out \
  --results-csv /path/to/out/results.csv \
  --output-format json \
  --score-format json \
  --set no-diarize=true \
  --set beam_size=1 \
  --set best_of=1 \
  --set temperature=0.0 \
  --set suppress_numerals=false
```

## Core options

Use `transcribe --help` and `whisperx-benchmark --help` for full option surfaces.
Most-used high-impact knobs:

- `--batch-size`
- `--cuda-devices`
- `--model`
- `--diarize` / `--no-diarize`
- `--beam-size`
- `--best-of`
- `--temperature`
- `--suppress-numerals`
- `--whisper-arg` / `--` passthrough

The benchmark runner maps `--sweep` axes to transcribe args with correct boolean handling for
`no-diarize`, `diarize`, and other on/off toggles.

## Behavior notes

- Whisper and alignment run inside a single process per `transcribe` invocation.
- By design, `transcribe` processes are single-process and single-GPU visible per run (`--cuda-devices`).
- For multi-GPU throughput, launch multiple independent invocations across shards.
- No ffmpeg transcoding is performed inside `transcribe`; prepare files using the helper scripts first.
- WER sweeps write run-level metrics to `results.csv`, including:
  - `avg_wer`
  - `files_per_second`
  - `seconds_per_file`
  - `input_toks_per_sec_*`
  - `decode_toks_per_sec_*`
  - `encode_toks_per_sec_*`

## Helper scripts

- `scripts/setup_librispeech_dataset.py`  
  one-step LibriSpeech download and conversion.
- `scripts/convert_flac_to_wav.py`  
  explicit FLAC-to-WAV conversion.
- `scripts/clean_audio.py`  
  local audio normalization/denoise helper.

## Repository baseline findings

On a 200-file `dev-clean` GPU1 sweep (`~/transcribe/temp/whisperx-benchmark`), the repo defaults currently prefer:

- `batch_size=16`
- `beam_size=1`
- `best_of=1`
- `temperature=0.0`
- `suppress_numerals=false`
- `no-diarize`

These defaults are stored in `config.toml`.
Tested on: NVIDIA GeForce RTX 3090 (24 GB VRAM), using GPU1 for all reported sweeps.

## Hardening and runtime behavior

- Inputs/outputs are normalized to absolute paths.
- Batch submission is split automatically to avoid command-line length limits.
- Docker mounts only required caches and mounts as caller UID/GID.
- Environment defaults keep model access offline:
  - `HF_HUB_OFFLINE=1`
  - `HF_DATASETS_OFFLINE=1`
  - `TRANSFORMERS_OFFLINE=1`

## Project layout

```text
.
├── transcribe
├── whisperx-benchmark
├── config.toml
├── Dockerfile.whisperx-torch280-cu128
├── output/
└── README.md
```

## Full CLI Args

### transcribe

```text
-c, --config PATH         Load defaults from TOML config file.
--no-config               Ignore config file.
--input-dir DIR           Input directory containing audio files.
--output-dir DIR          Output directory for transcript artifacts.
--cuda-devices CSV        Comma-separated CUDA device IDs (defaults to 0).
--recursive               Recurse into subdirectories when discovering files.
--no-recursive            Disable recursion (default).
--model PATH              WhisperX model path (required local snapshot).
--task TEXT               WhisperX task.
--language TEXT           WhisperX language.
--output-format TEXT      WhisperX output format.
--align-model PATH        WhisperX align model/path.
--interpolate-method TEXT WhisperX interpolation method.
--no-align                Disable alignment.
--return-char-alignments  Enable character-level alignments.
--vad-method TEXT         WhisperX VAD method.
--vad-onset TEXT          WhisperX VAD onset.
--vad-offset TEXT         WhisperX VAD offset.
--chunk-size TEXT         WhisperX chunk size.
--device TEXT             WhisperX --device argument.
--compute-type TEXT       WhisperX --compute_type argument.
--batch-size TEXT         WhisperX --batch_size argument.
--max-speakers TEXT       WhisperX --max_speakers.
--min-speakers TEXT       WhisperX --min_speakers.
--diarize-model PATH      WhisperX diarize model path.
--speaker-embeddings      Enable speaker embeddings.
--temperature TEXT        WhisperX --temperature.
--best-of TEXT            WhisperX --best_of.
--beam-size TEXT          WhisperX --beam_size.
--patience TEXT           WhisperX --patience.
--length-penalty TEXT     WhisperX --length_penalty.
--suppress-tokens TEXT    WhisperX --suppress_tokens.
--suppress-numerals       Enable suppress_numerals.
--initial-prompt TEXT     WhisperX --initial_prompt.
--condition-on-previous-text TEXT
                         WhisperX --condition_on_previous_text.
--fp16 TEXT              WhisperX --fp16.
--temperature-increment-on-fallback TEXT
                         WhisperX --temperature_increment_on_fallback.
--compression-ratio-threshold TEXT
                         WhisperX --compression_ratio_threshold.
--logprob-threshold TEXT  WhisperX --logprob_threshold.
--no-speech-threshold TEXT
                         WhisperX --no_speech_threshold.
--max-line-width TEXT     WhisperX --max_line_width.
--max-line-count TEXT     WhisperX --max_line_count.
--highlight-words         Enable highlight_words output.
--segment-resolution TEXT WhisperX --segment_resolution.
--threads TEXT            WhisperX --threads.
--print-progress TEXT     WhisperX --print_progress.
--verbose TEXT            WhisperX --verbose.
--log-level TEXT          WhisperX --log-level.
--hotwords TEXT           WhisperX --hotwords.
--diarize                 Enable diarization.
--no-diarize              Disable diarization.
--skip-transcribe-existing Skip files when output artifacts already exist.
--whisper-arg ARG         Extra whisper argument, repeatable.
--docker-image TEXT       Docker image for whisperx runner.
--docker-pull-policy {always,missing}
                         Docker --pull policy (default: missing).
--docker-cache PATH       Host cache directory for HF/torch caches.
-- [ARGS ...]            Pass-through args for whisperx.
```

### whisperx-benchmark

```text
-c, --config PATH          Load benchmark defaults from TOML config file.
--dataset PATH              Dataset root directory.
--ext {wav}                 Audio extension to process (without dot).
--transcribe PATH           transcribe executable.
--output-root PATH          Benchmark output root.
--output-format {all,txt,json,srt,vtt,tsv,aud}
                           Output format passed to transcribe.
--score-format {txt,json,srt,vtt,tsv,aud}
                           Reference file extension used when scoring.
--max-files N               Optional cap on files per run (0 = no cap).
--batch-size N              batch-size passed to transcribe.
--transcribe-config PATH    transcribe config.toml for every run.
--set KEY=VAL               Fixed whisper option for every run (repeatable).
--sweep KEY=VAL1,VAL2       Tune candidate set (repeatable).
--whisper-arg ARG           Extra transcribe --whisper-arg token (repeatable).
--skip-transcribe-existing   Pass transcribe --skip-transcribe-existing.
--results-csv PATH          Optional CSV destination.
--limit N                   Limit number of runs evaluated (debug).
--diarize                   Enable diarization.
--no-diarize                Pass transcribe --no-diarize (default).
--trace                     Collect GPU resource traces.
--trace-interval SEC        Interval between trace samples (seconds).
```

## License

MIT.
