# Contributor and agent guidance

`whisperX-batch` is a Docker-first local batch transcription and benchmark
harness. It owns orchestration, argument construction, cache and mount behavior,
resume semantics, and benchmark bookkeeping; it does not own WhisperX, speech
models, CUDA, corpora, or a multi-GPU scheduler.

Before changing implementation, read `README.md`, `CONTRIBUTING.md`,
`docs/ARCHITECTURE.md`, `docs/CLI.md`, `docs/PROVENANCE.md`, and
`docs/PUBLICATION.md`. Read `docs/BENCHMARKING.md` before changing or publishing
performance evidence.

## Safe validation

The ordinary reviewed checks are:

```bash
make test
git diff --check
```

They must remain offline and must not require Docker, a GPU, models, datasets,
audio, tokens, or network access. Image builds, model/cache preparation,
transcription, GPU runs, corpus work, and benchmarks are exceptional checks and
require separate explicit authorization and a recorded environment boundary.

## Privacy, provenance, and delivery

- Never commit audio, transcripts, local config, tokens, caches, models,
  datasets, benchmark output, raw host captures, or identifying local paths.
- Preserve read-only input/model mounts, explicit single-process GPU ownership,
  visible fallback behavior, and operator-managed sharding unless evidence
  supports a reviewed architecture change.
- Keep license, upstream revision, model/dataset terms, assistance disclosure,
  and claim limits accurate. A container start or one GPU result is not evidence
  of correctness, reproducibility, or scaling.
- Use a focused feature branch. Commit and push the validated change and open a
  pull request; incomplete or higher-risk work stays draft. Do not treat local
  files as delivered work.
- After publication, send the exact commit, PR, validation, outcome, risks, and
  next action to the repository's external coordination record. Do not claim
  completion until that remote handoff is verified.
