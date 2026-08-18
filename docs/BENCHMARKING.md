# Benchmark publication

The benchmark harness is useful for local tuning, but a CSV alone is not a
portable result.

For a publishable run, record:

- repository commit and clean-tree state;
- Dockerfile plus immutable image digest and resolved package inventory;
- driver, CUDA runtime, GPU model/count, CPU, RAM, and relevant storage;
- model, alignment, diarization, corpus, and scoring revisions and licenses;
- exact redacted config and command, including device/shard ownership;
- input subset, file count, duration distribution, and excluded/failed files;
- warmup policy, repetitions, ordering/randomization, and summary dispersion;
- transcript correctness/WER alongside runtime, throughput, VRAM, utilization,
  and trace sampling interval;
- whether caches were cold/warm and whether any network access occurred;
- raw machine-readable results small and safe enough for review.

Compare equivalent workloads. Independent GPU shards do not establish linear
scaling unless aggregate throughput, I/O contention, failures, and result
integrity were measured together.

## Historical observation

A 2026-03-14 maintainer sweep over 200 LibriSpeech `dev-clean` files on one RTX
3090 led to the current tuning defaults: batch size 16, beam size 1, best-of 1,
temperature 0, numeral suppression off, and diarization off. The repository does
not currently preserve the complete command, immutable image, raw repetitions,
failure inventory, or machine-readable evidence required to reproduce that
comparison. Treat the settings as a starting point, not a performance claim.
