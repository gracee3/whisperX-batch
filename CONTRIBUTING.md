# Contributing

This is a personal transcription and benchmark harness rather than a generally
supported package. Small changes that improve correctness, reproducibility,
privacy, or the documented local workflow are welcome for review when useful;
no response time or roadmap commitment is implied.

Before changing code:

- keep audio, transcripts, local config, tokens, model caches, datasets, and
  benchmark outputs out of Git;
- preserve the single-process/single-visible-GPU execution boundary unless an
  architecture change includes correctness and performance evidence;
- keep `make test` offline and free of Docker, GPU, model, dataset, audio, and
  network requirements;
- put resource-heavy validation behind an explicit, separately documented step;
- update architecture, config, changelog, and evidence claims with the behavior.

Run:

```bash
make test
git diff --check
```

Describe the exact configuration and safe public fixture behind a behavioral or
performance claim. Do not attach private recordings or raw host captures to an
issue.
