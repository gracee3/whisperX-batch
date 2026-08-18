# Publication, privacy, and research notes

## Responsibility and assistance

The maintainer is responsible for the code, configuration, experiments,
analysis, citations, and release decisions. Material AI assistance should be
disclosed in a release or evidence note when it affects implementation,
analysis, or prose. It is not independent replication or peer review.

## Data and people

Do not publish private audio, transcripts, diarization labels, prompts, tokens,
cache metadata, local configs, filenames, or raw host captures. Public-corpus
work must retain corpus identity, license, subset, and exclusions. Recording or
processing another person can require consent and a retention/deletion plan;
this repository does not supply either automatically.

Diarization and speaker embeddings can make voice identity more sensitive even
when transcript text looks harmless. Keep generated outputs local unless their
publication basis is explicit.

The current tree removes machine-local path examples and does not contain audio,
transcript-like result files, common hosted-token patterns, or credential-named
files. Earlier public commits do contain machine-local path examples. This
cleanup did not rewrite history, so review the full history—not only `main`—when
making a privacy statement or preparing an archive.

## Claim boundaries

- WER describes a pinned corpus, normalization rule, model/decoder, and run. It
  does not establish accuracy for other people, languages, acoustics, or tasks.
- Agreement with WhisperX is integration evidence, not an independent speech
  recognition validation.
- A successful container start is not evidence of transcription correctness,
  offline reproducibility, or performance.
- One RTX 3090 result is a statement about that run, not all GPUs or multi-GPU
  scaling.
- This project is not validated for medical, legal, surveillance, accessibility,
  or other high-stakes transcription decisions.

Record negative results and failed parameter combinations when they explain the
selected defaults. Do not discard failures or loosen scoring rules to make a
tuning result look cleaner.

## Release meaning

The existing v0.1.0 and v0.2.0 tags identify historical stack revisions. They do
not have matching GitHub releases, locked image digests, or a stated support
policy. A future release should align tag, changelog, README status, immutable
image identity, offline CI, opt-in GPU evidence, known limitations, and migration
notes at one commit.
