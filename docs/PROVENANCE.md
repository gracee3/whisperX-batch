# Provenance and third-party boundaries

The orchestration scripts and documentation in this repository are original
project work under the MIT license. The repository does not vendor WhisperX,
models, public corpora, CUDA, or Python packages and does not relicense them.

The Dockerfile currently obtains runtime components from these upstreams:

- NVIDIA CUDA container images;
- PyTorch and torchaudio;
- Triton;
- CTranslate2 and faster-whisper;
- WhisperX;
- Hugging Face Hub and Transformers;
- pyannote-audio;
- Ubuntu, Python, ffmpeg, and their distribution packages.

Optional runtime assets include Whisper/faster-whisper model snapshots,
alignment checkpoints, pyannote diarization models, and NLTK tokenizer data.
The dataset helper targets LibriSpeech from OpenSLR. Each image, package, model,
and dataset keeps its own license, acceptable-use terms, attribution, and access
conditions. Review the exact resolved versions/model cards before publishing an
image, result, or derived artifact.

The current image is not fully locked: `triton>=3.3.0`,
`transformers>=4.48.0`, distribution packages, and transitive dependencies can
change on rebuild. A release candidate needs an immutable image digest and
resolved inventory; the presence of a Dockerfile is not enough.
