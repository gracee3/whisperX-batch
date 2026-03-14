#!/usr/bin/env python3
"""Pre-seed offline cache artifacts used by this whisperx stack."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path
import tempfile


ALIGN_MODEL_URL = "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
ALIGN_MODEL_RELPATH = "torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth"
NLTK_ZIP_URL = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-seed WhisperX offline cache artifacts.")
    parser.add_argument(
        "--cache-dir",
        default="/data/models/.hf-cache",
        help="Base cache directory for WhisperX runtime artifacts.",
    )
    parser.add_argument(
        "--align-url",
        default=ALIGN_MODEL_URL,
        help="URL for the aligner checkpoint.",
    )
    parser.add_argument(
        "--align-relpath",
        default=ALIGN_MODEL_RELPATH,
        help="Relative path under cache-dir for aligner checkpoint.",
    )
    parser.add_argument(
        "--nltk-url",
        default=NLTK_ZIP_URL,
        help="URL for punkt_tab.zip.",
    )
    parser.add_argument(
        "--nltk-relpath",
        default="nltk_data",
        help="Relative path under cache-dir where NLTK data is extracted.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite artifacts even if already present.",
    )
    return parser.parse_args()


def download_to_file(url: str, dst: Path, force: bool = False) -> None:
    if dst.exists() and not force:
        print(f"SKIP: already present: {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"DOWNLOAD: {url} -> {dst}")
    with tempfile.NamedTemporaryFile(delete=False, dir=dst.parent) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def marker_exists(nltk_root: Path) -> bool:
    checks = [
        nltk_root / "tokenizers/punkt_tab/README",
        nltk_root / "tokenizers/punkt_tab/english/sent_starters.txt",
        nltk_root / "tokenizers/punkt_tab/english/abbrev_types.txt",
    ]
    return all(path.is_file() for path in checks)


def extract_nltk_punkt_tab(url: str, nltk_root: Path, force: bool = False) -> None:
    if marker_exists(nltk_root) and not force:
        print(f"SKIP: punkt_tab already present at: {nltk_root}")
        return

    print(f"DOWNLOAD+EXTRACT: punkt_tab -> {nltk_root}")
    nltk_root.mkdir(parents=True, exist_ok=True)
    tmp_zip = nltk_root.parent / ".tmp_punkt_tab.zip"
    with tempfile.NamedTemporaryFile(delete=False, dir=tmp_zip.parent) as tmp:
        tmp_path = Path(tmp.name)

    try:
        with urllib.request.urlopen(url) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        tmp_path.replace(tmp_zip)
        with zipfile.ZipFile(tmp_zip) as archive:
            archive.extractall(nltk_root)

        source_dir = nltk_root / "punkt_tab"
        tokenizers_dir = nltk_root / "tokenizers"
        target_dir = tokenizers_dir / "punkt_tab"

        if source_dir.exists():
            tokenizers_dir.mkdir(parents=True, exist_ok=True)
            if target_dir.exists():
                shutil.rmtree(target_dir)
            shutil.move(str(source_dir), str(target_dir))
    finally:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    if not marker_exists(nltk_root):
        raise RuntimeError(f"Failed to extract punkt_tab data to {nltk_root}")


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    align_path = cache_dir / args.align_relpath
    nltk_root = cache_dir / args.nltk_relpath

    print(f"Using cache dir: {cache_dir}")
    download_to_file(args.align_url, align_path, args.force)
    extract_nltk_punkt_tab(args.nltk_url, nltk_root, args.force)

    print("Done.")
    print(f"align model path: {align_path}")
    print(f"nltk punkt root:  {nltk_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
