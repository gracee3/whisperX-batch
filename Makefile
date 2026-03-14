SHELL := /usr/bin/env bash
.RECIPEPREFIX := >

REPO_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SCRIPTS := transcribe whisperx-benchmark
BASH_SCRIPTS :=
COMMANDS := transcribe whisperx-benchmark
CONFIG_TEMPLATE := $(REPO_DIR)/config.toml

PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin

.PHONY: all install uninstall preflight test help

all:
>@echo "Run 'make preflight' to validate the environment, then 'make install' to install scripts."

preflight:
>@command -v bash >/dev/null 2>&1 || { echo "ERROR: bash is required" >&2; exit 1; }
>@command -v docker >/dev/null 2>&1 || { echo "ERROR: docker is required for primary workflow" >&2; exit 1; }
>@command -v find >/dev/null 2>&1 || { echo "ERROR: find is required" >&2; exit 1; }
>@for script in $(SCRIPTS); do \
  [ -f "$(REPO_DIR)/$$script" ] || { echo "ERROR: missing $$script" >&2; exit 1; }; \
  [ -x "$(REPO_DIR)/$$script" ] || { echo "ERROR: script not executable: $$script" >&2; exit 1; }; \
done
>@python3 -m py_compile "$(REPO_DIR)/whisperx-benchmark"
>@python3 -m py_compile "$(REPO_DIR)/transcribe"
>@test -f "$(CONFIG_TEMPLATE)" || { echo "ERROR: missing config template: $(CONFIG_TEMPLATE)" >&2; exit 1; }
>@echo "preflight: OK"

install:
>@mkdir -p "$(BINDIR)"
>@ln -sf "$(REPO_DIR)/transcribe" "$(BINDIR)/transcribe"
>@ln -sf "$(REPO_DIR)/whisperx-benchmark" "$(BINDIR)/whisperx-benchmark"
>@for cmd in $(COMMANDS); do \
  echo "installed: $(BINDIR)/$$cmd"; \
done

uninstall:
>@for cmd in $(COMMANDS); do \
  rm -f "$(BINDIR)/$$cmd"; \
  echo "removed: $(BINDIR)/$$cmd"; \
done

test:
>@command -v bash >/dev/null 2>&1 || { echo "ERROR: bash is required" >&2; exit 1; }
>@command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required" >&2; exit 1; }
>@for script in $(SCRIPTS); do \
  [ -f "$(REPO_DIR)/$$script" ] || { echo "ERROR: missing $$script" >&2; exit 1; }; \
  [ -x "$(REPO_DIR)/$$script" ] || { echo "ERROR: script not executable: $$script" >&2; exit 1; }; \
  echo "found: $(REPO_DIR)/$$script"; \
done
>@if [ -n "$(BASH_SCRIPTS)" ]; then \
  for script in $(BASH_SCRIPTS); do \
    bash -n "$(REPO_DIR)/$$script"; \
    echo "test: checked $$script"; \
  done; \
fi
>@if ! python3 "$(REPO_DIR)/whisperx-benchmark" --help >/dev/null; then \
  echo "ERROR: failed --help check for whisperx-benchmark" >&2; \
  exit 1; \
fi
>@echo "test: checked whisperx-benchmark"
>@python3 -m py_compile \
  "$(REPO_DIR)/scripts/setup_librispeech_dataset.py" \
  "$(REPO_DIR)/scripts/preseed_whisperx_cache.py" \
  "$(REPO_DIR)/scripts/convert_flac_to_wav.py" \
  "$(REPO_DIR)/scripts/clean_audio.py" \
  "$(REPO_DIR)/whisperx-benchmark"
>@echo "test: OK"

help:
>@echo "make preflight"
>@echo "  Validate local prerequisites for running scripts."
>@echo "make install [BINDIR=path|PREFIX=path]"
>@echo "  Install transcribe and whisperx-benchmark into $(BINDIR)"
>@echo "make uninstall [BINDIR=path|PREFIX=path]"
>@echo "  Remove transcribe and whisperx-benchmark from $(BINDIR)."
>@echo "make test"
>@echo "  Run lightweight local syntax/help checks."
