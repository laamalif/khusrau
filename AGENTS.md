# AGENTS.md

## Purpose

Khusrau aligns rough readings of historical Urdu and Persian printed pages with
modern digital reference texts. It supports verse correspondence, continuous
prose scoring, and validated page-level dataset exports.

Treat the historical reading and the modern reference as separate witnesses.
Matching establishes correspondence; it does not establish textual authority or
license modernization of the historical text.

## Setup and Commands

- Python 3.11 or newer is required.
- Use `uv` for dependency management and command execution. Do not add `pip`,
  `requirements.txt`, virtualenv bootstrap scripts, or direct package installs.
- Install core dependencies with `uv sync`.
- Install optional browser tooling with `uv sync --extra browser`, followed by
  `uv run playwright install chromium`.
- Run the unified CLI as `uv run python khusrau.py ...`.
- Use `khusrau.example.toml` as the project configuration template.

Common local checks:

```bash
uv run python test_integrity.py
uv run python test_norm.py
uv run python selftest_align.py
uv run python -m py_compile *.py
git diff --check
```

`uv run python test_norm.py --corpus` is an additional corpus-dependent check.
It is expected to fail when the external reference corpus is unavailable.

## Repository Map

- `khusrau.py`: unified command-line interface.
- `ocr_page.py`, `ocr_batch.py`, `vertex.py`: paid Gemini OCR and artifacts.
- `ganjoor.py`, `verify_ganjoor.py`: Ganjoor retrieval and cache validation.
- `rekhta*.py`: Rekhta verse and prose extraction.
- `manufacture.py`, `selftest_align.py`: verse retrieval and tier assignment.
- `align_prose.py`, `score_real.py`: prose CER and token WER scoring.
- `urdu_norm.py`, `norm_profiles.py`: retrieval and apparatus normalization.
- `normalization_tests/`: executable scholarly normalization specification.
- `dataset_export.py`: JSONL manifests and page-pair exports.
- `test_integrity.py`: cross-module behavior and safety regression tests.

## Safety and Cost Boundaries

- Never make a Gemini call unless the user explicitly requests OCR or a model
  probe. Model access must remain guarded by `HTR_MODEL_ACCESS=1`.
- Reuse existing `.response.json`, `.txt`, and `.partial.txt` artifacts. Do not
  retry partial or failed pages automatically; API calls have a real cost.
- A response is complete only when its finish reason is `STOP`. Never promote a
  truncated or failed response to the canonical `.txt` transcription.
- Keep Gemini credentials local. Never copy `.secrets`, service-account JSON,
  access tokens, or environment files to `vagrant` or another scraping host.
- Network fetching and browser automation must be explicit. `status`, tests, and
  validation paths should remain local-only unless their command says otherwise.
- Preserve direct connections as the default. Proxies remain opt-in through
  `SCRAPE_PROXY` or `REKHTA_PROXY`.

## Data and Export Contracts

- `data/`, `cache/`, `work/`, `.secrets/`, and generated logs are intentionally
  untracked. Do not force-add generated corpora or credentials.
- The current ingestion contract is a full page image and a non-empty,
  same-basename `.txt` file, for example `168.jpg` and `168.txt`.
- Do not redefine this workflow as line images plus `.gt.txt`. Segmentation and
  recognition-training formats are outside the current scope.
- Build authoritative records with `khusrau export manifest`; materialize pairs
  with `khusrau export pages`.
- Candidate and aligned records require explicit `--allow-candidate` at export.
  Without it, only `reviewed` and `gold` records may be exported.
- Preserve content hashes, provenance, stable identifiers, and atomic writes.
  Export code must continue to reject changed source files and existing
  destination directories.

## Scholarly Invariants

- Preserve Urdu and Persian Unicode text as UTF-8. Do not transliterate,
  modernize spelling, silently fix OCR, or replace witness text with reference
  text.
- Retrieval normalization may be permissive; apparatus normalization must retain
  potentially textual, lexical, morphological, or editorial distinctions.
- Never merge the retrieval and apparatus profiles.
- Changes to normalization rules require matching cases in
  `normalization_tests/must_merge.jsonl` or
  `normalization_tests/must_not_merge.jsonl`.
- `gold` means strongly supported correspondence, not diplomatic transcription.
  Keep `review` as a human worklist and `reject` as lack of supported matching,
  not proof of absence.
- Ganjoor coverage boundaries matter. Do not infer that unmatched material is
  absent from a book when it may simply be outside the selected categories.

## Implementation Conventions

- Prefer the unified CLI for user-facing workflows. New commands should use
  `argparse`, return meaningful process codes, and be covered in
  `test_integrity.py`.
- Keep network, paid-model, and filesystem effects behind explicit command
  boundaries so core logic can be tested locally.
- Preserve deterministic ordering and stable tie-breaking in retrieval,
  manifests, exports, and JSONL output.
- Write authoritative JSON and JSONL outputs atomically. Avoid leaving a
  successful-looking partial file after errors.
- Use structured JSON/TOML parsing rather than ad hoc text manipulation.
- Keep edits scoped. Do not refactor frozen normalization behavior or historical
  research logic without fixtures that demonstrate the intended change.

## Validation by Change Type

- CLI, export, OCR artifact, fetcher, or cross-module changes:
  `uv run python test_integrity.py`.
- Normalization changes: `uv run python test_norm.py`; use `--corpus` when the
  external corpus is present.
- Verse retrieval or sequence changes: `uv run python selftest_align.py` and the
  relevant negative-control scripts.
- Dependency changes: update both `pyproject.toml` and `uv.lock` through `uv`.
- Documentation-only changes: verify commands against `khusrau.py --help` and
  run `git diff --check`.

Do not run paid OCR, bulk downloads, browser scraping, or large corpus checks as
part of routine validation.
