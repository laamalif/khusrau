# khusrau

Khusrau helps scholars identify Urdu and Persian works in historical printed
books. It aligns imperfect page readings with modern digital references while
keeping the historical witness and reference text visibly separate.

It supports:

- verse identification and line correspondence;
- continuous-prose alignment and error measurement;
- structured dictionary extraction;
- validated page-image and transcription exports.

Khusrau is research software for textual scholars, literary historians,
bibliographers, and digital-humanities projects. A match establishes
correspondence—not textual authority, diplomatic transcription, or permission
to modernize the historical reading.

## Try the bundled example

Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/) are required.

```bash
uv sync
uv run python khusrau.py \
  --config examples/ghalib-1925/khusrau.toml status
```

This checks three included pages from the 1925 Newal Kishore
*Kulliyāt-e Ghalib*, their rough readings, and the locally cached reference
status. It makes no network or model calls.

To reproduce the alignment:

```bash
# Network: fetch and verify the Ganjoor reference.
uv run python khusrau.py \
  --config examples/ghalib-1925/khusrau.toml reference fetch
uv run python khusrau.py \
  --config examples/ghalib-1925/khusrau.toml reference verify

# Local: align the bundled rough readings.
uv run python khusrau.py \
  --config examples/ghalib-1925/khusrau.toml align
```

The result can be compared with
[`examples/ghalib-1925/expected-alignments.jsonl`](examples/ghalib-1925/expected-alignments.jsonl).
The [example documentation](examples/ghalib-1925/README.md) records provenance
and separate data terms.

## What the output means

Verse output places the historical reading and modern reference side by side
with similarity and sequence evidence:

```json
{
  "page": "340.txt",
  "poem_id": 105865,
  "tier": "gold",
  "matched_lines": 19,
  "monotone_ratio": 1.0,
  "reference_span_recall": 0.95,
  "hypothesis_span_precision": 0.95,
  "pairs": [
    {
      "hyp": "آئینہ مدارید بہ پیش نفس ما",
      "ref": "آیینه مدارید به پیش نفس ما"
    }
  ]
}
```

The tiers are:

- `gold`: strongly supported correspondence;
- `review`: plausible correspondence requiring human assessment;
- `reject`: no supported match in the available reference and thresholds.

`Gold` does not mean that the rough reading is a corrected transcription.
`Reject` does not prove absence from the book. Khusrau preserves both witnesses
and their differences.

For verse, span recall measures how much of the inferred reference passage was
recovered between its first and last matched hemistich. Span precision measures
how much of the corresponding page-line interval participates in a monotone
one-to-one alignment. These expose internal omissions and duplicated or
out-of-order lines; they cannot infer material omitted outside the matched
boundaries.

For prose, output includes folded and strict character error rates and token
error rate. Dictionary output contains candidate entries, provenance, quality
flags, and an unresolved worklist where boundaries are uncertain.

## Workflows

### Existing rough readings

Copy and edit the example project manifest:

```bash
cp khusrau.example.toml khusrau.toml
uv run python khusrau.py --config khusrau.toml status
uv run python khusrau.py --config khusrau.toml reference fetch
uv run python khusrau.py --config khusrau.toml reference verify
uv run python khusrau.py --config khusrau.toml align
```

`reference fetch` is an explicit network operation. `status`, verification of
already cached artifacts, and alignment are local.

Ganjoor and Rekhta reference commands may also be given explicit arguments:

```bash
uv run python khusrau.py reference fetch 2572
uv run python khusrau.py reference verify 2572
uv run python khusrau.py reference rekhta-fetch \
  mirza-ghalib references/rekhta-ghalib
uv run python khusrau.py reference rekhta-verify \
  references/rekhta-ghalib
```

### Optional paid OCR

Gemini OCR is disabled unless model access is explicitly enabled:

```bash
export HTR_MODEL_ACCESS=1
export GOOGLE_APPLICATION_CREDENTIALS="$PWD/.secrets/service_account.json"
uv run python khusrau.py --config khusrau.toml ocr
```

Model calls cost money. Khusrau records each response beside its image and does
not automatically retry recorded partial or failed responses. Only a response
whose finish reason is `STOP` becomes the canonical `.txt`; unfinished text is
written to `.partial.txt`.

Keep credentials on the OCR machine. Never copy `.secrets`, service-account
JSON, tokens, or environment files to a scraping host.

### Continuous prose

Given a continuous reference and ordered page readings:

```bash
uv run python khusrau.py score prose \
  reference.txt pages/*.txt -o prose-score.json
```

Known Rekhta story pages can be extracted separately:

```bash
# Network: download server-rendered reference text.
uv run python rekhta_prose.py page \
  "https://www.rekhta.org/stories/..." > reference.txt
```

### Dictionaries

Structure dictionary entries from page images and existing OCR sidecars:

```bash
# Paid model call; requires HTR_MODEL_ACCESS=1.
uv run python khusrau.py dictionary extract \
  pages/017.jpg pages/018.jpg \
  --response work/017-018.response.json \
  --output work/017-018.entries.jsonl
```

Or conservatively identify entries using existing OCR text only:

```bash
# Local only.
uv run python khusrau.py dictionary local \
  pages/017.txt pages/018.txt \
  --output work/entries.jsonl \
  --unresolved-output work/unresolved.jsonl \
  --retained-output work/retained.jsonl
```

The local path preserves source text and routes uncertain boundaries to the
unresolved file rather than silently promoting them.

### Dataset manifests and page pairs

Build one authoritative manifest record per image/text pair:

```bash
uv run python khusrau.py export manifest \
  --dataset ghalib-sample \
  --status candidate \
  --mode verse \
  --alignment work/ghalib-pairs.jsonl \
  -o work/manifest.jsonl \
  pages/*.jpg
```

Materialize a validated dataset:

```bash
uv run python khusrau.py export pages \
  --allow-candidate \
  work/manifest.jsonl work/export
```

By default, only `reviewed` and `gold` records may be exported. Candidate and
aligned records require the explicit `--allow-candidate` shown above. Omit it
for reviewed or gold material. Export rechecks content hashes, rejects existing
destination directories, and preserves stable IDs and provenance.

## Verified results

Ten ghazal-section pages from the 1925 Newal Kishore *Kulliyāt-e Ghalib* were
read with Gemini 3.8 Flash and aligned against Ganjoor category 2572:

```text
pages                10
transcribed lines    412
gold (page, poem)    26
review                2
gold line pairs      386 (93.7% of transcribed lines)
median folded CER    0.0476
```

The continuous-prose path was checked against two stories by Ghulam Abbas:

| Story | Scan pages | Reference tokens | Folded CER | Token error |
|---|---:|---:|---:|---:|
| *Anandi* | 20 | 6,213 | 2.09% | 6.81% |
| *Katba* | 11 | 3,719 | 4.97% | 9.09% |

These results demonstrate correspondence within the selected references. They
do not establish the authority of either edition.

## Scholarly and technical design

Retrieval normalization is permissive enough to find candidates despite common
script and spelling differences. Apparatus normalization is deliberately
conservative and retains potentially meaningful distinctions. The two profiles
are separate and regression-tested.

Verse matching operates at the hemistich level and uses rising verse sequence
as structural evidence. This helps distinguish true correspondence from poems
that merely share vocabulary or a refrain.

See [Methodology](docs/methodology.md) for the rationale, interpretation
limits, and relation to prior work. See the [Module guide](docs/modules.md) for
the repository map.

## Configuration and validation

`khusrau.example.toml` documents page globs, reference sources, OCR settings,
alignment thresholds, and output paths. Paths are resolved relative to the
manifest; explicit command-line arguments override configured values.

Common local checks:

```bash
uv run python test_integrity.py
uv run python test_norm.py
uv run python selftest_align.py
uv run python -m py_compile *.py
git diff --check
```

The corpus-dependent `uv run python test_norm.py --corpus` requires an external
reference corpus. Browser-based Rekhta tooling requires:

```bash
uv sync --extra browser
uv run playwright install chromium
```

Run `uv run python khusrau.py --help` for the complete command interface.

## Limits

- Material absent from the selected reference cannot be retrieved.
- Rough OCR is not a substitute for diplomatic transcription.
- Headings, marginalia, commentary, and non-reference content may remain
  unmatched.
- Network retrieval and browser automation are explicit operations.
- Rights in modern reference texts and project data must be assessed
  separately.

## License

Khusrau source code and documentation are licensed under the
[Apache License 2.0](LICENSE).

Page images, scans, reference texts, transcriptions, caches, model responses,
and generated datasets are excluded unless explicitly stated. Their own
copyright, licensing, and terms apply. The bundled Ghalib example records its
scan provenance and CC0-1.0 terms separately.
