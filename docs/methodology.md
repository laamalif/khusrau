# Methodology

Khusrau aligns rough readings of historical Urdu and Persian printed pages
with modern digital reference texts. The historical reading and the reference
are treated as separate witnesses: correspondence does not establish textual
authority and does not license modernization of the historical text.

## Why sequence matters

Shared vocabulary is weak evidence in ghazal poetry. Different poems may use
the same devotional or amatory lexicon, and every rhyming line in a radīf
family may end with the same refrain.

Khusrau therefore matches at the hemistich level and asks whether several
matches point to the same poem with verse numbers rising in page order. Two
poems may share a refrain; they are much less likely to share a sequence of
otherwise corresponding lines.

Automatic decisions have three tiers:

- `gold`: several lines support the same poem in the expected sequence;
- `review`: a plausible correspondence requiring human assessment;
- `reject`: the available reference and thresholds do not support a match.

`Review` is a worklist, not a discard category. `Reject` is not proof that the
material is absent from the work: it may be outside the selected reference
categories or otherwise uncovered by the corpus.

## Historical spelling remains evidence

Khusrau uses two deliberately separate forms of comparison:

- Retrieval normalization is permissive. It may set aside script and spelling
  differences to locate a candidate.
- Apparatus normalization is conservative. It reports the shallowest level at
  which two readings agree while retaining distinctions that may be textual,
  lexical, morphological, or editorial.

The apparatus profile is a ladder:

```text
identical → encoding → orthographic → vocalisation → segmentation → None
```

`None` means that the difference remains substantive at the implemented
levels. The permissive retrieval form is never used to rewrite the witness.
The original rough reading and reference reading remain separate in output.

The distinction is enforced by executable fixtures in
`normalization_tests/must_merge.jsonl` and
`normalization_tests/must_not_merge.jsonl`. Each normalization profile ID is a
content hash over its rule data, so stored comparisons identify the rules that
produced them.

## Verse, prose, and dictionaries

For verse, Khusrau retrieves candidate hemistichs independently, groups them by
poem, and tests their sequence before assigning a tier.

Accepted alignments also report coverage inside the inferred parallel span.
The longest strictly increasing one-to-one chain supplies the aligned-line
count. Reference-span recall divides that count by the hemistich interval from
the first through last matched reference line; hypothesis-span precision uses
the corresponding page-line interval. Their harmonic mean is reported as span
F1. This detects internal omissions, duplicates, and ordering failures without
pretending that unmatched leading or trailing material can be inferred.

For prose, it compares an ordered page run with a continuous reference and
reports folded and strict character error rates plus token error rate. Offset
alignment is appropriate here because prose forms one continuous stream.

For dictionaries, image-aware extraction can structure entries from page
images and rough OCR. A separate local-only path conservatively identifies
entry boundaries from existing OCR text and sends uncertain blocks to an
unresolved worklist.

## Relation to prior work

Khusrau ports the “manufactured ground truth” approach from the `siraj`
workflow in
[`steps-re/afghan-press-archive`](https://github.com/steps-re/afghan-press-archive)
to Urdu and Indo-Persian verse.

The principal changes are:

- the matching unit is the hemistich rather than the page;
- verse order is used as structural evidence;
- offset voting is retained only for continuous prose;
- output has a review tier rather than a binary accept/reject division.

The name refers to Amir Khusrau (c. 1253–1325), associated with Persian and
early Hindavi literary traditions.

## Interpretation limits

- A strong match establishes correspondence, not the authority of either
  edition.
- Material absent from the reference corpus cannot be retrieved.
- Rough OCR is not a substitute for diplomatic transcription.
- Headings, marginalia, commentary, and other non-reference material may
  remain unmatched.
- Copyright and publication rights for modern reference texts must be assessed
  separately.
