# Kulliyāt-e Ghalib (Newal Kishore Press, 1925)

This example contains three consecutive ghazal-section pages from the 1925
Newal Kishore Press edition of *Kulliyāt-e Ghalib*:

- `pages/340.jpg` through `pages/342.jpg`: page scans;
- matching `.txt` files: rough Gemini 3.8 Flash readings, preserved without
  silent correction from the modern reference;
- `expected-alignments.jsonl`: existing Khusrau correspondence output against
  Ganjoor category 2572.

The sample is deliberately small, but includes complete ghazals, page
continuations, imperfect readings, multiple matches on one page, and a
`review` result alongside strongly supported `gold` correspondences.

## Run it

From the repository root:

```bash
uv sync
uv run python khusrau.py --config examples/ghalib-1925/khusrau.toml \
  reference fetch
uv run python khusrau.py --config examples/ghalib-1925/khusrau.toml \
  reference verify
uv run python khusrau.py --config examples/ghalib-1925/khusrau.toml align
```

Fetching the Ganjoor reference is an explicit network operation. Alignment
itself uses the resulting local cache and writes `alignments.jsonl` beside this
README. Compare it with `expected-alignments.jsonl`; small numeric differences
may occur if retrieval logic or the upstream reference changes.

No Gemini call is needed: the rough readings are already included. Model
response artifacts are intentionally omitted.

## Provenance and rights

Bibliographic record:

> Mirza Asadullah Khan Ghalib, *Kulliyāt-e Ghalib*, Newal Kishore Press,
> 1925.

The scans were obtained from Internet Archive item
[`kulliyatghalib-v2`](https://archive.org/details/kulliyatghalib-v2), where they
are marked Public Domain. The included rough readings and alignment output are
research artifacts derived from those pages and are provided under CC0-1.0.
The Ganjoor reference text is not redistributed in this example and remains
subject to its source's terms.
