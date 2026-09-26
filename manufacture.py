#!/usr/bin/env python3
"""Manufacture image<->text gold from page reads plus a born-digital reference.

Pipeline, per page:
  1. match every transcribed hemistich against the reference hemistich index
  2. group the matches by reference poem
  3. accept a (page, poem) pair only on STRUCTURAL evidence, not vocabulary overlap

Why structure and not a similarity score. §4/§5 measured that shared-vocabulary
features cannot separate a true pair from a decoy in this corpus, which is the same
wall siraj hit -- and their answer was to buy a model judge on every candidate. §5
showed that matching at the HEMISTICH instead of the page collapses the overlap band
from ~78% to 2.5%, so the judge is only needed on the residue. The gates below encode
that: several distinct reference verses matched, each at high per-line similarity, in
monotone verse order.

Verse ORDER is the discriminator that vocabulary cannot give. Two different ghazals in
one radif family share their radif in the same position on every line; what they do not
share is a rising sequence of verse indices. Note this is the opposite of siraj's
finding that LIS was useless -- there the offset bucket had already forced
monotonicity, so it carried no information. Here nothing forces it, so it does.

The output is deliberately three-tier (gold / review / reject) with `review` kept as a
worklist rather than discarded -- siraj's discipline, and their 43 uncertain + 22
different judgements were the most useful thing they had after the gold itself.

Usage: manufacture.py <catId[,catId...]> <page.txt ...>

Multiple category ids are comma-separated and indexed together. This is required, not
a convenience: `asrarorumuz` is ONE scan containing TWO Ganjoor works (اسرار خودی 1358
then رموز بیخودی 1359), and a single-cat index finds nothing on the second half of the
book. Confirmed on a 10-page sample -- pages 40-90 matched بخش ۱۰-۲۰ of 1358 and pages
100-130 matched بخش ۳-۱۳ of 1359.
"""
import json
import os
import sys
import tempfile
from collections import defaultdict

from ganjoor import load_index
from norm_profiles import APPARATUS_ID, RETRIEVAL_ID
from urdu_norm import cer, norm_match

MIN_JACCARD = float(os.environ.get("MFG_MIN_JAC", "0.55"))   # per-line accept
MIN_LINES = int(os.environ.get("MFG_MIN_LINES", "4"))        # distinct verses matched
MIN_MONO = float(os.environ.get("MFG_MIN_MONO", "0.80"))     # monotone fraction
REVIEW_LINES = int(os.environ.get("MFG_REVIEW_LINES", "2"))
OUT = os.environ.get("MFG_OUT", "work/gold_pairs.jsonl")


def lis_len(seq):
    """Longest strictly increasing subsequence length."""
    import bisect
    tails = []
    for x in seq:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def span_metrics(hits):
    """Coverage within the parallel span implied by strong line matches.

    The longest increasing chain is the one-to-one correspondence count. Using
    all strong hits would reward duplicated or out-of-order page lines. These
    metrics can detect omissions inside the first/last matched boundaries, but
    cannot infer text omitted before the first or after the last match.
    """
    if not hits:
        return {
            "aligned_lines": 0,
            "hypothesis_span_lines": 0,
            "hypothesis_span_precision": 0.0,
            "reference_span_lines": 0,
            "reference_span_recall": 0.0,
            "span_f1": 0.0,
        }
    ordered = sorted(hits, key=lambda hit: hit["line"])
    aligned = lis_len([hit["verse"] for hit in ordered])
    hypothesis_span = (
        max(hit["line"] for hit in hits)
        - min(hit["line"] for hit in hits)
        + 1
    )
    reference_span = (
        max(hit["verse"] for hit in hits)
        - min(hit["verse"] for hit in hits)
        + 1
    )
    precision = aligned / hypothesis_span
    recall = aligned / reference_span
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "aligned_lines": aligned,
        "hypothesis_span_lines": hypothesis_span,
        "hypothesis_span_precision": round(precision, 4),
        "reference_span_lines": reference_span,
        "reference_span_recall": round(recall, 4),
        "span_f1": round(f1, 4),
    }


def page_lines(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("##"):
                continue
            out.append(l)
    return out


def process(ix, path, min_jaccard=MIN_JACCARD, min_lines=MIN_LINES,
            min_monotone=MIN_MONO, review_lines=REVIEW_LINES):
    lines = page_lines(path)
    per_poem = defaultdict(list)
    for li, text in enumerate(lines):
        m = ix.match_line(text, min_jaccard=0.30)
        if not m:
            continue
        pid, vj, jac, ref = m
        per_poem[pid].append({"line": li, "verse": vj, "jaccard": round(jac, 4),
                              "hyp": text, "ref": ref})

    recs = []
    for pid, hits in per_poem.items():
        strong = [h for h in hits if h["jaccard"] >= min_jaccard]
        verses = sorted({h["verse"] for h in strong})
        n_verses = len(verses)
        order = [h["verse"] for h in sorted(strong, key=lambda h: h["line"])]
        mono = (lis_len(order) / len(order)) if order else 0.0

        if n_verses >= min_lines and mono >= min_monotone:
            tier = "gold"
        elif n_verses >= review_lines:
            tier = "review"
        else:
            tier = "reject"
        if tier == "reject":
            continue

        # CER over the matched span only -- never the whole page (siraj's rule).
        a = " ".join(t for h in strong for t in norm_match(h["hyp"]))
        b = " ".join(t for h in strong for t in norm_match(h["ref"]))
        coverage = span_metrics(strong)
        recs.append({
            "page": os.path.basename(path),
            "poem_id": pid,
            "poem_title": ix.poems[pid].get("fullTitle") or ix.poems[pid].get("title"),
            "tier": tier,
            "page_lines": len(lines),
            "matched_lines": len(strong),
            "distinct_verses": n_verses,
            "monotone_ratio": round(mono, 3),
            "mean_jaccard": round(sum(h["jaccard"] for h in strong) / max(1, len(strong)), 4),
            "cer_matched_span": round(cer(a, b), 4),
            **coverage,
            "pairs": sorted(strong, key=lambda h: h["line"]),
        })
    recs.sort(key=lambda r: (-r["distinct_verses"], r["cer_matched_span"]))
    return recs, len(lines)


def run(cat_ids, paths, output=OUT, min_jaccard=MIN_JACCARD,
        min_lines=MIN_LINES, min_monotone=MIN_MONO, review_lines=REVIEW_LINES):
    ix = load_index(*cat_ids)
    print(f"reference: {len(ix.poems)} poems, {len(ix.lines)} hemistichs\n", flush=True)

    all_recs, tot_lines, matched_lines = [], 0, 0
    for p in paths:
        recs, n = process(
            ix, p, min_jaccard=min_jaccard, min_lines=min_lines,
            min_monotone=min_monotone, review_lines=review_lines)
        tot_lines += n
        g = [r for r in recs if r["tier"] == "gold"]
        matched_lines += sum(r["matched_lines"] for r in g)
        all_recs += recs
        tag = ", ".join(f"{r['poem_title'].split('»')[-1].strip()}"
                        f"[{r['tier'][0]}{r['distinct_verses']}]" for r in recs) or "-"
        print(f"{os.path.basename(p):20s} {n:3d} lines -> {tag}")

    output_dir = os.path.dirname(os.path.abspath(output))
    os.makedirs(output_dir, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".jsonl", dir=output_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for r in all_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(temporary, output)

    gold = [r for r in all_recs if r["tier"] == "gold"]
    rev = [r for r in all_recs if r["tier"] == "review"]
    aligned_lines = sum(r.get("aligned_lines", r["distinct_verses"]) for r in gold)
    hypothesis_span_lines = sum(
        r.get("hypothesis_span_lines", r["matched_lines"]) for r in gold)
    reference_span_lines = sum(
        r.get("reference_span_lines", r["distinct_verses"]) for r in gold)
    span_precision = aligned_lines / max(1, hypothesis_span_lines)
    span_recall = aligned_lines / max(1, reference_span_lines)
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if span_precision + span_recall else 0.0
    )

    metadata = {
        "categories": cat_ids,
        "pages": [os.path.abspath(p) for p in paths],
        "output": os.path.abspath(output),
        "reference_poems": len(ix.poems),
        "reference_hemistichs": len(ix.lines),
        "thresholds": {
            "min_jaccard": min_jaccard,
            "min_lines": min_lines,
            "min_monotone_ratio": min_monotone,
            "review_lines": review_lines,
        },
        "profiles": {
            "retrieval": RETRIEVAL_ID,
            "apparatus": APPARATUS_ID,
        },
        "summary": {
            "pages": len(paths),
            "transcribed_lines": tot_lines,
            "gold_page_poem_pairs": len(gold),
            "review_page_poem_pairs": len(rev),
            "gold_matched_lines": matched_lines,
            "aligned_lines": aligned_lines,
            "hypothesis_span_lines": hypothesis_span_lines,
            "hypothesis_span_precision": round(span_precision, 4),
            "reference_span_lines": reference_span_lines,
            "reference_span_recall": round(span_recall, 4),
            "span_f1": round(span_f1, 4),
            "coverage_scope": "between first and last matched boundaries",
        },
    }
    meta_path = output + ".meta.json"
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".json", dir=output_dir)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(temporary, meta_path)

    cers = sorted(r["cer_matched_span"] for r in gold)
    print(f"\npages                {len(paths)}")
    print(f"transcribed lines    {tot_lines}")
    print(f"gold (page,poem)     {len(gold)}   review {len(rev)}")
    print(f"gold LINE pairs      {matched_lines}   "
          f"({100*matched_lines/max(1,tot_lines):.1f}% of transcribed lines)")
    print(f"parallel-span lines  {aligned_lines}/{reference_span_lines} reference "
          f"({100*span_recall:.1f}% recall)")
    print(f"span precision       {aligned_lines}/{hypothesis_span_lines} hypothesis "
          f"({100*span_precision:.1f}%)")
    print(f"span F1              {span_f1:.4f}")
    if cers:
        print(f"CER over matched span  median {cers[len(cers)//2]:.4f}  "
              f"p25 {cers[len(cers)//4]:.4f}  p75 {cers[3*len(cers)//4]:.4f}")
    print(f"\nwrote {output}")
    print(f"metadata {meta_path}")
    print("NOTE: gold TEXT comes from the reference, not the read (§18.1). The read's "
          "only job was to retrieve; CER above describes the READER, not the gold.")
    print("COVERAGE LIMIT: span recall measures omissions only between the first and "
          "last matched boundaries; it cannot infer omitted leading or trailing lines.")
    return 0


def main():
    if len(sys.argv) < 3:
        print("usage: manufacture.py <catId[,catId...]> <page.txt ...>", file=sys.stderr)
        return 2
    cat_ids = [int(x) for x in sys.argv[1].split(",") if x.strip()]
    return run(cat_ids, sys.argv[2:])


if __name__ == "__main__":
    raise SystemExit(main())
