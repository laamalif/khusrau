#!/usr/bin/env python3
"""Negative control on real data: withhold the true poem, confirm the pipeline says no.

manufacture.py matched 91.5% of transcribed lines on the pilot. That number is
meaningless on its own -- a matcher that always returns its nearest neighbour scores
the same. §5 made this point in simulation; this asserts it on the real pipeline.

For each page: rebuild the index with the poems that page legitimately matched
REMOVED, then re-run. Every "gold" record that still appears is a false positive, and
the false-positive rate is what decides whether the gold set is trustworthy.

Also reports the structural check the pilot output hinted at: in a divan ordered by
radif, the reference poem number should rise with the page number. That is evidence
no threshold can give.

Usage: reject_real.py <catId> <page.txt ...>
"""
import json
import os
import re
import sys

from ganjoor import Index, _find, cat_poems, load_index
from manufacture import process

CACHE = os.environ.get("GANJOOR_CACHE",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "work/ganjoor"))
_FA = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def poem_no(title):
    m = re.search(r"([۰-۹]+)", title or "")
    return int(m.group(1).translate(_FA)) if m else None


def page_no(path):
    m = re.search(r"_(\d+)\.txt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def all_poems(cat_id):
    out = []
    for meta in cat_poems(cat_id):
        p = _find(f"poem_{meta['id']}.json", cat_id)
        if os.path.exists(p):
            out.append(json.load(open(p, encoding="utf-8")))
    return out


def main():
    cat_id = int(sys.argv[1])
    paths = sys.argv[2:]
    poems = all_poems(cat_id)
    full = load_index(cat_id)
    print(f"reference: {len(poems)} poems, {len(full.lines)} hemistichs\n")

    # ---- positive pass + structural check
    truth = {}
    rows = []
    for p in paths:
        recs, n = process(full, p)
        gold = [r for r in recs if r["tier"] == "gold"]
        truth[p] = {r["poem_id"] for r in gold}
        best = max(gold, key=lambda r: r["distinct_verses"], default=None)
        rows.append((page_no(p), poem_no(best["poem_title"]) if best else None,
                     len(gold), sum(r["matched_lines"] for r in gold), n))

    print("page  top-ghazal#  gold-poems  gold-lines  page-lines")
    ok = 0
    prev = None
    for pg, gz, ng, ml, n in sorted(rows):
        mono = "" if prev is None or gz is None else ("  rising" if gz > prev else "  <-- OUT OF ORDER")
        if prev is not None and gz is not None and gz > prev:
            ok += 1
        prev = gz if gz is not None else prev
        print(f"{pg:5d} {str(gz):>11s} {ng:11d} {ml:11d} {n:11d}{mono}")
    print(f"\nmonotone page->ghazal steps: {ok}/{len(rows)-1}")

    # ---- negative pass: withhold each page's true poems
    print("\n--- negative control: true poems withheld per page ---")
    fp_pages, fp_recs, fp_lines = 0, 0, 0
    for p in paths:
        held = truth[p]
        if not held:
            continue
        sub = Index([q for q in poems if q["id"] not in held])
        recs, _ = process(sub, p)
        gold = [r for r in recs if r["tier"] == "gold"]
        if gold:
            fp_pages += 1
            fp_recs += len(gold)
            fp_lines += sum(r["matched_lines"] for r in gold)
            for r in gold[:2]:
                print(f"  FP {os.path.basename(p)} -> {r['poem_title']}  "
                      f"verses={r['distinct_verses']} mono={r['monotone_ratio']} "
                      f"jac={r['mean_jaccard']} cer={r['cer_matched_span']}")
    tot_pages = sum(1 for p in paths if truth[p])
    print(f"\npages with a false 'gold' after withholding: {fp_pages}/{tot_pages} "
          f"({100*fp_pages/max(1,tot_pages):.1f}%)")
    print(f"false gold records: {fp_recs} | false line pairs: {fp_lines}")
    if fp_pages == 0:
        print("\nNo false positives: the gates are structural, not similarity-based.")


if __name__ == "__main__":
    main()
