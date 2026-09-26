#!/usr/bin/env python3
"""Run the normalization regression corpus against the frozen profiles.

The corpus is DATA (normalization_tests/*.jsonl), not assertions embedded in code, so
the profiles are tested against a stated specification rather than the specification
being whatever the implementation happens to do. Every case carries a `reason` and a
`source`, so a failure tells you which scholarly claim broke.

`must_not_merge` is the more important file. A false divergence is a recoverable
annoyance -- you investigate it later. A false merge deletes a textual distinction and
by the time it is downstream the information is gone.

Also checks that retrieval-v1 still reproduces the behaviour the existing gold sets
were built with, by comparing against urdu_norm on the real corpus.

Usage: test_norm.py [--corpus]      # --corpus adds the 16k-hemistich fidelity check
"""
import glob
import json
import os
import sys

from norm_profiles import (APPARATUS_ID, APPARATUS_V2_ID, LEVELS, RETRIEVAL_ID,
                           agreement_level, agreement_level_v2,
                           retrieval_ident, retrieval_merges)

TESTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "normalization_tests")
_RANK = {"identical": 0}
_RANK.update({n: i + 1 for i, n in enumerate(LEVELS)})


def load(name):
    p = os.path.join(TESTS, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def check(case):
    """-> (ok, detail)."""
    a, b, prof = case["a"], case["b"], case["profile"]
    if prof == "retrieval":
        # Retrieval has two forms and they answer different questions: the token list
        # keeps word boundaries, the ident form drops them. A segmentation case can
        # only merge in the ident form, so the case must say which it means.
        mode = case.get("mode", "tokens")
        if mode == "ident":
            merged = retrieval_ident(a) == retrieval_ident(b)
        else:
            merged = retrieval_merges(a, b)
        want = case["expect"] == "merge"
        return merged == want, f"merged={merged} want={want} mode={mode}"
    # 'apparatus2' cases exercise apparatus-v2, which folds digit blocks at the encoding
    # rung. They are a separate profile rather than added to the apparatus cases, because v1
    # is frozen and MUST still report these as distinct.
    lvl = (agreement_level_v2(a, b) if prof == "apparatus2" else agreement_level(a, b))
    if case["expect"] == "distinct":
        return lvl is None, f"agreement_level={lvl} want=None"
    if lvl is None:
        return False, "agreement_level=None but expected merge"
    want = case.get("max_level")
    if want is None:
        return True, f"agreement_level={lvl}"
    # merging EARLIER than allowed is fine; merging later than allowed is not
    ok = _RANK.get(lvl, 99) <= _RANK.get(want, 99)
    return ok, f"agreement_level={lvl} max_level={want}"


def main():
    print(f"{RETRIEVAL_ID}\n{APPARATUS_ID}\n")
    failed = 0
    for fname, label in (("must_merge.jsonl", "MUST MERGE"),
                         ("must_not_merge.jsonl", "MUST NOT MERGE")):
        cases = load(fname)
        bad = []
        for c in cases:
            ok, detail = check(c)
            if not ok:
                bad.append((c, detail))
        status = "PASS" if not bad else f"FAIL ({len(bad)})"
        print(f"{label:16s} {len(cases):3d} cases  {status}")
        for c, detail in bad:
            print(f"    {c['id']:16s} {c['a']!r} vs {c['b']!r} [{c['profile']}]")
            print(f"      {detail}")
            print(f"      reason: {c['reason']}")
            print(f"      source: {c['source']}")
        failed += len(bad)

    if "--corpus" in sys.argv:
        # retrieval-v1 is frozen to the behaviour the 2026-09-04 gold sets used.
        # Any drift here silently invalidates work/gold_*.jsonl.
        from urdu_norm import norm_ident, norm_match
        from norm_profiles import retrieval_ident, retrieval_tokens
        lines = []
        for d in ("rekhta.org/ghalib/ghazals", "rekhta.org/ghalib/unpublished-ghazal",
                  "rekhta.org/ghalib/letters"):
            for f in glob.glob(d + "/*.txt"):
                lines += [l.strip() for l in open(f, encoding="utf-8") if l.strip()]
        for f in glob.glob("work/ganjoor/*/*/poem_*.json")[:400]:
            lines += [v for v in json.load(open(f, encoding="utf-8"))["verses"]
                      if v.strip()]
        if not lines:
            print("\nRETRIEVAL FIDELITY  FAIL (no corpus files found)")
            return 1
        drift = sum(1 for l in lines
                    if norm_match(l) != retrieval_tokens(l)
                    or norm_ident(l) != retrieval_ident(l))
        print(f"\nRETRIEVAL FIDELITY  {len(lines):,} real hemistichs  "
              f"{'PASS' if not drift else f'FAIL ({drift} drifted)'}")
        failed += drift

    print(f"\n{'ALL PASS' if not failed else f'{failed} FAILURES'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
