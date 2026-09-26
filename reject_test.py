#!/usr/bin/env python3
"""V0b: the precision question. Can a threshold reject a page whose text is absent?

selftest_align.py asks "given that the true ghazal IS in the corpus, is it found?"
and answers yes. That is the recall half, and on its own it is close to worthless
for deciding the architecture, because a scorer that always returns its single best
guess scores perfectly on it while being unusable in production.

The real situation is the opposite one. A Momin kulliyat is ~456 pages and Rekhta
gives us 49 of his ghazals; a Ghalib divan is better covered but still not total.
So MOST pages have no counterpart, and the pipeline's job is mostly to say "no".
That is exactly where siraj/ found distributional features collapse and had to buy
a model judge for every candidate.

So: run each trial twice against identical input.
  present -- the true ghazal is in the candidate set (the score we must accept)
  absent  -- the true ghazal is removed (the score we must reject)
If those two score distributions separate, one threshold does the job and the judge
is only needed for the overlap. If they do not, siraj's architecture is required
and the judge is a per-candidate cost we have to plan for.

Scores are normalized per query so page length does not drive the comparison.
"""
import os
import random
import sys
from collections import defaultdict

from corpus import load, radif_families
from selftest_align import (
    PAGE_LINES, AnchorTypes, HemistichVote, IdfOverlap, corrupt,
)
from urdu_norm import norm_match

TRIALS = int(os.environ.get("TRIALS", "150"))
NOISES = [float(x) for x in os.environ.get("NOISES", "0.15,0.30,0.45").split(",")]
SEED = int(os.environ.get("SEED", "0"))


def normalized(hits, page_lines, scorer):
    """Per-query normalization so short and long pages are comparable."""
    if not hits:
        return 0.0
    best = max(hits.values())
    if scorer.name == "hemistich_vote":
        return best / max(1, len(page_lines))          # mean Jaccard per line
    n_types = len({t for ln in page_lines for t in ln})
    return best / max(1, n_types)                      # fraction of query types hit


def sweep(present, absent):
    """Best single threshold, and the error rates it buys.

    Reported as the operating point a real pipeline would run at: pick the
    threshold maximizing accuracy, then show what it costs on each side.
    """
    if not present or not absent:
        return None
    cuts = sorted(set(present + absent))
    best = None
    for c in cuts:
        tp = sum(1 for s in present if s >= c)
        fp = sum(1 for s in absent if s >= c)
        acc = (tp + (len(absent) - fp)) / (len(present) + len(absent))
        if best is None or acc > best[1]:
            best = (c, acc, tp / len(present), fp / len(absent))
    # separation: does the absent distribution reach into the present one?
    present_s, absent_s = sorted(present), sorted(absent)
    overlap = sum(1 for s in absent_s if s >= present_s[len(present_s) // 20])
    return {
        "cut": best[0], "acc": best[1], "recall": best[2], "fpr": best[3],
        "present_p05": present_s[len(present_s) // 20],
        "present_med": present_s[len(present_s) // 2],
        "absent_med": absent_s[len(absent_s) // 2],
        "absent_max": absent_s[-1],
        "absent_above_present_p05": overlap / len(absent_s),
    }


def main():
    poet_filter = sys.argv[1] if len(sys.argv) > 1 else None
    ghazals = load(poet=poet_filter)
    fam = radif_families(ghazals)
    by_poet = defaultdict(set)
    for g in ghazals:
        by_poet[g.poet].add(g.gid)
    all_gids = {g.gid for g in ghazals}

    scorers = [AnchorTypes(ghazals), IdfOverlap(ghazals), HemistichVote(ghazals)]
    pool = [g for g in ghazals if len(g.lines) >= PAGE_LINES[0]
            and g.radif and len(fam.get(g.radif, [])) >= 2]
    print(f"corpus {len(ghazals)} ghazals | pool {len(pool)} | trials {TRIALS}\n")

    hdr = (f"{'scope':6s} {'noise':>5s} {'scorer':16s} {'cut':>6s} {'recall':>7s} "
           f"{'FPR':>7s} {'pres_p05':>9s} {'abs_med':>8s} {'abs_max':>8s} {'overlap':>8s}")
    for scope in ("scoped", "open"):
        print(hdr)
        print("-" * len(hdr))
        for noise in NOISES:
            rnd = random.Random(SEED)
            trials = []
            for _ in range(TRIALS):
                g = rnd.choice(pool)
                k = min(rnd.randint(*PAGE_LINES), len(g.lines))
                s = rnd.randrange(0, max(1, len(g.lines) - k + 1))
                page = [norm_match(corrupt(l, noise, rnd))
                        for l in g.lines_raw[s:s + k]]
                page = [p for p in page if p]
                if page:
                    trials.append((g, page))

            for sc in scorers:
                present, absent = [], []
                for g, page in trials:
                    base = by_poet[g.poet] if scope == "scoped" else all_gids
                    present.append(normalized(sc.score(page, base), page, sc))
                    # identical page, true ghazal withheld -- the rejection case
                    held = base - {g.gid}
                    absent.append(normalized(sc.score(page, held), page, sc))
                r = sweep(present, absent)
                if not r:
                    continue
                print(f"{scope:6s} {noise:5.2f} {sc.name:16s} {r['cut']:6.3f} "
                      f"{100*r['recall']:6.1f}% {100*r['fpr']:6.1f}% "
                      f"{r['present_p05']:9.3f} {r['absent_med']:8.3f} "
                      f"{r['absent_max']:8.3f} {100*r['absent_above_present_p05']:7.1f}%")
        print()

    print("recall  = present pages accepted at the best threshold (gold pairs kept)")
    print("FPR     = absent pages wrongly accepted (poisoned gold -- the fatal error)")
    print("overlap = absent scores reaching the 5th percentile of present scores;")
    print("          this is the band a model judge would have to arbitrate")


if __name__ == "__main__":
    main()
