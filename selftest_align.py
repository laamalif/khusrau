#!/usr/bin/env python3
"""V0: does distributional retrieval survive Urdu's radif decoys, at zero API cost?

siraj/ found that EVERY distributional discriminator failed on its pair, because
reference and target shared author, era and topic. It therefore had to pay a model
judge on every candidate. Before porting that architecture to Urdu we should find
out whether Urdu actually needs it, because if it does not, the judge is the single
largest cost line in the pipeline and can be spent only on the hard tail.

The Urdu decoy structure is harsher in kind than siraj's: a radif family makes
dozens of DIFFERENT ghazals end their rhyming hemistichs with the SAME tokens in
the SAME position. corpus.py measures families up to 162 ghazals.

Method (siraj's ALIGN_SELFTEST discipline, adapted). Take a real Rekhta ghazal as
ground truth, cut a contiguous run of hemistichs to imitate one divan page, corrupt
it at a known rate to imitate a lithograph read, and ask each scorer to find it
again among the other ghazals. We know the answer, so precision and recall are both
measurable and the gates become falsifiable.

The noise model is siraj's (word drop + trailing-character clip) plus a
substitution channel, because a nasta'liq reader's dominant error is a wrong-but-
plausible character (a lost or gained nuqta), not a dropped word.

Two candidate scopes, both reported:
  scoped -- candidates restricted to the same poet. Realistic: we always know which
            divan we are OCRing, so the reference set is one poet's ghazals.
  open   -- candidates are all ~3,000 ghazals. Stress test, and it measures whether
            the method could also recover attribution for an unidentified page.
"""
import os
import random
import sys
import time
from collections import Counter, defaultdict

from corpus import load, radif_families
from urdu_norm import norm_match

RARE_MAX = int(os.environ.get("RARE_MAX", "25"))     # df above this is not distinctive
RARE_MINLEN = 3
TRIALS = int(os.environ.get("TRIALS", "150"))
PAGE_LINES = (8, 16)                                  # hemistichs on a divan page
NOISES = [float(x) for x in os.environ.get(
    "NOISES", "0.0,0.15,0.30,0.45").split(",")]
SEED = int(os.environ.get("SEED", "0"))
TRIGRAM_DF_CAP = 3000                                 # skip trigrams this common


# ---------------------------------------------------------------- noise channel

URDU_CONFUSE = {
    # nuqta gains and losses: the dominant lithograph reading error
    "ب": "پ", "پ": "ب", "ت": "ٹ", "ٹ": "ت", "ج": "چ", "چ": "ج",
    "د": "ڈ", "ڈ": "د", "ر": "ز", "ز": "ر", "ڑ": "ر", "س": "ش",
    "ش": "س", "ع": "غ", "غ": "ع", "ف": "ق", "ق": "ف", "ک": "گ",
    "گ": "ک", "ن": "ں", "ں": "ن", "ی": "ے", "ے": "ی", "ہ": "ھ",
    "ھ": "ہ", "ح": "خ", "خ": "ح", "ط": "ظ", "ظ": "ط", "ص": "ض",
    "ض": "ص",
}


def corrupt(line: str, rate: float, rnd: random.Random) -> str:
    """Three channels at the same rate: word drop, trailing clip, nuqta confusion."""
    words = line.split()
    words = [w for w in words if rnd.random() > rate]
    out = []
    for w in words:
        if len(w) > 2 and rnd.random() < rate:
            w = w[:-1]                                  # trailing clip
        if w and rnd.random() < rate:
            i = rnd.randrange(len(w))
            sub = URDU_CONFUSE.get(w[i])
            if sub:
                w = w[:i] + sub + w[i + 1:]             # nuqta gain/loss
        if w:
            out.append(w)
    return " ".join(out)


# ---------------------------------------------------------------- scorers

class Scorer:
    """Base: build an index once, then score a page against candidate ghazals."""
    name = "base"

    def __init__(self, ghazals):
        self.ghazals = ghazals
        self.by_gid = {g.gid: g for g in ghazals}

    def score(self, page_lines, allowed):
        raise NotImplementedError


class AnchorTypes(Scorer):
    """siraj's `anchor_types`: how many DISTINCT rare tokens the two sides share."""
    name = "anchor_types"

    def __init__(self, ghazals):
        super().__init__(ghazals)
        df = Counter()
        for g in ghazals:
            df.update(set(g.tokens))
        self.df = df
        self.index = defaultdict(set)
        for g in ghazals:
            for t in set(g.tokens):
                if len(t) >= RARE_MINLEN and df[t] <= RARE_MAX:
                    self.index[t].add(g.gid)

    def score(self, page_lines, allowed):
        q = {t for ln in page_lines for t in ln
             if len(t) >= RARE_MINLEN and self.df.get(t, 0) <= RARE_MAX}
        hits = Counter()
        for t in q:
            for gid in self.index.get(t, ()):
                if gid in allowed:
                    hits[gid] += 1
        return hits


class IdfOverlap(Scorer):
    """Same overlap, weighted by inverse document frequency.

    A rare-token count treats a hapax and a 25-document token as equal evidence.
    Weighting should widen the gap between a true match and a radif decoy, because
    the radif itself is by construction the highest-df thing on the page.
    """
    name = "idf_overlap"

    def __init__(self, ghazals):
        super().__init__(ghazals)
        import math
        df = Counter()
        for g in ghazals:
            df.update(set(g.tokens))
        n = len(ghazals)
        self.w = {t: math.log(n / c) for t, c in df.items()}
        self.index = defaultdict(set)
        for g in ghazals:
            for t in set(g.tokens):
                if len(t) >= RARE_MINLEN and df[t] <= RARE_MAX:
                    self.index[t].add(g.gid)

    def score(self, page_lines, allowed):
        q = {t for ln in page_lines for t in ln}
        hits = Counter()
        for t in q:
            wt = self.w.get(t, 0.0)
            for gid in self.index.get(t, ()):
                if gid in allowed:
                    hits[gid] += wt
        return hits


class HemistichVote(Scorer):
    """Match at the level of the LINE, not the page.

    This is the adaptation that siraj's offset voting cannot express. A single
    correctly-read hemistich of 6-8 words is very nearly a unique fingerprint for
    one ghazal. So instead of one page-level score, each query hemistich finds its
    own best partner line by character-trigram Jaccard and votes for that line's
    ghazal, weighted by how good the match was. A radif decoy can only ever win the
    radif tail of a line, never the whole line, so the per-line ceiling caps the
    decoy's total in a way page-level bag-of-tokens scoring does not.
    """
    name = "hemistich_vote"

    def __init__(self, ghazals):
        super().__init__(ghazals)
        self.lines = []                    # (gid, trigram set)
        self.index = defaultdict(list)
        for g in ghazals:
            for ln in g.lines:
                if not ln:
                    continue
                tg = self._trigrams(" ".join(ln))
                if not tg:
                    continue
                lid = len(self.lines)
                self.lines.append((g.gid, tg))
                for t in tg:
                    self.index[t].append(lid)
        self.hot = {t for t, v in self.index.items() if len(v) > TRIGRAM_DF_CAP}

    @staticmethod
    def _trigrams(s):
        s = s.replace(" ", "_")
        return {s[i:i + 3] for i in range(len(s) - 2)}

    def score(self, page_lines, allowed):
        votes = Counter()
        for ln in page_lines:
            qt = self._trigrams(" ".join(ln))
            qt -= self.hot
            if not qt:
                continue
            overlap = Counter()
            for t in qt:
                for lid in self.index.get(t, ()):
                    overlap[lid] += 1
            best, best_gid = 0.0, None
            for lid, ov in overlap.items():
                gid, tg = self.lines[lid]
                if gid not in allowed:
                    continue
                j = ov / (len(qt) + len(tg) - ov)
                if j > best:
                    best, best_gid = j, gid
            if best_gid is not None and best >= 0.30:      # ignore junk partners
                votes[best_gid] += best
        return votes


# ---------------------------------------------------------------- harness

def run():
    poet_filter = sys.argv[1] if len(sys.argv) > 1 else None
    ghazals = load(poet=poet_filter)
    fam = radif_families(ghazals)
    by_poet = defaultdict(set)
    for g in ghazals:
        by_poet[g.poet].add(g.gid)
    all_gids = {g.gid for g in ghazals}
    print(f"corpus: {len(ghazals)} ghazals, {len(by_poet)} poets\n")

    t0 = time.time()
    scorers = [AnchorTypes(ghazals), IdfOverlap(ghazals), HemistichVote(ghazals)]
    print(f"indexes built in {time.time()-t0:.1f}s\n")

    # only ghazals long enough to cut a page from, and in a radif family with
    # decoys -- a ghazal with no competitor proves nothing about decoy pressure
    pool = [g for g in ghazals if len(g.lines) >= PAGE_LINES[0]
            and g.radif and len(fam.get(g.radif, [])) >= 2]
    print(f"trial pool: {len(pool)} ghazals (>= {PAGE_LINES[0]} hemistichs, "
          f"radif family >= 2)\n")

    hdr = (f"{'scope':6s} {'noise':>5s} {'scorer':16s} {'top1':>6s} {'top5':>6s} "
           f"{'margin_med':>10s} {'decoy=radif':>12s} {'n_contested':>12s}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for scope in ("scoped", "open"):
        for noise in NOISES:
            rnd = random.Random(SEED)
            # one fixed set of trials per (scope, noise) so scorers are compared
            # on identical inputs
            trials = []
            for _ in range(TRIALS):
                g = rnd.choice(pool)
                k = rnd.randint(*PAGE_LINES)
                k = min(k, len(g.lines))
                s = rnd.randrange(0, max(1, len(g.lines) - k + 1))
                raw = g.lines_raw[s:s + k]
                page = [norm_match(corrupt(l, noise, rnd)) for l in raw]
                page = [p for p in page if p]
                if page:
                    trials.append((g, page))

            for sc in scorers:
                top1 = top5 = 0
                margins, decoy_radif, decided = [], 0, 0
                for g, page in trials:
                    allowed = by_poet[g.poet] if scope == "scoped" else all_gids
                    hits = sc.score(page, allowed)
                    if not hits:
                        continue
                    ranked = hits.most_common()
                    order = [gid for gid, _ in ranked]
                    if order[0] == g.gid:
                        top1 += 1
                    if g.gid in order[:5]:
                        top5 += 1
                    # margin between truth and the best NON-truth candidate
                    truth = hits.get(g.gid, 0)
                    decoys = [(gid, s_) for gid, s_ in ranked if gid != g.gid]
                    if truth and decoys:
                        decided += 1
                        best_gid, best_s = decoys[0]
                        margins.append((truth - best_s) / truth)
                        if shares_radif(sc.by_gid.get(best_gid), g):
                            decoy_radif += 1
                n = len(trials)
                margins.sort()
                med = margins[len(margins) // 2] if margins else float("nan")
                pct = (100 * decoy_radif / decided) if decided else float("nan")
                # n_contested matters: a scorer that returns the truth and NOTHING
                # ELSE has no margin to report, which is the best possible outcome,
                # not a missing number. Without this column that reads as a bug.
                print(f"{scope:6s} {noise:5.2f} {sc.name:16s} "
                      f"{100*top1/n:5.1f}% {100*top5/n:5.1f}% "
                      f"{med:10.3f} {pct:11.1f}% {decided:9d}/{n}")
                rows.append((scope, noise, sc.name, top1 / n, top5 / n, med, pct, decided, n))
        print()
    return rows


def shares_radif(a, b):
    """Is the top decoy a radif sibling? High = the feared failure mode is real."""
    return bool(a and b and a.radif and a.radif == b.radif)


if __name__ == "__main__":
    run()
