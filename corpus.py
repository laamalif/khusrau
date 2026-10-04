#!/usr/bin/env python3
"""Load the Rekhta-derived clean-text side and expose the structure that makes
Urdu alignment different from the Afghan case.

In siraj/ the reference was ONE continuous text (757 pp of Tarzi's collected
articles), so a page could vote for a start offset in a single token stream.
Here the reference is ~3,000 discrete short documents in no canonical order, so
there is no offset to vote for. Alignment becomes retrieval over documents.

The adversarial structure also changes. siraj's false positives came from shared
author/topic/era. Here they come from the RADIF: every ghazal in a radif family
ends every second hemistich with the same words, in the same position. That is
structured, positional agreement, which is exactly what an order-sensitive check
like LIS is supposed to detect -- so LIS is not just uninformative here (as siraj
found) but actively misleading.

Run directly to print the radif-family size distribution, i.e. how many decoys a
true match has to beat.
"""
import os
import re
import sys
from collections import Counter, defaultdict

from urdu_norm import norm_match


def _default_corpus() -> str:
    env = os.environ.get("REKHTA_CORPUS")
    if env:
        return env
    project_corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), "work", "corpus")
    if os.path.isdir(project_corpus):
        return project_corpus
    config_home = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    user_corpus = os.path.join(config_home, "khusrau", "corpus")
    if os.path.isdir(user_corpus):
        return user_corpus
    return project_corpus


CORPUS = _default_corpus()

_SLUG = re.compile(r"^(?P<slug>.+?)-(?P<poet>[a-z]+-[a-z]+)-ghazals(?:-\d+)?$")


def _poet_from_name(stem: str) -> str:
    m = _SLUG.match(stem)
    return m.group("poet") if m else "unknown"


def radif(lines: list[list[str]]) -> tuple[str, ...]:
    """Longest common token suffix of the rhyming hemistichs (odd indices).

    A ghazal rhymes on lines 1, 3, 5, ... (and line 0, the matla's first
    hemistich, also rhymes). Using only the odd lines avoids being thrown by a
    ghazal whose matla is missing from the source text.
    """
    rhyming = [ln for i, ln in enumerate(lines) if i % 2 == 1 and ln]
    if len(rhyming) < 2:
        return ()
    suffix = list(rhyming[0])
    for ln in rhyming[1:]:
        keep = 0
        while keep < min(len(suffix), len(ln)) and suffix[-1 - keep] == ln[-1 - keep]:
            keep += 1
        suffix = suffix[len(suffix) - keep:] if keep else []
        if not suffix:
            break
    return tuple(suffix[-3:])          # a radif longer than 3 tokens is rare


class Ghazal:
    __slots__ = ("gid", "poet", "lines_raw", "lines", "radif", "tokens")

    def __init__(self, gid, poet, lines_raw):
        self.gid = gid
        self.poet = poet
        self.lines_raw = lines_raw
        self.lines = [norm_match(l) for l in lines_raw]
        self.radif = radif(self.lines)
        self.tokens = [t for ln in self.lines for t in ln]

    def __repr__(self):
        return f"<Ghazal {self.gid} {self.poet} {len(self.lines)}L radif={' '.join(self.radif)!r}>"


def load(path: str | None = None, poet: str | None = None) -> list[Ghazal]:
    if path is None:
        path = CORPUS
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"reference corpus directory not found: {path} "
            "(set REKHTA_CORPUS environment variable or provide a valid path)"
        )
    out = []
    for name in sorted(os.listdir(path)):
        if not name.endswith(".txt"):
            continue
        stem = name[:-4]
        p = _poet_from_name(stem)
        if poet and poet not in p:
            continue
        with open(os.path.join(path, name), encoding="utf-8") as f:
            lines_raw = [l.strip() for l in f if l.strip()]
        if len(lines_raw) < 4:
            continue
        out.append(Ghazal(stem, p, lines_raw))
    return out


def radif_families(ghazals: list[Ghazal]) -> dict[tuple, list[Ghazal]]:
    fam = defaultdict(list)
    for g in ghazals:
        if g.radif:
            fam[g.radif].append(g)
    return dict(fam)


def main():
    poet = sys.argv[1] if len(sys.argv) > 1 else None
    gs = load(poet=poet)
    print(f"corpus: {len(gs)} ghazals, {sum(len(g.lines) for g in gs):,} hemistichs, "
          f"{sum(len(g.tokens) for g in gs):,} tokens"
          + (f"  [poet filter: {poet}]" if poet else ""))
    print(f"poets: {len(set(g.poet for g in gs))}")

    no_radif = [g for g in gs if not g.radif]
    print(f"radif detected: {len(gs) - len(no_radif)}/{len(gs)} "
          f"({len(no_radif)} with none -- qafiya-only ghazals)")

    fam = radif_families(gs)
    sizes = Counter(len(v) for v in fam.values())
    print("\nradif-family size -> how many families that size:")
    for size in sorted(sizes, reverse=True)[:12]:
        print(f"  {size:4d} ghazals share a radif : {sizes[size]} family/ies")
    top = sorted(fam.items(), key=lambda kv: -len(kv[1]))[:10]
    print("\nlargest radif families (the decoy pool a true match must beat):")
    for r, v in top:
        poets = len(set(g.poet for g in v))
        print(f"  {len(v):4d} ghazals  radif={' '.join(r)!r:28s} across {poets} poet(s)")

    # How much of a hemistich IS the radif? If most of the line is radif, there is
    # very little distinctive signal left to anchor on.
    share = []
    for g in gs:
        if not g.radif:
            continue
        for i, ln in enumerate(g.lines):
            if i % 2 == 1 and len(ln) > len(g.radif):
                share.append(len(g.radif) / len(ln))
    if share:
        share.sort()
        print(f"\nradif share of a rhyming hemistich's tokens: "
              f"median {share[len(share)//2]:.2f}  p90 {share[int(.9*len(share))]:.2f}")


if __name__ == "__main__":
    main()
