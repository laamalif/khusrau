#!/usr/bin/env python3
"""Locate a reference poem inside a whole-page transcription, then score that span.

A page carries material the reference never contains -- the tail of the previous
ghazal, section headings, the start of the next. Scoring the whole page against one
ghazal charges the reader for text with no counterpart, which is the mistake siraj
called out (§ "Always report CER over the parallel span, never whole-page").

Both sides are one hemistich per line, so the span is found by sliding the reference
over the hypothesis and taking the offset with the lowest mean per-line CER. That is
the line-level analogue of siraj's offset voting, and it is cheap enough to be exact
rather than sampled.

Usage: align_verse.py <hypothesis.txt> <reference.txt>
"""
import sys

from urdu_norm import cer, norm_match
from score_real import spaceless_cer, word_error_rate


def load(path, drop_headings=True):
    out = []
    for l in open(path, encoding="utf-8"):
        l = l.strip()
        if not l:
            continue
        if drop_headings and l.startswith("##"):
            continue
        out.append(l)
    return out


def best_offset(hyp, ref):
    """Offset of `ref` within `hyp` minimising mean per-line folded CER."""
    if len(hyp) < len(ref):
        return 0, float("inf")
    best, best_off = float("inf"), 0
    for off in range(len(hyp) - len(ref) + 1):
        tot = 0.0
        for i, r in enumerate(ref):
            a = " ".join(norm_match(hyp[off + i]))
            b = " ".join(norm_match(r))
            tot += cer(a, b)
        mean = tot / len(ref)
        if mean < best:
            best, best_off = mean, off
    return best_off, best


def main():
    hyp_all = load(sys.argv[1])
    ref = load(sys.argv[2])
    off, mean_cer = best_offset(hyp_all, ref)
    hyp = hyp_all[off:off + len(ref)]

    print(f"hypothesis lines        {len(hyp_all)}  (span found at offset {off})")
    print(f"reference lines         {len(ref)}")
    wer, d, tot = word_error_rate(hyp, ref)
    m_hyp = " ".join(t for l in hyp for t in norm_match(l))
    m_ref = " ".join(t for l in ref for t in norm_match(l))
    print(f"reference tokens        {tot}")
    print(f"CER (norm_match)        {cer(m_hyp, m_ref):.4f}")
    print(f"CER spaceless           {spaceless_cer(hyp, ref):.4f}")
    print(f"WER (edit distance)     {wer:.4f}   ({d}/{tot})")
    exact = sum(1 for h, r in zip(hyp, ref)
                if norm_match(h) == norm_match(r))
    print(f"exact lines (folded)    {exact}/{len(ref)}")
    print()
    for i, (h, r) in enumerate(zip(hyp, ref)):
        a, b = " ".join(norm_match(h)), " ".join(norm_match(r))
        if a != b:
            print(f"  {i:2d} cer={cer(a,b):.3f}")
            print(f"     hyp: {a}")
            print(f"     ref: {b}")


if __name__ == "__main__":
    main()
