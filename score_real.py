#!/usr/bin/env python3
"""V1a: score a REAL read of a real page against a born-digital reference.

This replaces the synthetic noise axis in selftest_align.py with a measured number.
§15 of FINDINGS.md names that number as the one thing the whole method depends on:
the method works below roughly 0.30-0.35 token corruption and collapses by 0.45, and
until now every rate we had was a synthetic corruption probability, not a read.

Usage:  score_real.py <hypothesis.txt> <reference.txt>
Both files: one line per hemistich, aligned line-for-line.

Reports CER at both normalization strengths, because the gap between them IS the
orthographic-recension floor (§3.2, §13.4): an Urdu katib setting Persian writes
ہ ے بے, Ganjoor holds ه ی بی. Folding those is legitimate for matching and dishonest
for scoring, so both numbers are shown and neither is called "the" CER.
"""
import sys

from urdu_norm import cer, norm_match, norm_score


def _seq_lev(a, b):
    """Edit distance over token SEQUENCES (not characters)."""
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_error_rate(hyp_lines, ref_lines):
    """WER: token edit distance / reference tokens, after MATCH folding.

    Must be edit distance, not positional comparison. The dominant real error on
    these pages is WORD SEGMENTATION -- a katib writes درنورد, Ganjoor writes
    در نورد -- and a positional comparison treats one boundary shift as corrupting
    every token after it. That reported 0.46 on a page whose CER was 0.026.
    """
    d = tot = 0
    for h, r in zip(hyp_lines, ref_lines):
        ht, rt = norm_match(h), norm_match(r)
        d += _seq_lev(ht, rt)
        tot += len(rt)
    return d / max(1, tot), d, tot


def spaceless_cer(hyp_lines, ref_lines):
    """CER with ALL spaces removed, so word-segmentation disagreement cannot count.

    Isolates genuine character misreads -- the errors an HTR model would actually
    make -- from a tokenization convention we do not care about.
    """
    a = "".join(t for l in hyp_lines for t in norm_match(l))
    b = "".join(t for l in ref_lines for t in norm_match(l))
    return cer(a, b)


def main():
    hyp = [l.strip() for l in open(sys.argv[1], encoding="utf-8") if l.strip()]
    ref = [l.strip() for l in open(sys.argv[2], encoding="utf-8") if l.strip()]
    if len(hyp) != len(ref):
        print(f"WARNING: {len(hyp)} hyp lines vs {len(ref)} ref lines; "
              f"scoring the first {min(len(hyp), len(ref))}", file=sys.stderr)
    n = min(len(hyp), len(ref))
    hyp, ref = hyp[:n], ref[:n]

    m_hyp = " ".join(t for l in hyp for t in norm_match(l))
    m_ref = " ".join(t for l in ref for t in norm_match(l))
    s_hyp = " ".join(t for l in hyp for t in norm_score(l))
    s_ref = " ".join(t for l in ref for t in norm_score(l))

    wer, d, tot = word_error_rate(hyp, ref)

    print(f"lines scored            {n}")
    print(f"reference tokens        {tot}")
    print(f"CER (norm_match)        {cer(m_hyp, m_ref):.4f}   <- folded; the matching view")
    print(f"CER (norm_score)        {cer(s_hyp, s_ref):.4f}   <- strict; incl. recension floor")
    print(f"CER spaceless           {spaceless_cer(hyp, ref):.4f}   <- genuine misreads only")
    print(f"WER (edit distance)     {wer:.4f}   ({d}/{tot})  <- compare to selftest `noise`")
    print()

    print("per-line (CER on norm_match) --")
    for i, (h, r) in enumerate(zip(hyp, ref)):
        a = " ".join(norm_match(h))
        b = " ".join(norm_match(r))
        flag = "" if a == b else "  <-- differs"
        print(f"  {i:2d}  {cer(a, b):.3f}{flag}")
        if a != b:
            print(f"        hyp: {a}")
            print(f"        ref: {b}")


if __name__ == "__main__":
    main()
