#!/usr/bin/env python3
"""Score an ordered run of OCR prose pages against one continuous reference text."""
import argparse
import json
import os
import tempfile

from rapidfuzz.distance import Levenshtein

from urdu_norm import norm_match, norm_score


def load_lines(paths):
    lines = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            lines.extend(line.strip() for line in f if line.strip() and not line.startswith("##"))
    return lines


def score(pages, reference):
    hyp_lines = load_lines(pages)
    ref_lines = load_lines([reference])
    hyp_tokens = [token for line in hyp_lines for token in norm_match(line)]
    ref_tokens = [token for line in ref_lines for token in norm_match(line)]
    token_distance = Levenshtein.distance(hyp_tokens, ref_tokens)

    folded_hyp = "".join(hyp_tokens)
    folded_ref = "".join(ref_tokens)
    strict_hyp = "".join(token for line in hyp_lines for token in norm_score(line))
    strict_ref = "".join(token for line in ref_lines for token in norm_score(line))
    folded_distance = Levenshtein.distance(folded_hyp, folded_ref)
    strict_distance = Levenshtein.distance(strict_hyp, strict_ref)

    return {
        "pages": [os.path.abspath(path) for path in pages],
        "reference": os.path.abspath(reference),
        "hypothesis_lines": len(hyp_lines),
        "reference_paragraphs": len(ref_lines),
        "hypothesis_tokens": len(hyp_tokens),
        "reference_tokens": len(ref_tokens),
        "token_edit_distance": token_distance,
        "token_error_rate": token_distance / max(1, len(ref_tokens)),
        "folded_character_distance": folded_distance,
        "folded_character_error_rate": folded_distance / max(1, len(folded_ref)),
        "strict_character_distance": strict_distance,
        "strict_character_error_rate": strict_distance / max(1, len(strict_ref)),
    }


def write_score(result, output):
    output = os.path.abspath(output)
    directory = os.path.dirname(output)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary, output)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("pages", nargs="+")
    parser.add_argument("-o", "--output")
    args = parser.parse_args(argv)
    result = score(args.pages, args.reference)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        write_score(result, args.output)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
