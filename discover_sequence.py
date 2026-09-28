#!/usr/bin/env python3
"""Discover weak line matches supported by two exact sequence anchors."""
import argparse
import hashlib
import json
import os
import tempfile

from ganjoor import load_index
from manufacture import process
from urdu_norm import norm_match

SCHEMA = "khusrau-sequence-discovery-v1"


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_for(path):
    stem = os.path.splitext(path)[0]
    for extension in (".jpg", ".jpeg", ".png"):
        image = stem + extension
        if os.path.isfile(image):
            return os.path.abspath(image)
    return None


def _candidate_score(ix, text, poem_id, verse):
    tokens = ix.trigrams(" ".join(norm_match(text))) - ix.hot
    reference = ix.poems[poem_id]["verses"][verse]
    reference_tokens = ix.trigrams(" ".join(norm_match(reference)))
    if not tokens or not reference_tokens:
        return 0.0, reference
    overlap = len(tokens & reference_tokens)
    score = overlap / (len(tokens) + len(reference_tokens) - overlap)
    return score, reference


def discover(ix, path, discovery_floor=0.30, min_jaccard=0.55,
             min_lines=4, min_monotone=0.80, review_lines=2):
    records, _ = process(
        ix, path, min_jaccard=min_jaccard, min_lines=min_lines,
        min_monotone=min_monotone, review_lines=review_lines)
    gold = {record["poem_id"]: record for record in records
            if record["tier"] == "gold"}
    if not gold:
        return []

    with open(path, encoding="utf-8") as f:
        lines = [
            line.strip() for line in f
            if line.strip() and not line.startswith("##")
        ]
    discoveries = []
    for poem_id, record in gold.items():
        anchors = sorted(record["pairs"], key=lambda pair: pair["line"])
        for left, right in zip(anchors, anchors[1:]):
            line_gap = right["line"] - left["line"]
            verse_gap = right["verse"] - left["verse"]
            if line_gap <= 1 or line_gap != verse_gap:
                continue
            offset = left["verse"] - left["line"]
            for line_number in range(left["line"] + 1, right["line"]):
                verse = line_number + offset
                score, reference = _candidate_score(
                    ix, lines[line_number], poem_id, verse)
                top = ix.match_line(lines[line_number], min_jaccard=0)
                top_summary = None
                if top:
                    top_summary = {
                        "poem_id": top[0],
                        "verse": top[1],
                        "jaccard": round(top[2], 4),
                        "ref": top[3],
                    }
                image = _image_for(path)
                discoveries.append({
                    "schema": SCHEMA,
                    "page": os.path.basename(path),
                    "transcription_path": os.path.abspath(path),
                    "transcription_sha256": _sha256(path),
                    "image_path": image,
                    "image_sha256": _sha256(image) if image else None,
                    "poem_id": poem_id,
                    "poem_title": record["poem_title"],
                    "candidate": {
                        "line": line_number,
                        "verse": verse,
                        "jaccard": round(score, 4),
                        "hyp": lines[line_number],
                        "ref": reference,
                        "is_global_top": bool(
                            top and top[0] == poem_id and top[1] == verse),
                    },
                    "global_top": top_summary,
                    "anchors": {"left": left, "right": right},
                    "evidence": {
                        "method": "two-sided-exact-offset",
                        "offset": offset,
                        "discovery_floor": discovery_floor,
                        "acceptance_threshold": min_jaccard,
                        "exclusion": (
                            "below_discovery_floor"
                            if score < discovery_floor
                            else "below_acceptance_threshold"
                            if score < min_jaccard
                            else "not_excluded"
                        ),
                    },
                    "adjudication": None,
                })
    return sorted(
        discoveries,
        key=lambda row: (row["page"], row["candidate"]["line"], row["poem_id"]),
    )


def write_discovery(rows, output):
    output = os.path.abspath(output)
    directory = os.path.dirname(output)
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".khusrau-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("categories", help="comma-separated Ganjoor category ids")
    parser.add_argument("pages", nargs="+", help="rough transcription files")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--discovery-floor", type=float, default=0.30)
    parser.add_argument("--min-jaccard", type=float, default=0.55)
    args = parser.parse_args(argv)
    categories = [int(value) for value in args.categories.split(",") if value.strip()]
    ix = load_index(*categories)
    rows = []
    for page in args.pages:
        rows.extend(discover(
            ix, page, discovery_floor=args.discovery_floor,
            min_jaccard=args.min_jaccard))
    write_discovery(rows, args.output)
    print(f"wrote {args.output}: {len(rows)} sequence-supported candidates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
