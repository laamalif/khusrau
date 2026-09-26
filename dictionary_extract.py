#!/usr/bin/env python3
"""Extract candidate dictionary entries from page images and OCR sidecars."""
import argparse
import json
import os
import re
import tempfile

from vertex import generate, image_part

SCHEMA = "khusrau-dictionary-entry-v1"

SYSTEM = """You extract entries from a modern printed Urdu literary dictionary.
The page images are authoritative; supplied OCR is only a fallible reading aid.
Return one JSON object with an "entries" array and no commentary.

Read ordinary Urdu prose from right to left and top to bottom. The pages are
single-column, but verse may print two hemistichs side by side. For such verse,
pair each row and record the right hemistich before the left hemistich. Bold text
at the right edge normally begins a dictionary headword. Parenthesized English
immediately following a headword is an English equivalent, not part of the Urdu
headword. Running heads and page numbers are not entries.

Merge entries continued across supplied pages. Include an incomplete entry at
the first or last page, but add "starts_before_range" or "continues_after_range"
to quality.flags. Never invent missing text.

Each entry must contain:
- headword: the printed Urdu headword only
- english_terms: printed English equivalents immediately following the headword
- definition: prose definition, preserving original wording and spelling
- examples: array of objects with type ("couplet", "verse", or "quotation"),
  lines in reading order, and attribution (null when absent)
- cross_references: targets explicitly introduced by دیکھیے
- raw_text: exact complete entry transcription, including headword, equivalents,
  prose, examples, and attributions; preserve meaningful line breaks
- pages: ordered array of three-digit supplied page ids on which the entry occurs
- entry_type: "definition" or "cross_reference"
- quality: {"score": number from 0 to 1, "flags": array of short ASCII strings}

Do not modernize, correct, summarize, translate, or normalize Urdu. Lower quality
for uncertain headword boundaries, illegible text, uncertain verse pairing, or
suspected OCR/image conflict. Do not omit a low-quality entry."""


def _atomic_json(path, value):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_jsonl(path, records):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".jsonl", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _page_id(path):
    page = os.path.splitext(os.path.basename(path))[0]
    if not re.fullmatch(r"\d{3}", page):
        raise ValueError(f"page image must have a three-digit basename: {path}")
    return page


def request_parts(images):
    parts = [{"text": "Extract the supplied consecutive dictionary pages."}]
    for image in images:
        page = _page_id(image)
        text_path = os.path.splitext(image)[0] + ".txt"
        if not os.path.isfile(text_path):
            raise FileNotFoundError(text_path)
        with open(text_path, encoding="utf-8") as f:
            ocr = f.read()
        parts.extend([
            {"text": f"PAGE {page} IMAGE:"},
            image_part(image),
            {"text": f"PAGE {page} EXISTING OCR (fallible):\n{ocr}"},
        ])
    return parts


def validate_entries(value, images, dataset):
    if not isinstance(value, dict) or not isinstance(value.get("entries"), list):
        raise ValueError("model output must be an object containing an entries array")
    allowed_pages = {_page_id(path) for path in images}
    records = []
    page_ordinals = {}
    for sequence, entry in enumerate(value["entries"], 1):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {sequence} is not an object")
        for key in ("headword", "raw_text"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError(f"entry {sequence} has invalid {key}")
        definition = entry.get("definition")
        if not isinstance(definition, str):
            raise ValueError(f"entry {sequence} has invalid definition")
        if not definition.strip() and entry.get("entry_type") != "cross_reference":
            raise ValueError(f"entry {sequence} has empty definition")
        pages = entry.get("pages")
        if not isinstance(pages, list) or not pages or not set(pages) <= allowed_pages:
            raise ValueError(f"entry {sequence} has invalid pages")
        english_terms = entry.get("english_terms")
        if english_terms is None:
            english_terms = []
        elif isinstance(english_terms, str):
            english_terms = [english_terms]
        if not all(isinstance(term, str) and term.strip() for term in english_terms):
            raise ValueError(f"entry {sequence} has invalid english_terms")
        for key in ("examples", "cross_references"):
            if not isinstance(entry.get(key), list):
                raise ValueError(f"entry {sequence} has invalid {key}")
        quality = entry.get("quality")
        if not isinstance(quality, dict) or not isinstance(quality.get("flags"), list):
            raise ValueError(f"entry {sequence} has invalid quality")
        score = quality.get("score")
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"entry {sequence} has invalid quality score")
        clean = dict(entry)
        first_page = pages[0]
        page_ordinals[first_page] = page_ordinals.get(first_page, 0) + 1
        clean["schema"] = SCHEMA
        clean["id"] = f"{dataset}-{first_page}-{page_ordinals[first_page]:03d}"
        clean["sequence"] = sequence
        clean["english_terms"] = english_terms
        clean["quality"] = {
            "status": "candidate",
            "score": score,
            "flags": quality["flags"],
        }
        records.append(clean)
    if not records:
        raise ValueError("model returned no dictionary entries")
    return records


def publish_response(response_path, output, images, dataset="farhangiadabiyat"):
    with open(response_path, encoding="utf-8") as f:
        response = json.load(f)
    if response.get("finish") != "STOP":
        raise RuntimeError(f"dictionary extraction did not finish: {response.get('finish')}")
    try:
        value = json.loads(response.get("text", ""))
    except json.JSONDecodeError as error:
        raise ValueError(f"dictionary extraction returned invalid JSON: {error}") from error
    records = validate_entries(value, images, dataset)
    _atomic_jsonl(output, records)
    return records


def extract(images, output, response_path, dataset="farhangiadabiyat",
            model="gemini-3.8-flash", thinking="low"):
    if os.path.exists(response_path):
        raise FileExistsError(f"response artifact already exists: {response_path}")
    response = generate(
        request_parts(images), model=model, system=SYSTEM, max_tokens=32768,
        json_out=True, temperature=0, thinking=thinking)
    response["request"] = {
        "model": model,
        "thinking": thinking,
        "pages": [_page_id(path) for path in images],
    }
    _atomic_json(response_path, response)
    return publish_response(response_path, output, images, dataset)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("images", nargs="+")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--dataset", default="farhangiadabiyat")
    parser.add_argument("--model", default="gemini-3.8-flash")
    parser.add_argument("--thinking", default="low")
    parser.add_argument("--reuse-response", action="store_true",
                        help="validate and publish the existing response without an API call")
    args = parser.parse_args(argv)
    if args.reuse_response:
        records = publish_response(args.response, args.output, args.images, args.dataset)
    else:
        records = extract(
            args.images, args.output, args.response, args.dataset,
            args.model, args.thinking)
    print(f"wrote {args.output}: {len(records)} candidate entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
