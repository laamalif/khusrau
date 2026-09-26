#!/usr/bin/env python3
"""Conservatively extract dictionary entries from OCR text without model calls."""
import json
import os
import re
import tempfile

SCHEMA = "khusrau-local-dictionary-entry-v1"

_HEADER = re.compile(r"^\s*(?:فر[ہه]نگ[ِ ]*ادبیات|\d+)\s*$")
_ENGLISH = re.compile(
    r"^(?P<head>[^()[\]،۔:؛]{1,60}?)\s*"
    r"\((?P<english>[A-Za-z][A-Za-z0-9 ./,'&+-]{0,60})\)"
    r"(?P<body>\s+.*|\s*)$")
_DIRECT_REFERENCE = re.compile(
    r"^(?P<head>[^()[\]،۔:؛]{1,60}?)\s+دیکھیے\s+(?P<target>.+)$")
_LAYOUT = re.compile(r"^(?P<head>\S(?:.*?\S)?)\s*(?:\t+| {3,})(?P<body>\S.*)$")
_REFERENCE_TAIL = re.compile(r"[۔.]?\s*\(?دیکھیے\s+(.+?)\)?[۔.]?\s*$")
_BAD_HEAD = re.compile(r"[()[\].،۔:؛!?؟\"“”؎=]")
_BAD_HEAD_START = {
    "اور", "اس", "ان", "اگر", "جس", "جو", "کہ", "کے", "کی", "کا", "کو",
    "میں", "نے", "پر", "سے", "وہ", "یہ", "یا",
}
_BAD_HEAD_END = {
    "اور", "اگر", "جس", "جو", "کہ", "کے", "کی", "کا", "کو", "میں", "پر",
    "سے", "یا", "کسی", "چند", "ایک", "وغیرہ", "ہے", "ہیں", "تھا", "تھی",
    "تھے", "والا", "والی", "والے",
}
_BODY_START = re.compile(
    r"^(?:"
    r"\(\d+\)|"
    r"دیکھیے|بمعنی|لغوی|لفظی|مرادی|اصطلاحاً?|"
    r"وہ\s|جو\s|جس\s|کسی\s|کوئی\s|ایسا\s|ایسی\s|ایسے\s|"
    r"ایک\s|دو\s|تین\s|چار\s|"
    r"عربی\s|فارسی\s|اردو\s|انگریزی\s|سنسکرت\s|یونانی\s|ہندی\s|"
    r"شعر\s|شاعر\s|شاعری\s|نظم\s|نثر\s|ادب\s|فن\s|فنی\s|زبان\s|"
    r"مخصوص\s|مترادف\s|مخفف\s|صفت\s|زحاف\s|"
    r"بیان\s|یعنی\s|اگر\s|اسے\s|تشبیب\s|تشبیہ\s|شرح\s|شاہی\s|شمالی\s|لفظی\s|"
    r"جملے\s|تکلف\s|مولوی\s|موسیقانہ\s|مشبہ\s|بحر\s|مظاہر\s)"
)
_PREFERRED_BODY_START = re.compile(
    r"^(?:شاعر|فنی|شمالی|اسے|اگر|یعنی|تشبیب|تشبیہ|جملے|تکلف|مولوی|موسیقانہ|"
    r"مشبہ|بحر|مظاہر|صفت)\s"
)
_STRONG_EARLY_BODY_START = re.compile(
    r"^(?:\(\d+\)|وہ|جو|جس|کسی|کوئی|ایسا|ایسی|ایسے|ایک|دو|تین|چار|دیکھیے)\s"
)
_DIACRITICS = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_URDU_ORDER = {
    char: index for index, char in enumerate(
        "اآبپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھءیے", 1)
}
_CORRUPTION = re.compile(
    r'(?:"(?:name|description)"\s*:|(?:\b\S+\s+){0,1}Steve(?:\s+\S+){20,})'
)


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


def _valid_headword(value):
    value = value.strip()
    words = value.split()
    return (
        1 < len(value) <= 60
        and len(words) <= 7
        and not _BAD_HEAD.search(value)
        and not value.startswith(("(", "[", "ع "))
        and words[0] not in _BAD_HEAD_START
        and words[-1] not in _BAD_HEAD_END
        and "\u0600" <= value[0] <= "\u06ff"
        and any("\u0600" <= char <= "\u06ff" for char in value)
    )


def _split_heuristic(line):
    words = line.split()
    for index in range(1, min(7, len(words) - 1)):
        if words[index - 1] == words[index]:
            headword = " ".join(words[:index])
            body = " ".join(words[index:])
            if _valid_headword(headword) and _BODY_START.match(body):
                return headword, body
    fallback = None
    for count in range(1, min(7, len(words))):
        headword = " ".join(words[:count])
        body = " ".join(words[count:])
        if _valid_headword(headword) and _BODY_START.match(body):
            candidate = (headword, body)
            if _STRONG_EARLY_BODY_START.match(body):
                return candidate
            if _PREFERRED_BODY_START.match(body):
                return candidate
            fallback = fallback or candidate
    return fallback


def classify_boundary(line):
    """Return boundary metadata; only explicit/layout anchors are publishable."""
    match = _LAYOUT.match(line)
    if match and _valid_headword(match["head"]):
        return {
            "headword": match["head"].strip(),
            "body": match["body"].strip(),
            "english_terms": [],
            "rule": "layout",
            "publish": True,
        }
    match = _ENGLISH.match(line)
    if match and _valid_headword(match["head"]):
        return {
            "headword": match["head"].strip(),
            "body": match["body"].strip(),
            "english_terms": [match["english"].strip()],
            "rule": "english_term",
            "publish": True,
        }
    match = _DIRECT_REFERENCE.match(line)
    if match and _valid_headword(match["head"]):
        return {
            "headword": match["head"].strip(),
            "body": "دیکھیے " + match["target"].strip(),
            "english_terms": [],
            "rule": "direct_reference",
            "publish": True,
        }
    split = _split_heuristic(line)
    if split:
        return {
            "headword": split[0],
            "body": split[1],
            "english_terms": [],
            "rule": "heuristic",
            "publish": False,
        }
    return None


def _page_id(path):
    return os.path.splitext(os.path.basename(path))[0]


def _sort_key(value):
    value = _DIACRITICS.sub("", value).replace("ؤ", "و").replace("ئ", "ی")
    return tuple(_URDU_ORDER.get(char, 0 if char.isspace() else 1000 + ord(char))
                 for char in value)


def _same_initial(left, right):
    left_key = _sort_key(left)
    right_key = _sort_key(right)
    return bool(left_key and right_key and left_key[0] == right_key[0])


def _prefix(value, length=2):
    return tuple(weight for weight in _sort_key(value) if weight)[:length]


def _ordered_soft_boundaries(boundaries):
    """Find soft candidates alphabetically bounded by explicit anchors."""
    promoted = set()
    anchors = [index for index, (_, _, boundary) in enumerate(boundaries)
               if boundary["publish"]]
    for position, (_, _, boundary) in enumerate(boundaries):
        if boundary["publish"]:
            continue
        previous = next((anchor for anchor in reversed(anchors) if anchor < position), None)
        following = next((anchor for anchor in anchors if anchor > position), None)
        key = _sort_key(boundary["headword"])
        if previous is not None:
            previous_head = boundaries[previous][2]["headword"]
            if not _same_initial(boundary["headword"], previous_head):
                continue
            if key < _sort_key(previous_head):
                continue
        if following is not None:
            following_head = boundaries[following][2]["headword"]
            if not _same_initial(boundary["headword"], following_head):
                continue
            if key > _sort_key(following_head):
                continue
        if previous is not None or following is not None:
            promoted.add(position)
    deduplicated = set()
    seen = set()
    for position in sorted(promoted):
        _, item, boundary = boundaries[position]
        key = (item["page"], _sort_key(boundary["headword"]))
        if key not in seen:
            deduplicated.add(position)
            seen.add(key)
    return deduplicated


def read_lines(paths):
    lines = []
    corrupt_pages = set()
    for path in paths:
        page = _page_id(path)
        with open(path, encoding="utf-8") as f:
            content = f.read()
        if _CORRUPTION.search(content):
            corrupt_pages.add(page)
        for number, raw in enumerate(content.splitlines(), 1):
            text = raw.strip()
            if text and not _HEADER.match(text):
                lines.append({"page": page, "line": number, "text": text})
    return lines, corrupt_pages


def _references(body):
    match = _REFERENCE_TAIL.search(body)
    if not match:
        return []
    return [
        target.strip()
        for target in re.split(r"[,،/]", match.group(1))
        if target.strip()
    ]


def extract_local(paths):
    lines, corrupt_pages = read_lines(paths)
    boundaries = []
    for index, item in enumerate(lines):
        boundary = classify_boundary(item["text"])
        if (
            boundary is None
            and _valid_headword(item["text"])
            and index + 1 < len(lines)
            and re.match(r"^\(1\)", lines[index + 1]["text"])
        ):
            boundary = {
                "headword": item["text"],
                "body": "",
                "english_terms": [],
                "rule": "numbered_definition",
                "publish": False,
            }
        if boundary:
            boundaries.append((index, item, boundary))
    promoted = _ordered_soft_boundaries(boundaries)

    records = []
    unresolved = []
    covered = set()
    page_ordinals = {}
    for position, (start, item, boundary) in enumerate(boundaries):
        end = boundaries[position + 1][0] if position + 1 < len(boundaries) else len(lines)
        if (
            boundary["rule"] == "direct_reference"
            and re.search(r"[۔.)]\s*$", item["text"])
        ):
            end = start + 1
        span = lines[start:end]
        if not boundary["publish"] and position not in promoted:
            covered.update(range(start, end))
            unresolved.append({
                "page": item["page"],
                "pages": list(dict.fromkeys(part["page"] for part in span)),
                "line": item["line"],
                "previous_text": lines[start - 1]["text"] if start else None,
                "candidate_headword": boundary["headword"],
                "rule": boundary["rule"],
                "raw_text": "\n".join(part["text"] for part in span),
            })
            continue
        if position in promoted:
            boundary = dict(boundary, rule="ordered_heuristic")
        if item["page"] in corrupt_pages:
            covered.update(range(start, end))
            unresolved.append({
                "page": item["page"],
                "pages": list(dict.fromkeys(part["page"] for part in span)),
                "line": item["line"],
                "previous_text": lines[start - 1]["text"] if start else None,
                "candidate_headword": boundary["headword"],
                "rule": "page_corruption",
                "raw_text": "\n".join(part["text"] for part in span),
            })
            continue
        covered.update(range(start, end))

        pages = list(dict.fromkeys(part["page"] for part in span))
        raw_lines = [item["text"]]
        raw_lines.extend(part["text"] for part in span[1:])
        raw_text = "\n".join(raw_lines)
        definition_lines = [boundary["body"]]
        definition_lines.extend(part["text"] for part in span[1:])
        definition = "\n".join(line for line in definition_lines if line).strip()
        references = _references(definition)
        entry_type = (
            "cross_reference"
            if definition.startswith("دیکھیے ") and len(span) == 1
            else "definition"
        )
        page_ordinals[item["page"]] = page_ordinals.get(item["page"], 0) + 1
        flags = ["text_only"]
        if end == len(lines) and not re.search(r"[۔.)]\s*$", raw_text):
            flags.append("continues_after_range")
        if any(classify_boundary(part["text"]) is None for part in span[1:]):
            flags.append("boundary_inferred")
        score = 0.98 if entry_type == "cross_reference" else {
            "layout": 0.92,
            "english_term": 0.85,
            "direct_reference": 0.85,
            "ordered_heuristic": 0.72,
        }[boundary["rule"]]
        records.append({
            "schema": SCHEMA,
            "id": f"farhangiadabiyat-{item['page']}-{page_ordinals[item['page']]:03d}",
            "headword": boundary["headword"],
            "english_terms": boundary["english_terms"],
            "definition": "" if entry_type == "cross_reference" else definition,
            "examples": [],
            "cross_references": references,
            "raw_text": raw_text,
            "pages": pages,
            "source": {
                "start": {"page": item["page"], "line": item["line"]},
                "end": {"page": span[-1]["page"], "line": span[-1]["line"]},
                "rule": boundary["rule"],
            },
            "entry_type": entry_type,
            "quality": {
                "status": "candidate",
                "score": score,
                "flags": flags,
            },
        })
    gap_start = None
    for index in range(len(lines) + 1):
        if index < len(lines) and index not in covered:
            gap_start = index if gap_start is None else gap_start
            continue
        if gap_start is not None:
            span = lines[gap_start:index]
            unresolved.append({
                "page": span[0]["page"],
                "pages": list(dict.fromkeys(part["page"] for part in span)),
                "line": span[0]["line"],
                "previous_text": lines[gap_start - 1]["text"] if gap_start else None,
                "candidate_headword": None,
                "rule": "no_boundary",
                "raw_text": "\n".join(part["text"] for part in span),
            })
            gap_start = None
    unresolved.sort(key=lambda record: (record["page"], record["line"]))
    return records, unresolved


def recover_unresolved(records, unresolved):
    """Recover lower-confidence entries using source and dictionary order."""
    ordered = sorted(
        records,
        key=lambda record: (
            int(record["source"]["start"]["page"]),
            record["source"]["start"]["line"],
        ),
    )
    known = {_sort_key(record["headword"]) for record in records}
    recovered = []
    page_ordinals = {}
    for block in unresolved:
        if block.get("rule") == "page_corruption":
            continue
        if not re.search(r"[۔.)]\s*$", block.get("previous_text") or ""):
            continue
        headword = block.get("candidate_headword")
        if not headword or not _valid_headword(headword):
            first = block["raw_text"].splitlines()[0]
            split = _split_heuristic(first)
            headword = split[0] if split else None
        if (
            not headword
            or not _valid_headword(headword)
            or len(headword) > 35
            or len(headword.split()) > 5
        ):
            continue
        key = _sort_key(headword)
        if key in known:
            continue
        position = (int(block["page"]), block["line"])
        previous = next((
            record for record in reversed(ordered)
            if (int(record["source"]["start"]["page"]),
                record["source"]["start"]["line"]) < position
        ), None)
        following = next((
            record for record in ordered
            if (int(record["source"]["start"]["page"]),
                record["source"]["start"]["line"]) > position
        ), None)
        if not previous or not following:
            continue
        strictly_ordered = (
            _same_initial(headword, previous["headword"])
            and _same_initial(headword, following["headword"])
            and _sort_key(previous["headword"]) < key < _sort_key(following["headword"])
        )
        prefix_supported = (
            _prefix(headword) == _prefix(previous["headword"])
            or _prefix(headword) == _prefix(following["headword"])
        )
        if not (strictly_ordered or prefix_supported):
            continue
        first, *continuation = block["raw_text"].splitlines()
        if not first.startswith(headword):
            continue
        definition = first[len(headword):].strip()
        definition = "\n".join([definition, *continuation]).strip()
        if not definition:
            continue
        page_ordinals[block["page"]] = page_ordinals.get(block["page"], 0) + 1
        recovered.append({
            "schema": SCHEMA,
            "id": (
                f"farhangiadabiyat-{block['page']}-"
                f"r{page_ordinals[block['page']]:03d}"
            ),
            "headword": headword,
            "english_terms": [],
            "definition": definition,
            "examples": [],
            "cross_references": _references(definition),
            "raw_text": block["raw_text"],
            "pages": block.get("pages") or [block["page"]],
            "source": {
                "start": {"page": block["page"], "line": block["line"]},
                "rule": "ordered_recovery",
            },
            "entry_type": "definition",
            "quality": {
                "status": "candidate",
                "score": 0.55,
                "flags": ["text_only", "recovered", "needs_review"],
            },
        })
        known.add(key)
    return recovered


def write_local(paths, output, unresolved_output, retained_output=None,
                recovered_output=None, min_score=0.85):
    records, unresolved = extract_local(paths)
    _atomic_jsonl(output, records)
    _atomic_jsonl(unresolved_output, unresolved)
    retained = [
        record for record in records
        if record["quality"]["score"] >= min_score
        and "continues_after_range" not in record["quality"]["flags"]
    ]
    if retained_output:
        _atomic_jsonl(retained_output, retained)
    recovered = recover_unresolved(records, unresolved)
    if recovered_output:
        _atomic_jsonl(recovered_output, recovered)
    return records, unresolved, retained, recovered
