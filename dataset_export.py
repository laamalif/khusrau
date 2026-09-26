#!/usr/bin/env python3
"""Build authoritative page manifests and materialize validated page-pair datasets."""
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

SCHEMA = "khusrau-page-manifest-v1"
STATUSES = ("candidate", "aligned", "reviewed", "gold")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identifier(value):
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    if not value:
        raise ValueError("dataset and page identifiers must contain letters or digits")
    return value


def _read_alignment(path):
    if not path:
        return {}, None
    with open(path, encoding="utf-8") as f:
        content = f.read()
    try:
        value = json.loads(content)
        rows = value if isinstance(value, list) else [value]
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
    if not rows:
        return {}, os.path.abspath(path)
    if len(rows) == 1 and "pages" in rows[0]:
        summary = {key: value for key, value in rows[0].items()
                   if key not in ("pages", "reference")}
        return {"*": summary}, os.path.abspath(path)
    by_page = {}
    for row in rows:
        page = row.get("page")
        if page:
            by_page.setdefault(page, []).append(row)
    return by_page, os.path.abspath(path)


def _alignment_summary(rows):
    if not rows:
        return None
    if isinstance(rows, dict):
        return rows
    tiers = [row.get("tier") for row in rows]
    poems = [{"id": row.get("poem_id"), "title": row.get("poem_title"),
              "tier": row.get("tier")} for row in rows]
    summary = {
        "tiers": {tier: tiers.count(tier) for tier in sorted(set(tiers)) if tier},
        "matched_lines": sum(row.get("matched_lines", 0) for row in rows),
        "poems": poems,
    }
    covered = [
        row for row in rows
        if row.get("tier") == "gold" and row.get("aligned_lines") is not None
    ]
    if covered:
        aligned = sum(row["aligned_lines"] for row in covered)
        hypothesis_span = sum(row["hypothesis_span_lines"] for row in covered)
        reference_span = sum(row["reference_span_lines"] for row in covered)
        precision = aligned / max(1, hypothesis_span)
        recall = aligned / max(1, reference_span)
        summary["coverage"] = {
            "aligned_lines": aligned,
            "hypothesis_span_lines": hypothesis_span,
            "hypothesis_span_precision": round(precision, 4),
            "reference_span_lines": reference_span,
            "reference_span_recall": round(recall, 4),
            "span_f1": round(
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0,
                4,
            ),
            "scope": "between first and last matched boundaries",
        }
    return summary


def build_manifest(images, output, dataset, status="candidate", mode=None,
                   model=None, thinking=None, reference=None, alignment=None):
    if status not in STATUSES:
        raise ValueError(f"invalid status: {status}")
    output = os.path.abspath(output)
    base = os.path.dirname(output)
    dataset_id = _identifier(dataset)
    alignment_by_page, alignment_path = _read_alignment(alignment)
    records = []
    seen = set()
    for image in sorted(os.path.abspath(path) for path in images):
        if not os.path.isfile(image):
            raise FileNotFoundError(image)
        stem, _ = os.path.splitext(image)
        text = stem + ".txt"
        response = stem + ".response.json"
        if not os.path.isfile(text) or not os.path.getsize(text):
            raise ValueError(f"missing non-empty text sidecar: {text}")
        page_id = _identifier(os.path.basename(stem))
        record_id = f"{dataset_id}-{page_id}"
        if record_id in seen:
            raise ValueError(f"duplicate record id: {record_id}")
        seen.add(record_id)

        response_data = {}
        if os.path.isfile(response):
            with open(response, encoding="utf-8") as f:
                response_data = json.load(f)
        request = response_data.get("request") or {}
        finish = response_data.get("finish")
        if response_data and finish != "STOP":
            raise ValueError(f"OCR response is not complete for {image}: {finish}")

        page_alignment = (alignment_by_page.get(os.path.basename(text))
                          or alignment_by_page.get("*"))
        record = {
            "schema": SCHEMA,
            "id": record_id,
            "dataset": dataset_id,
            "page": page_id,
            "image": os.path.relpath(image, base),
            "text": os.path.relpath(text, base),
            "image_sha256": sha256(image),
            "text_sha256": sha256(text),
            "status": status,
            "mode": mode or request.get("mode"),
            "ocr": {
                "model": model or request.get("model"),
                "thinking": thinking or request.get("thinking"),
                "finish": finish,
                "usage": response_data.get("usage") or {},
            },
        }
        if reference:
            record["reference"] = reference
        summary = _alignment_summary(page_alignment)
        if summary:
            record["alignment"] = summary
        if alignment_path:
            record["alignment_source"] = os.path.relpath(alignment_path, base)
        records.append(record)

    os.makedirs(base, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", suffix=".jsonl", dir=base)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temporary, output)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise
    return records


def read_manifest(path):
    base = os.path.dirname(os.path.abspath(path))
    records = []
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("schema") != SCHEMA:
                raise ValueError(f"{path}:{number}: unsupported schema")
            for key in ("id", "image", "text", "image_sha256", "text_sha256", "status"):
                if not record.get(key):
                    raise ValueError(f"{path}:{number}: missing {key}")
            record["_image_path"] = os.path.abspath(os.path.join(base, record["image"]))
            record["_text_path"] = os.path.abspath(os.path.join(base, record["text"]))
            records.append(record)
    if not records:
        raise ValueError("manifest contains no records")
    return records


def _split(record_id, seed, validation, test):
    value = int(hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()[:8], 16)
    bucket = value / 0xFFFFFFFF
    if bucket < test:
        return "test"
    if bucket < test + validation:
        return "validation"
    return "train"


def export_pages(manifest, destination, method="hardlink", allow_candidate=False,
                 seed=42, validation=0.1, test=0.1):
    if validation < 0 or test < 0 or validation + test >= 1:
        raise ValueError("validation and test shares must be non-negative and total less than 1")
    destination = os.path.abspath(destination)
    if os.path.exists(destination):
        raise FileExistsError(f"destination already exists: {destination}")
    records = read_manifest(manifest)
    allowed = {"reviewed", "gold"}
    if allow_candidate:
        allowed.update(("candidate", "aligned"))
    rejected = [record["id"] for record in records if record["status"] not in allowed]
    if rejected:
        raise ValueError(
            f"{len(rejected)} records are not reviewed/gold; "
            "use --allow-candidate for an explicitly non-ground-truth export")

    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = tempfile.mkdtemp(prefix=".khusrau-export-", dir=parent)
    exported = []
    split_files = {name: [] for name in ("train", "validation", "test")}
    try:
        for record in records:
            image = record["_image_path"]
            text = record["_text_path"]
            if sha256(image) != record["image_sha256"] or sha256(text) != record["text_sha256"]:
                raise ValueError(f"content hash mismatch: {record['id']}")
            image_ext = Path(image).suffix.lower()
            target_image = os.path.join(temporary, record["id"] + image_ext)
            target_text = os.path.join(temporary, record["id"] + ".txt")
            if method == "copy":
                shutil.copy2(image, target_image)
                shutil.copy2(text, target_text)
            elif method == "hardlink":
                os.link(image, target_image)
                os.link(text, target_text)
            else:
                raise ValueError(f"unsupported export method: {method}")
            split = _split(record["id"], seed, validation, test)
            split_files[split].append(os.path.basename(target_image))
            clean = {key: value for key, value in record.items() if not key.startswith("_")}
            clean["export_image"] = os.path.basename(target_image)
            clean["export_text"] = os.path.basename(target_text)
            clean["split"] = split
            exported.append(clean)

        with open(os.path.join(temporary, "manifest.jsonl"), "w", encoding="utf-8") as f:
            for record in exported:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        for split, names in split_files.items():
            with open(os.path.join(temporary, split + ".lst"), "w", encoding="utf-8") as f:
                f.write("\n".join(sorted(names)) + ("\n" if names else ""))
        os.replace(temporary, destination)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return exported
