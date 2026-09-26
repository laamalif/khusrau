#!/usr/bin/env python3
"""Unified command-line interface for the Khusrau research workflow."""
import argparse
import glob
import json
import os
import sys
import tomllib


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _ratio(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return value


def _config_value(config, *keys):
    value = config
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _paths(values, base):
    paths = []
    for value in values or []:
        value = value if os.path.isabs(value) else os.path.join(base, value)
        matches = sorted(glob.glob(value))
        paths.extend(matches or [value])
    return list(dict.fromkeys(os.path.abspath(path) for path in paths))


def load_project(path):
    if not os.path.exists(path):
        if path != "khusrau.toml":
            raise FileNotFoundError(f"project manifest not found: {path}")
        return {}, os.getcwd()
    with open(path, "rb") as f:
        return tomllib.load(f), os.path.dirname(os.path.abspath(path))


def apply_project(args, config, base):
    command = args.command
    if command in ("status", "ocr") and not args.images:
        args.images = _paths(_config_value(config, "pages", "images"), base)
    if command == "status" and not args.category:
        args.category = _config_value(config, "reference", "ganjoor", "categories")
    if command == "status" and not args.rekhta_dir:
        directory = _config_value(config, "reference", "rekhta", "directory")
        args.rekhta_dir = _paths([directory], base)[0] if directory else None
    if command == "ocr":
        ocr = config.get("ocr", {})
        args.workers = args.workers or ocr.get("workers") or int(os.environ.get("OCR_WORKERS", "6"))
        args.model = args.model or ocr.get("model") or os.environ.get("HTR_MODEL")
        args.thinking = args.thinking or ocr.get("thinking") or os.environ.get("HTR_THINKING")
        args.mode = args.mode or ocr.get("mode") or os.environ.get("HTR_MODE", "verse")
    if command in ("status", "ocr"):
        args.suffix = args.suffix if args.suffix is not None else (
            _config_value(config, "ocr", "suffix") or os.environ.get("HTR_SUFFIX", ""))
    if command == "align":
        alignment = config.get("alignment", {})
        if not args.pages:
            args.pages = _paths(_config_value(config, "pages", "transcriptions"), base)
        if not args.category:
            args.category = _config_value(config, "reference", "ganjoor", "categories")
        output = args.output or alignment.get("output") or "work/gold_pairs.jsonl"
        args.output = output if os.path.isabs(output) else os.path.join(base, output)
        args.min_jaccard = (args.min_jaccard if args.min_jaccard is not None
                            else alignment.get("min_jaccard", 0.55))
        args.min_lines = args.min_lines or alignment.get("min_lines", 4)
        args.min_monotone = (args.min_monotone if args.min_monotone is not None
                             else alignment.get("min_monotone", 0.80))
        args.review_lines = args.review_lines or alignment.get("review_lines", 2)
    if command == "reference":
        ganjoor_categories = _config_value(config, "reference", "ganjoor", "categories")
        rekhta = _config_value(config, "reference", "rekhta") or {}
        if args.reference_command in ("fetch", "verify") and not args.category:
            args.category = ganjoor_categories
        if args.reference_command in ("rekhta-fetch", "rekhta-verify"):
            args.poet = args.poet or rekhta.get("poet")
            if not args.directory:
                directory = rekhta.get("directory")
                args.directory = _paths([directory], base)[0] if directory else None
    return args


def _rekhta_status(directory):
    if not directory or not os.path.isdir(directory):
        return 0, 0
    usable = bad = 0
    for path in glob.glob(os.path.join(directory, "*.txt")):
        with open(path, encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
        if len(lines) >= 2:
            usable += 1
        else:
            bad += 1
    return usable, bad


def cmd_status(args):
    from ocr_page import artifact_paths

    if not args.images and not args.category and not args.rekhta_dir:
        print("khusrau status: provide images, --category, and/or --rekhta-dir",
              file=sys.stderr)
        return 2
    counts = {"complete": 0, "partial": 0, "failed": 0, "pending": 0}
    for path in args.images:
        complete, partial, response = artifact_paths(path, args.suffix)
        if os.path.isfile(complete) and os.path.getsize(complete):
            state = "complete"
        elif os.path.isfile(partial) and os.path.getsize(partial):
            state = "partial"
        elif os.path.isfile(response):
            state = "failed"
        else:
            state = "pending"
        counts[state] += 1
        print(f"{state:8s} {path}")

    reference_bad = 0
    if args.category:
        import ganjoor

        for category in args.category:
            listing = ganjoor._find(f"cat_{category}.json", category)
            if not os.path.isfile(listing):
                print(f"missing  Ganjoor category {category}: catalogue is not cached")
                reference_bad += 1
                continue
            with open(listing, encoding="utf-8") as f:
                poems = json.load(f)
            missing = empty = 0
            for poem in poems:
                path = ganjoor._find(f"poem_{poem['id']}.json", category)
                if not os.path.isfile(path):
                    missing += 1
                    continue
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
                if not any(v.strip() for v in record.get("verses") or []):
                    empty += 1
            state = "ready" if not (missing or empty) else "incomplete"
            print(f"{state:8s} Ganjoor category {category}: {len(poems)} poems, "
                  f"{missing} missing, {empty} empty")
            reference_bad += missing + empty
    if args.rekhta_dir:
        usable, bad = _rekhta_status(args.rekhta_dir)
        state = "ready" if usable and not bad else "incomplete"
        print(f"{state:8s} Rekhta directory {args.rekhta_dir}: "
              f"{usable} usable, {bad} invalid")
        reference_bad += bad + (not usable)

    if args.images:
        print("\nOCR: " + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return 1 if counts["partial"] or counts["failed"] or counts["pending"] or reference_bad else 0


def cmd_ocr(args):
    import ocr_batch

    ocr_batch.MODEL = args.model
    ocr_batch.THINKING = args.thinking
    ocr_batch.SUFFIX = args.suffix
    ocr_batch.MODE = args.mode
    return ocr_batch.run(args.images, workers=args.workers)


def cmd_reference_fetch(args):
    from ganjoor import fetch_all

    failed = 0
    for category in args.category:
        _, missing = fetch_all(category)
        failed += missing
    return 1 if failed else 0


def cmd_reference_verify(args):
    import ganjoor
    from verify_ganjoor import check

    failed = 0
    for category in args.category:
        listing = ganjoor._find(f"cat_{category}.json", category)
        if not os.path.isfile(listing):
            print(f"FAIL category {category}: catalogue is not cached", file=sys.stderr)
            failed += 1
            continue
        result = check(category)
        missing = len(result["missing"])
        empty = len(result["empty"])
        state = "ok" if not (missing or empty) else "FAIL"
        print(f"{state:4s} category {category}: listed={result['listed']} "
              f"files={result['files']} verses={result['verses']} "
              f"missing={missing} empty={empty}")
        failed += missing + empty
    return 1 if failed else 0


def cmd_rekhta_fetch(args):
    from rekhta import fetch_poet

    _, _, failed = fetch_poet(args.poet, args.directory)
    return 1 if failed else 0


def cmd_rekhta_verify(args):
    usable, bad = _rekhta_status(args.directory)
    state = "ok" if usable and not bad else "FAIL"
    print(f"{state:4s} Rekhta {args.poet or ''}: {usable} usable files, {bad} invalid")
    return 0 if usable and not bad else 1


def cmd_align(args):
    from manufacture import run

    return run(
        args.category, args.pages, output=args.output,
        min_jaccard=args.min_jaccard, min_lines=args.min_lines,
        min_monotone=args.min_monotone, review_lines=args.review_lines)


def cmd_export_manifest(args):
    from dataset_export import build_manifest

    records = build_manifest(
        args.images, args.output, args.dataset, status=args.status,
        mode=args.mode, model=args.model, thinking=args.thinking,
        reference=args.reference,
        alignment=args.alignment)
    print(f"wrote {args.output}: {len(records)} page records")
    return 0


def cmd_export_pages(args):
    from dataset_export import export_pages

    records = export_pages(
        args.manifest, args.destination, method=args.method,
        allow_candidate=args.allow_candidate, seed=args.seed,
        validation=args.validation, test=args.test)
    splits = {name: sum(record["split"] == name for record in records)
              for name in ("train", "validation", "test")}
    print(f"exported {len(records)} page pairs to {args.destination}: "
          + ", ".join(f"{name}={count}" for name, count in splits.items()))
    return 0


def cmd_score_prose(args):
    from align_prose import score, write_score

    result = score(args.pages, args.reference)
    if args.output:
        write_score(result, args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_dictionary_extract(args):
    from dictionary_extract import extract, publish_response

    if args.reuse_response:
        records = publish_response(
            args.response, args.output, args.images, args.dataset)
    else:
        records = extract(
            args.images, args.output, args.response, args.dataset,
            args.model, args.thinking)
    print(f"wrote {args.output}: {len(records)} candidate entries")
    return 0


def cmd_dictionary_local(args):
    from dictionary_local import write_local

    records, unresolved, retained, recovered = write_local(
        args.pages, args.output, args.unresolved_output,
        args.retained_output, args.recovered_output, args.min_score)
    print(
        f"wrote {args.output}: {len(records)} candidate entries; "
        f"{len(unresolved)} unresolved blocks; {len(retained)} retained; "
        f"{len(recovered)} recovered")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="khusrau",
        description="Align rough readings of historical Urdu and Persian pages "
                    "with modern digital reference texts.")
    parser.add_argument("--config", default="khusrau.toml",
                        help="project manifest (default: khusrau.toml when present)")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser(
        "status", help="inspect local OCR artifacts and reference caches without network calls")
    status.add_argument("images", nargs="*", help="page images to inspect")
    status.add_argument("-c", "--category", action="append", type=int,
                        help="cached Ganjoor category to inspect; repeat as needed")
    status.add_argument("--rekhta-dir", help="local Rekhta reference directory")
    status.add_argument("--suffix", help="OCR artifact suffix")
    status.set_defaults(func=cmd_status)

    ocr = sub.add_parser("ocr", help="transcribe images; recorded responses are never retried")
    ocr.add_argument("images", nargs="*")
    ocr.add_argument("--workers", type=_positive_int)
    ocr.add_argument("--model")
    ocr.add_argument("--thinking")
    ocr.add_argument("--mode", choices=("verse", "prose", "dictionary"))
    ocr.add_argument("--suffix")
    ocr.set_defaults(func=cmd_ocr)

    reference = sub.add_parser("reference", help="manage Ganjoor reference data")
    reference_sub = reference.add_subparsers(dest="reference_command", required=True)
    fetch = reference_sub.add_parser("fetch", help="download and cache categories")
    fetch.add_argument("category", nargs="*", type=int)
    fetch.set_defaults(func=cmd_reference_fetch)
    verify = reference_sub.add_parser("verify", help="check cached categories for missing text")
    verify.add_argument("category", nargs="*", type=int)
    verify.set_defaults(func=cmd_reference_verify)
    rekhta_fetch = reference_sub.add_parser(
        "rekhta-fetch", help="download a poet's Rekhta ghazals")
    rekhta_fetch.add_argument("poet", nargs="?")
    rekhta_fetch.add_argument("directory", nargs="?")
    rekhta_fetch.set_defaults(func=cmd_rekhta_fetch)
    rekhta_verify = reference_sub.add_parser(
        "rekhta-verify", help="check a local Rekhta text directory")
    rekhta_verify.add_argument("directory", nargs="?")
    rekhta_verify.add_argument("--poet")
    rekhta_verify.set_defaults(func=cmd_rekhta_verify)

    align = sub.add_parser("align", help="match page readings against Ganjoor references")
    align.add_argument("pages", nargs="*", help="rough transcription files")
    align.add_argument("-c", "--category", action="append", type=int,
                       help="Ganjoor category; repeat to combine works")
    align.add_argument("-o", "--output")
    align.add_argument("--min-jaccard", type=_ratio)
    align.add_argument("--min-lines", type=_positive_int)
    align.add_argument("--min-monotone", type=_ratio)
    align.add_argument("--review-lines", type=_positive_int)
    align.set_defaults(func=cmd_align)

    export = sub.add_parser("export", help="build page manifests and materialize page pairs")
    export_sub = export.add_subparsers(dest="export_command", required=True)
    manifest = export_sub.add_parser(
        "manifest", help="create an authoritative JSONL manifest from image/text sidecars")
    manifest.add_argument("images", nargs="+")
    manifest.add_argument("-o", "--output", required=True)
    manifest.add_argument("--dataset", required=True)
    manifest.add_argument("--status", choices=("candidate", "aligned", "reviewed", "gold"),
                          default="candidate")
    manifest.add_argument("--mode", choices=("verse", "prose", "dictionary"))
    manifest.add_argument("--model")
    manifest.add_argument("--thinking")
    manifest.add_argument("--reference")
    manifest.add_argument("--alignment", help="verse JSONL or prose score JSON")
    manifest.set_defaults(func=cmd_export_manifest)
    pages = export_sub.add_parser(
        "pages", help="materialize a validated image/text page-pair dataset")
    pages.add_argument("manifest")
    pages.add_argument("destination")
    pages.add_argument("--method", choices=("hardlink", "copy"), default="hardlink")
    pages.add_argument("--allow-candidate", action="store_true",
                       help="explicitly export unreviewed candidate/aligned text")
    pages.add_argument("--seed", type=int, default=42)
    pages.add_argument("--validation", type=_ratio, default=0.1)
    pages.add_argument("--test", type=_ratio, default=0.1)
    pages.set_defaults(func=cmd_export_pages)

    score = sub.add_parser("score", help="measure OCR against aligned reference text")
    score_sub = score.add_subparsers(dest="score_command", required=True)
    prose = score_sub.add_parser("prose", help="report CER and token WER for prose pages")
    prose.add_argument("reference", help="continuous reference text")
    prose.add_argument("pages", nargs="+", help="ordered page transcription files")
    prose.add_argument("-o", "--output", help="atomic JSON result path")
    prose.set_defaults(func=cmd_score_prose)

    dictionary = sub.add_parser(
        "dictionary", help="extract structured candidate entries from dictionary pages")
    dictionary_sub = dictionary.add_subparsers(
        dest="dictionary_command", required=True)
    extract = dictionary_sub.add_parser(
        "extract", help="extract image-aware entry JSONL from consecutive pages")
    extract.add_argument("images", nargs="+")
    extract.add_argument("-o", "--output", required=True)
    extract.add_argument("--response", required=True)
    extract.add_argument("--dataset", default="farhangiadabiyat")
    extract.add_argument("--model", default="gemini-3.8-flash")
    extract.add_argument("--thinking", default="low")
    extract.add_argument("--reuse-response", action="store_true",
                         help="publish an existing response without an API call")
    extract.set_defaults(func=cmd_dictionary_extract)
    local = dictionary_sub.add_parser(
        "local", help="conservatively extract entry JSONL from OCR text without network calls")
    local.add_argument("pages", nargs="+", help="ordered OCR text files")
    local.add_argument("-o", "--output", required=True)
    local.add_argument("--unresolved-output", required=True)
    local.add_argument("--retained-output")
    local.add_argument("--recovered-output")
    local.add_argument("--min-score", type=_ratio, default=0.85)
    local.set_defaults(func=cmd_dictionary_local)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config, base = load_project(args.config)
        args = apply_project(args, config, base)
        if args.command == "align" and not 0 <= args.min_jaccard <= 1:
            parser.error("alignment.min_jaccard must be between 0 and 1")
        if args.command == "align" and not 0 <= args.min_monotone <= 1:
            parser.error("alignment.min_monotone must be between 0 and 1")
        if args.command == "align" and args.min_lines < 1:
            parser.error("alignment.min_lines must be at least 1")
        if args.command == "align" and args.review_lines < 1:
            parser.error("alignment.review_lines must be at least 1")
        if args.command == "ocr" and not args.images:
            parser.error("ocr requires images or pages.images in the project manifest")
        if args.command == "align" and (not args.pages or not args.category):
            parser.error("align requires pages and Ganjoor categories")
        if args.command == "reference":
            if args.reference_command in ("fetch", "verify") and not args.category:
                parser.error("Ganjoor categories are required")
            if args.reference_command in ("rekhta-fetch", "rekhta-verify") \
                    and not args.directory:
                parser.error("a Rekhta directory is required")
            if args.reference_command == "rekhta-fetch" and not args.poet:
                parser.error("a Rekhta poet slug is required")
        return args.func(args)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
