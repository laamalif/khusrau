#!/usr/bin/env python3
"""Parallel page reader. Skips pages already transcribed so it is restartable.

Sequential reading measured ~2 min/page on gemini-3.8-flash (thinking is on), so a
20-page sample took 40 minutes. The calls are independent and network-bound, so a
small thread pool is the whole fix.

Restartability matters more than speed: siraj lost 168 already-paid-for calls to one
malformed record killing the write. Here every page is written the moment it returns,
and an existing .txt is never re-requested.

Usage: ocr_batch.py <image ...>        [OCR_WORKERS=6]
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from ocr_page import artifact_paths, read_page, write_response

WORKERS = int(os.environ.get("OCR_WORKERS", "6"))
MODEL = os.environ.get("HTR_MODEL")
THINKING = os.environ.get("HTR_THINKING")
SUFFIX = os.environ.get("HTR_SUFFIX", "")
MODE = os.environ.get("HTR_MODE", "verse")


def one(path):
    out, partial, response = artifact_paths(path, SUFFIX)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return path, out, "cached", None
    if os.path.exists(response):
        with open(response, encoding="utf-8") as f:
            recorded = json.load(f)
        status = "cached_partial" if os.path.exists(partial) else "cached_error"
        return path, partial if os.path.exists(partial) else response, status, recorded.get("finish")
    r = read_page(path, MODEL, THINKING, mode=MODE)
    r["request"] = {"model": MODEL, "thinking": THINKING, "mode": MODE}
    write_response(response, r)
    if "error" in r:
        return path, response, "error", r["error"][:200]
    txt = r.get("text", "").strip()
    if not txt:
        return path, response, "empty", r.get("finish")
    if r.get("finish") != "STOP":
        with open(partial, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
        return path, partial, "partial", r.get("finish")
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    return path, out, "STOP", r.get("usage", {}).get("candidatesTokenCount")


def run(paths, workers=WORKERS):
    done = fail = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, p) for p in paths]
        for f in as_completed(futs):
            path, out, status, extra = f.result()
            if status in ("error", "empty", "partial", "cached_partial", "cached_error"):
                fail += 1
                print(f"FAIL {os.path.basename(path)}: {status} {extra}", flush=True)
            else:
                done += 1
                flag = "" if status in ("STOP", "cached") else f"  !! finish={status}"
                print(f"ok   {os.path.basename(path)} -> {os.path.basename(out)} "
                      f"({status}, out={extra}){flag}", flush=True)
    print(f"\n{done} ok, {fail} failed, of {len(paths)}")
    return 1 if fail else 0


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: ocr_batch.py <image ...>", file=sys.stderr)
        return 2
    return run(paths)


if __name__ == "__main__":
    raise SystemExit(main())
