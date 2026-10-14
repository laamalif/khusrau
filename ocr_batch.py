#!/usr/bin/env python3
"""Parallel page reader. Restartable, and it never pays twice for a page.

Sequential reading measured ~2 min/page on gemini-3.8-flash (thinking is on), so a
20-page sample took 40 minutes. The calls are independent and network-bound, so a
small thread pool is the whole fix.

Restartability matters more than speed: siraj lost 168 already-paid-for calls to one
malformed record killing the write. Here every attempt is recorded the moment it
returns (see ocr_artifacts.py). A page with usable text or any recorded attempt is
never requested again unless `force` is given. Even then, only a STOP read replaces
the canonical text, and the text it replaces is archived first.

Usage: ocr_batch.py [--force] <image ...>        [OCR_WORKERS=6]
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

from ocr_artifacts import DONE, PageArtifacts
from ocr_page import read_page
from vertex import DEFAULT_MODEL

WORKERS = int(os.environ.get("OCR_WORKERS", "6"))
MODEL = os.environ.get("HTR_MODEL")
THINKING = os.environ.get("HTR_THINKING")
SUFFIX = os.environ.get("HTR_SUFFIX", "")
MODE = os.environ.get("HTR_MODE", "verse")


class Result(NamedTuple):
    path: str
    status: str        # an ocr_artifacts state, or a record() outcome
    cached: bool       # True when no request was made
    detail: object     # finish reason, error text, or output token count

    @property
    def ok(self):
        return self.status in DONE


def one(path, force=False):
    page = PageArtifacts(path, SUFFIX)
    state = page.state()
    if not force and state != "pending":
        if state == "corrupt":
            return Result(path, state, True, page.problem())
        last = page.last_attempt() or {}
        return Result(path, state, True, last.get("error") or last.get("finish"))
    active_model = MODEL or DEFAULT_MODEL
    response = read_page(path, active_model, THINKING, mode=MODE)
    actual_model = response.get("model") or active_model
    request = {"model": actual_model, "thinking": THINKING, "mode": MODE}
    outcome = page.record(response, request)
    if outcome == "complete":
        return Result(path, "complete", False,
                      response.get("usage", {}).get("candidatesTokenCount"))
    return Result(path, outcome, False,
                  (response.get("error") or "")[:200] or response.get("finish"))


def _settle(path, future):
    """A page's Result; an unexpected exception becomes that page's failure."""
    try:
        return future.result()
    except Exception as error:                              # noqa: BLE001
        # A paid response is never lost here: record() publishes the attempt
        # before anything else can fail.
        return Result(path, "exception", False, f"{type(error).__name__}: {error}")


def run(paths, workers=WORKERS, force=False):
    paths = list(dict.fromkeys(os.path.abspath(path) for path in paths))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [(path, ex.submit(one, path, force)) for path in paths]
        results = [_settle(path, future) for path, future in futures]
    for result in results:
        label = "ok  " if result.ok else "FAIL"
        source = "cached" if result.cached else "new"
        print(f"{label} {os.path.basename(result.path)}: {result.status} "
              f"({source}, {result.detail})", flush=True)
    failed = sum(not result.ok for result in results)
    print(f"\n{len(results) - failed} ok, {failed} failed, of {len(results)}")
    return 1 if failed else 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="Transcribe page images with Gemini.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--force", action="store_true",
                        help="request again even if text or an attempt is recorded")
    args = parser.parse_args(argv)
    return run(args.images, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
