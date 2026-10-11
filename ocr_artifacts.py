#!/usr/bin/env python3
"""OCR artifacts for one page image: explicit state and append-only attempts.

Every paid response is kept, and a usable read is never lost. A page has one
canonical pair, `<stem>.txt` and `<stem>.response.json`, and the two change
together, only when an attempt finishes with STOP. Each attempt is first
written once to `<stem>.attempts/<id>.response.json`. An incomplete attempt
writes its text to `<stem>.partial.txt` and leaves the canonical pair alone.
When a STOP attempt replaces the canonical pair, the old pair is copied into
the attempts directory beside it first. Hand-corrected and reviewed text
therefore survives any rerun.

The canonical names and their meaning are unchanged, so the image/`.txt`
ingestion contract and `dataset_export` work as before.

Local filesystem only: this module never calls a model.
"""
import datetime
import json
import os
import secrets
import shutil
import tempfile

# Usable text: nothing more to request unless forced.
DONE = ("complete", "edited", "unrecorded")
STATES = DONE + ("partial", "failed", "pending")


def _atomic_write(path, content):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2) + "\n"


def _new_attempt():
    now = datetime.datetime.now(datetime.timezone.utc)
    return {
        "id": now.strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(3),
        "created_at": now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
    }


def _nonempty(path):
    return os.path.isfile(path) and os.path.getsize(path) > 0


class PageArtifacts:
    """Paths, state, and recording for one page image's OCR artifacts."""

    def __init__(self, image, suffix=""):
        self.image = image
        base = os.path.splitext(image)[0] + suffix
        self.text = base + ".txt"
        self.partial = base + ".partial.txt"
        self.response = base + ".response.json"
        self.attempts = base + ".attempts"

    def _attempt_ids(self):
        if not os.path.isdir(self.attempts):
            return []
        return sorted(name[:-len(".response.json")] for name in os.listdir(self.attempts)
                      if name.endswith(".response.json") and ".superseded." not in name)

    def _canonical_response(self):
        if not os.path.isfile(self.response):
            return None
        with open(self.response, encoding="utf-8") as f:
            return json.load(f)

    def state(self):
        """One of STATES, from the files on disk. Never modifies anything."""
        if _nonempty(self.text):
            response = self._canonical_response()
            if response is None:
                return "unrecorded"
            with open(self.text, encoding="utf-8") as f:
                text = f.read().strip()
            matches = response.get("finish") == "STOP" and \
                (response.get("text") or "").strip() == text
            return "complete" if matches else "edited"
        if _nonempty(self.partial):
            return "partial"
        if self._attempt_ids() or os.path.isfile(self.response):
            return "failed"
        return "pending"

    def last_attempt(self):
        """Most recent recorded attempt, or the legacy canonical response."""
        ids = self._attempt_ids()
        if ids:
            with open(os.path.join(self.attempts, ids[-1] + ".response.json"),
                      encoding="utf-8") as f:
                return json.load(f)
        return self._canonical_response()

    def record(self, response, request):
        """Record one model response. Returns complete | partial | empty | error."""
        attempt = _new_attempt()
        response = dict(response, request=request, attempt=attempt)
        payload = _json(response)
        os.makedirs(self.attempts, exist_ok=True)
        # Write-once: exclusive create, so an attempt can never be overwritten.
        with open(os.path.join(self.attempts, attempt["id"] + ".response.json"),
                  "x", encoding="utf-8") as f:
            f.write(payload)

        if "error" in response:
            return "error"
        text = (response.get("text") or "").strip()
        if not text:
            return "empty"
        if response.get("finish") != "STOP":
            _atomic_write(self.partial, text + "\n")
            return "partial"

        archive = os.path.join(self.attempts, attempt["id"] + ".superseded")
        if os.path.isfile(self.text):
            shutil.copy2(self.text, archive + ".txt")
        if os.path.isfile(self.response):
            shutil.copy2(self.response, archive + ".response.json")
        _atomic_write(self.response, payload)
        _atomic_write(self.text, text + "\n")
        return "complete"
