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

An unreadable response or attempt record makes the page `corrupt`. That state
is never treated as finished and never triggers a request. The damaged file is
left in place for diagnosis, and `--force` archives it before replacing it.

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
STATES = DONE + ("partial", "failed", "corrupt", "pending")


class Corrupt(Exception):
    """An artifact exists but cannot be read as a JSON object."""


def _load_record(path):
    """The JSON object at `path`, None if absent, or raise Corrupt."""
    try:
        with open(path, encoding="utf-8") as f:
            value = json.load(f)
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as error:          # JSONDecodeError, UnicodeDecodeError
        raise Corrupt(path) from error
    if not isinstance(value, dict):
        raise Corrupt(path)
    return value


def _durable_temporary(directory, content):
    """Write `content` to a fsynced temporary file in `directory`; return its path."""
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".khusrau-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
    except BaseException:
        os.unlink(temporary)
        raise
    return temporary


def _atomic_write(path, content):
    """Publish `content` at `path`, replacing any existing file."""
    temporary = _durable_temporary(os.path.dirname(os.path.abspath(path)), content)
    try:
        os.replace(temporary, path)
    except BaseException:
        os.unlink(temporary)
        raise


def _atomic_create(path, content):
    """Publish `content` at `path` only if nothing is there (write-once)."""
    temporary = _durable_temporary(os.path.dirname(os.path.abspath(path)), content)
    try:
        os.link(temporary, path)                      # FileExistsError if taken
    finally:
        os.unlink(temporary)


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

    def _attempt_path(self, attempt_id):
        return os.path.join(self.attempts, attempt_id + ".response.json")

    def problem(self):
        """Why the page is corrupt -- '<file>: <reason>' -- or None if it is not."""
        paths = [self.response] + [self._attempt_path(i) for i in self._attempt_ids()]
        for path in paths:
            try:
                _load_record(path)
            except Corrupt as error:
                return f"{os.path.basename(path)}: {error.__cause__ or 'not a JSON object'}"
        return None

    def state(self):
        """One of STATES, from the files on disk. Never modifies anything."""
        if self.problem():
            return "corrupt"
        response = _load_record(self.response)
        if _nonempty(self.text):
            if response is None:
                return "unrecorded"
            with open(self.text, encoding="utf-8") as f:
                text = f.read().strip()
            matches = response.get("finish") == "STOP" and \
                (response.get("text") or "").strip() == text
            return "complete" if matches else "edited"
        if _nonempty(self.partial):
            return "partial"
        if response is not None or self._attempt_ids():
            return "failed"
        return "pending"

    def last_attempt(self):
        """Most recent recorded attempt (or legacy canonical response); None if
        there is none or it is unreadable."""
        ids = self._attempt_ids()
        try:
            return _load_record(self._attempt_path(ids[-1]) if ids else self.response)
        except Corrupt:
            return None

    def record(self, response, request):
        """Record one model response. Returns complete | partial | empty | error."""
        attempt = _new_attempt()
        response = dict(response, request=request, attempt=attempt)
        payload = _json(response)
        # Complete before visible, and never overwritten.
        _atomic_create(self._attempt_path(attempt["id"]), payload)

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
