#!/usr/bin/env python3
"""OCR artifact provenance: page state, attempt recording, and canonical safety."""
import json
import os
import tempfile
import unittest

from ocr_artifacts import PageArtifacts


def _write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _json(path, value):
    _write(path, json.dumps(value))


def _read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


STOP = {"text": "full read", "finish": "STOP", "usage": {}}
TRUNCATED = {"text": "half re", "finish": "MAX_TOKENS", "usage": {}}


class PageStateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.page = PageArtifacts(os.path.join(self.dir, "168.jpg"))

    def tearDown(self):
        self._tmp.cleanup()

    def test_nothing_recorded_is_pending(self):
        self.assertEqual(self.page.state(), "pending")

    def test_text_matching_stop_response_is_complete(self):
        _write(self.page.text, "full read\n")
        _json(self.page.response, STOP)
        self.assertEqual(self.page.state(), "complete")

    def test_text_differing_from_response_is_edited(self):
        _write(self.page.text, "hand corrected\n")
        _json(self.page.response, STOP)
        self.assertEqual(self.page.state(), "edited")

    def test_text_beside_incomplete_response_is_edited_not_complete(self):
        # Legacy layout: an incomplete rerun overwrote the shared response.
        _write(self.page.text, "full read\n")
        _json(self.page.response, TRUNCATED)
        self.assertEqual(self.page.state(), "edited")

    def test_text_without_response_is_unrecorded(self):
        _write(self.page.text, "supplied reading\n")
        self.assertEqual(self.page.state(), "unrecorded")

    def test_partial_text_without_canonical_is_partial(self):
        _write(self.page.partial, "half re\n")
        _json(self.page.response, TRUNCATED)
        self.assertEqual(self.page.state(), "partial")

    def test_recorded_attempt_without_text_is_failed(self):
        self.page.record({"error": "HTTP 500"}, {"model": "m"})
        self.assertEqual(self.page.state(), "failed")

    def test_empty_canonical_text_does_not_count(self):
        _write(self.page.text, "")
        self.assertEqual(self.page.state(), "pending")

    def test_suffix_selects_a_separate_artifact_family(self):
        page = PageArtifacts(os.path.join(self.dir, "168.jpg"), suffix=".pro")
        self.assertEqual(page.text, os.path.join(self.dir, "168.pro.txt"))
        self.assertEqual(page.attempts, os.path.join(self.dir, "168.pro.attempts"))


class RecordTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.page = PageArtifacts(os.path.join(self.dir, "168.jpg"))

    def tearDown(self):
        self._tmp.cleanup()

    def _attempt_files(self):
        return sorted(os.listdir(self.page.attempts))

    def test_stop_attempt_becomes_canonical_with_provenance(self):
        outcome = self.page.record(dict(STOP), {"model": "m", "mode": "verse"})
        self.assertEqual(outcome, "complete")
        self.assertEqual(_read(self.page.text), "full read\n")
        canonical = json.loads(_read(self.page.response))
        self.assertEqual(canonical["request"], {"model": "m", "mode": "verse"})
        attempt = canonical["attempt"]
        self.assertRegex(attempt["id"], r"^\d{8}T\d{12}Z-[0-9a-f]{6}$")
        self.assertTrue(attempt["created_at"].endswith("Z"))
        self.assertEqual(self._attempt_files(), [attempt["id"] + ".response.json"])
        self.assertEqual(self.page.state(), "complete")

    def test_incomplete_rerun_preserves_canonical_pair(self):
        self.page.record(dict(STOP), {})
        before_text = _read(self.page.text)
        before_response = _read(self.page.response)
        outcome = self.page.record(dict(TRUNCATED), {})
        self.assertEqual(outcome, "partial")
        self.assertEqual(_read(self.page.text), before_text)
        self.assertEqual(_read(self.page.response), before_response)
        self.assertEqual(_read(self.page.partial), "half re\n")
        self.assertEqual(len(self._attempt_files()), 2)
        self.assertEqual(self.page.state(), "complete")

    def test_stop_rerun_archives_displaced_edited_text(self):
        self.page.record(dict(STOP), {})
        _write(self.page.text, "hand corrected\n")
        old_response = _read(self.page.response)
        self.page.record({"text": "second read", "finish": "STOP"}, {})
        self.assertEqual(_read(self.page.text), "second read\n")
        attempt_id = json.loads(_read(self.page.response))["attempt"]["id"]
        archived = os.path.join(self.page.attempts, attempt_id + ".superseded.txt")
        self.assertEqual(_read(archived), "hand corrected\n")
        archived_response = os.path.join(
            self.page.attempts, attempt_id + ".superseded.response.json")
        self.assertEqual(_read(archived_response), old_response)

    def test_error_and_empty_attempts_are_recorded_but_never_promoted(self):
        _write(self.page.text, "supplied reading\n")
        self.assertEqual(self.page.record({"error": "HTTP 500"}, {}), "error")
        self.assertEqual(self.page.record({"text": "  ", "finish": "STOP"}, {}), "empty")
        self.assertEqual(_read(self.page.text), "supplied reading\n")
        self.assertFalse(os.path.exists(self.page.response))
        self.assertEqual(len(self._attempt_files()), 2)

    def test_attempt_records_are_never_overwritten(self):
        for _ in range(5):
            self.page.record({"error": "HTTP 500"}, {})
        self.assertEqual(len(set(self._attempt_files())), 5)

    def test_record_leaves_no_temporary_files(self):
        self.page.record(dict(STOP), {})
        self.page.record(dict(TRUNCATED), {})
        leftovers = [name for root, _, names in os.walk(self.dir)
                     for name in names if name.startswith(".khusrau-")]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
