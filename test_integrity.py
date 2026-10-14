#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import ganjoor
import corpus
import khusrau
import manufacture
import ocr_batch
import ocr_page
import dataset_export
from discover_sequence import discover
from dictionary_extract import validate_entries
from dictionary_local import (
    classify_boundary,
    extract_local,
    recover_unresolved,
    write_local,
)
from align_prose import score as score_prose
from dataset_export import build_manifest, export_pages
from rekhta_prose import story_from_html


class IntegrityTests(unittest.TestCase):
    def test_sequence_discovery_requires_two_exact_offset_anchors(self):
        class FakeIndex:
            poems = {1: {"title": "Test", "verses": [
                "", "", "", "", "", "", "", "", "", "",
                "reference one", "reference two", "reference three",
            ]}}
            hot = set()

            @staticmethod
            def trigrams(text):
                return {text[index:index + 3] for index in range(len(text) - 2)}

            def match_line(self, text, min_jaccard=0.30):
                return {
                    "anchor one": (1, 10, 0.9, "reference one"),
                    "recover me": (1, 11, 0.4, "reference two"),
                    "anchor two": (1, 12, 0.9, "reference three"),
                }[text]

        record = {
            "page": "page.txt",
            "poem_id": 1,
            "poem_title": "Test",
            "tier": "gold",
            "pairs": [
                {"line": 0, "verse": 10, "jaccard": 0.9,
                 "hyp": "anchor one", "ref": "reference one"},
                {"line": 2, "verse": 12, "jaccard": 0.9,
                 "hyp": "anchor two", "ref": "reference three"},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.txt")
            with open(page, "w", encoding="utf-8") as f:
                f.write("  ## Indented Section Heading\nanchor one\nrecover me\nanchor two\n")
            with patch("discover_sequence.process", return_value=([record], 3)), \
                    patch("discover_sequence._candidate_score",
                          return_value=(0.4, "reference two")):
                rows = discover(FakeIndex(), page)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate"]["verse"], 11)
        self.assertEqual(rows[0]["candidate"]["hyp"], "recover me")
        self.assertEqual(rows[0]["evidence"]["offset"], 10)
        self.assertTrue(rows[0]["candidate"]["is_global_top"])
        self.assertEqual(
            rows[0]["evidence"]["exclusion"], "below_acceptance_threshold")
        self.assertIsNone(rows[0]["adjudication"])

    def test_alignment_span_metrics_penalize_internal_omissions(self):
        hits = [
            {"line": 2, "verse": 10},
            {"line": 3, "verse": 11},
            {"line": 5, "verse": 13},
        ]
        metrics = manufacture.span_metrics(hits)
        self.assertEqual(metrics["aligned_lines"], 3)
        self.assertEqual(metrics["hypothesis_span_lines"], 4)
        self.assertEqual(metrics["reference_span_lines"], 4)
        self.assertEqual(metrics["hypothesis_span_precision"], 0.75)
        self.assertEqual(metrics["reference_span_recall"], 0.75)
        self.assertEqual(metrics["span_f1"], 0.75)

    def test_alignment_span_metrics_do_not_reward_duplicate_matches(self):
        hits = [
            {"line": 0, "verse": 4},
            {"line": 1, "verse": 4},
            {"line": 2, "verse": 5},
        ]
        metrics = manufacture.span_metrics(hits)
        self.assertEqual(metrics["aligned_lines"], 2)
        self.assertEqual(metrics["hypothesis_span_precision"], 0.6667)
        self.assertEqual(metrics["reference_span_recall"], 1.0)

    def test_manifest_preserves_alignment_span_coverage(self):
        rows = [{
            "tier": "gold",
            "poem_id": 1,
            "poem_title": "Test",
            "matched_lines": 3,
            "aligned_lines": 3,
            "hypothesis_span_lines": 4,
            "reference_span_lines": 5,
        }]
        summary = dataset_export._alignment_summary(rows)
        self.assertEqual(summary["coverage"]["hypothesis_span_precision"], 0.75)
        self.assertEqual(summary["coverage"]["reference_span_recall"], 0.6)
        self.assertEqual(summary["coverage"]["span_f1"], 0.6667)

    def test_local_dictionary_rules_publish_only_explicit_boundaries(self):
        english = classify_boundary("ابجد (alphabet) یونانی حروف کا مجموعہ۔")
        heuristic = classify_boundary("ابجد یونانی حروف کا مجموعہ۔")
        self.assertEqual(english["headword"], "ابجد")
        self.assertEqual(english["english_terms"], ["alphabet"])
        self.assertTrue(english["publish"])
        self.assertFalse(heuristic["publish"])

    def test_local_dictionary_extracts_reference_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "017.txt")
            with open(page, "w", encoding="utf-8") as f:
                f.write("فرہنگ ادبیات\n")
                f.write("آپ آرٹ (op art) دیکھیے آپٹک آرٹ۔\n")
                f.write("آپٹک آرٹ (optic art) بصری فن۔\n")
            records, _ = extract_local([page])
        self.assertEqual(records[0]["headword"], "آپ آرٹ")
        self.assertEqual(records[0]["cross_references"], ["آپٹک آرٹ"])
        self.assertEqual(records[0]["source"]["start"]["line"], 2)

    def test_local_dictionary_promotes_alphabetically_bounded_headword(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "275.txt")
            with open(page, "w", encoding="utf-8") as f:
                f.write("ترکیب (composition) تعریف۔\n")
                f.write("ترکیب بند نظم کے بند کی ایک ہیئت۔\n")
                f.write("ترکیبی زبانیں (organic languages) تعریف۔\n")
            records, unresolved = extract_local([page])
        self.assertEqual(
            [record["headword"] for record in records],
            ["ترکیب", "ترکیب بند", "ترکیبی زبانیں"])
        self.assertEqual(records[1]["source"]["rule"], "ordered_heuristic")
        self.assertFalse(unresolved)

    def test_local_cross_reference_cannot_absorb_following_text(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "017.txt")
            with open(page, "w", encoding="utf-8") as f:
                f.write("آزاد روپ دیکھیے آزاد صرفیہ۔\n")
                f.write("ناقابل شناخت اگلا اندراج\n")
            records, unresolved = extract_local([page])
        self.assertEqual(records[0]["raw_text"], "آزاد روپ دیکھیے آزاد صرفیہ۔")
        self.assertEqual(unresolved[0]["rule"], "no_boundary")
        self.assertIn("ناقابل شناخت", unresolved[0]["raw_text"])

    def test_local_dictionary_routes_corrupt_page_to_unresolved(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "817.txt")
            with open(page, "w", encoding="utf-8") as f:
                f.write("نیاناول دیکھیے جدید ناول۔\n")
                f.write('"name": "نیچری", "description": "خراب"\n')
            records, unresolved = extract_local([page])
        self.assertFalse(records)
        self.assertEqual(unresolved[0]["rule"], "page_corruption")

    def test_local_dictionary_writes_reproducible_retained_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "017.txt")
            output = os.path.join(directory, "all.jsonl")
            unresolved = os.path.join(directory, "unresolved.jsonl")
            retained = os.path.join(directory, "retained.jsonl")
            with open(page, "w", encoding="utf-8") as f:
                f.write("آپ آرٹ (op art) دیکھیے آپٹک آرٹ۔\n")
            records, _, kept, recovered = write_local(
                [page], output, unresolved, retained, min_score=0.85)
            with open(retained, encoding="utf-8") as f:
                written = [json.loads(line) for line in f]
        self.assertEqual(kept, records)
        self.assertFalse(recovered)
        self.assertEqual(written[0]["headword"], "آپ آرٹ")

    def test_local_recovery_requires_neighboring_dictionary_order(self):
        records = [
            {
                "headword": "اباحی",
                "source": {"start": {"page": "029", "line": 1}},
            },
            {
                "headword": "ابتدا و ضرب",
                "source": {"start": {"page": "029", "line": 10}},
            },
        ]
        unresolved = [{
            "page": "029",
            "pages": ["029"],
            "line": 5,
            "previous_text": "پچھلا اندراج مکمل۔",
            "candidate_headword": "اَبَتَث",
            "raw_text": "اَبَتَث مفرد حروف کا مجموعہ۔",
        }]
        recovered = recover_unresolved(records, unresolved)
        self.assertEqual(recovered[0]["headword"], "اَبَتَث")
        self.assertIn("needs_review", recovered[0]["quality"]["flags"])

    def test_dictionary_entry_validation_adds_candidate_provenance(self):
        value = {"entries": [{
            "headword": "آپ آرٹ",
            "english_terms": ["op art"],
            "definition": "دیکھیے آپٹک آرٹ۔",
            "examples": [],
            "cross_references": ["آپٹک آرٹ"],
            "raw_text": "آپ آرٹ (op art) دیکھیے آپٹک آرٹ۔",
            "pages": ["017"],
            "entry_type": "cross_reference",
            "quality": {"score": 0.98, "flags": []},
        }]}
        records = validate_entries(value, ["/tmp/017.jpg"], "farhang")
        self.assertEqual(records[0]["id"], "farhang-017-001")
        self.assertEqual(records[0]["english_terms"], ["op art"])
        self.assertEqual(records[0]["quality"]["status"], "candidate")

    def test_dictionary_cross_reference_may_have_empty_definition(self):
        value = {"entries": [{
            "headword": "آتم کتھا",
            "english_terms": None,
            "definition": "",
            "examples": [],
            "cross_references": ["آپ بیتی"],
            "raw_text": "آتم کتھا دیکھیے آپ بیتی۔",
            "pages": ["018"],
            "entry_type": "cross_reference",
            "quality": {"score": 1.0, "flags": []},
        }]}
        records = validate_entries(value, ["/tmp/018.jpg"], "farhang")
        self.assertEqual(records[0]["english_terms"], [])

    def test_dictionary_ids_use_page_local_ordinals(self):
        def entry(headword, page):
            return {
                "headword": headword,
                "english_terms": [],
                "definition": "تعریف",
                "examples": [],
                "cross_references": [],
                "raw_text": f"{headword} تعریف",
                "pages": [page],
                "entry_type": "definition",
                "quality": {"score": 1.0, "flags": []},
            }

        value = {"entries": [
            entry("اول", "017"),
            entry("دوم", "017"),
            entry("سوم", "018"),
        ]}
        records = validate_entries(
            value, ["/tmp/017.jpg", "/tmp/018.jpg"], "farhang")
        self.assertEqual(
            [record["id"] for record in records],
            ["farhang-017-001", "farhang-017-002", "farhang-018-001"])

    def test_dictionary_extract_is_exposed_by_unified_cli(self):
        parser = khusrau.build_parser()
        args = parser.parse_args([
            "dictionary", "extract", "017.jpg",
            "--output", "entries.jsonl", "--response", "response.json",
        ])
        self.assertEqual(args.func, khusrau.cmd_dictionary_extract)

    def test_match_tie_uses_stable_poem_order(self):
        poems = [
            {"id": 2, "verses": ["defxxx"]},
            {"id": 1, "verses": ["abcxxx"]},
        ]
        match = ganjoor.Index(poems).match_line("abcdef", min_jaccard=0)
        self.assertEqual(match[0], 1)

    def test_load_index_rejects_missing_poem(self):
        with tempfile.TemporaryDirectory() as cache:
            with patch.object(ganjoor, "CACHE", cache):
                directory = ganjoor.cache_dir(999)
                with open(os.path.join(directory, "cat_999.json"), "w", encoding="utf-8") as f:
                    json.dump([{"id": 10, "title": "missing"}], f)
                with self.assertRaisesRegex(RuntimeError, "missing poem 10"):
                    ganjoor.load_index(999)

    def test_batch_caches_partial_without_an_api_call(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            response = os.path.join(directory, "page.response.json")
            partial = os.path.join(directory, "page.partial.txt")
            with open(response, "w", encoding="utf-8") as f:
                json.dump({"text": "partial", "finish": "MAX_TOKENS"}, f)
            with open(partial, "w", encoding="utf-8") as f:
                f.write("partial\n")
            with patch.object(ocr_batch, "read_page") as read_page:
                result = ocr_batch.one(image)
            read_page.assert_not_called()
            self.assertTrue(result.cached)
            self.assertEqual((result.status, result.detail), ("partial", "MAX_TOKENS"))

    def test_batch_never_rerequests_failed_or_supplied_pages(self):
        with tempfile.TemporaryDirectory() as directory:
            failed = os.path.join(directory, "001.jpg")
            supplied = os.path.join(directory, "002.jpg")
            with open(os.path.join(directory, "001.response.json"), "w", encoding="utf-8") as f:
                json.dump({"error": "HTTP 500"}, f)
            with open(os.path.join(directory, "002.txt"), "w", encoding="utf-8") as f:
                f.write("supplied reading\n")
            with patch.object(ocr_batch, "read_page") as read_page, \
                    redirect_stdout(StringIO()):
                result = ocr_batch.run([failed, supplied], workers=2)
            read_page.assert_not_called()
            self.assertEqual(result, 1)

    def test_batch_collapses_duplicate_paths_to_one_request(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            aliases = [image, image, os.path.join(directory, ".", "page.jpg")]
            with patch.object(ocr_batch, "read_page",
                              return_value={"text": "read", "finish": "STOP"}) as read_page, \
                    redirect_stdout(StringIO()):
                result = ocr_batch.run(aliases, workers=3)
            self.assertEqual(read_page.call_count, 1)
            self.assertEqual(result, 0)

    def test_forced_incomplete_rerun_keeps_canonical_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            text = os.path.join(directory, "page.txt")
            with open(text, "w", encoding="utf-8") as f:
                f.write("reviewed reading\n")
            truncated = {"text": "half", "finish": "MAX_TOKENS"}
            with patch.object(ocr_batch, "read_page", return_value=truncated) as read_page, \
                    redirect_stdout(StringIO()):
                result = ocr_batch.run([image], workers=1, force=True)
            self.assertEqual(read_page.call_count, 1)
            self.assertEqual(result, 1)
            with open(text, encoding="utf-8") as f:
                self.assertEqual(f.read(), "reviewed reading\n")
            with open(os.path.join(directory, "page.partial.txt"), encoding="utf-8") as f:
                self.assertEqual(f.read(), "half\n")

    def test_batch_reports_damaged_pages_and_still_processes_healthy_ones(self):
        with tempfile.TemporaryDirectory() as directory:
            corrupt_response, corrupt_attempt, healthy = (
                os.path.join(directory, f"{n}.jpg") for n in ("001", "002", "003"))
            with open(os.path.join(directory, "001.txt"), "w", encoding="utf-8") as f:
                f.write("reading\n")
            with open(os.path.join(directory, "001.response.json"), "w", encoding="utf-8") as f:
                f.write('{"text": "rea')
            attempts = os.path.join(directory, "002.attempts")
            os.makedirs(attempts)
            with open(os.path.join(attempts, "20261011T000000000000Z-abcdef.response.json"),
                      "w", encoding="utf-8") as f:
                f.write('{"err')
            out = StringIO()
            with patch.object(ocr_batch, "read_page",
                              return_value={"text": "read", "finish": "STOP"}) as read_page, \
                    redirect_stdout(out):
                result = ocr_batch.run([corrupt_response, corrupt_attempt, healthy], workers=3)
            read_page.assert_called_once()
            self.assertEqual(read_page.call_args.args[0], healthy)
            self.assertEqual(result, 1)
            report = out.getvalue()
            self.assertIn("FAIL 001.jpg: corrupt (cached, 001.response.json:", report)
            self.assertIn("FAIL 002.jpg: corrupt (cached, "
                          "20261011T000000000000Z-abcdef.response.json:", report)
            self.assertIn("ok   003.jpg: complete", report)
            self.assertIn("1 ok, 2 failed, of 3", report)

    def test_batch_isolates_an_unexpected_page_exception(self):
        with tempfile.TemporaryDirectory() as directory:
            broken, healthy = (os.path.join(directory, f"{n}.jpg") for n in ("001", "002"))

            def read(path, *args, **kwargs):
                if path == broken:
                    raise OSError("disk unplugged")
                return {"text": "read", "finish": "STOP"}

            out = StringIO()
            with patch.object(ocr_batch, "read_page", side_effect=read), redirect_stdout(out):
                result = ocr_batch.run([broken, healthy], workers=2)
            self.assertEqual(result, 1)
            self.assertIn("FAIL 001.jpg: exception (new, OSError: disk unplugged)", out.getvalue())
            self.assertIn("ok   002.jpg: complete", out.getvalue())
            self.assertIn("1 ok, 1 failed, of 2", out.getvalue())

    def test_status_reports_corrupt_artifacts_as_unfinished(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "001.jpg")
            with open(os.path.join(directory, "001.txt"), "w", encoding="utf-8") as f:
                f.write("reading\n")
            with open(os.path.join(directory, "001.response.json"), "w", encoding="utf-8") as f:
                f.write("{")
            out = StringIO()
            with redirect_stdout(out):
                result = khusrau.main(["status", image])
            self.assertEqual(result, 1)
            self.assertIn("corrupt", out.getvalue())

    def test_ocr_page_script_reuses_existing_canonical_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            with open(os.path.join(directory, "page.txt"), "w", encoding="utf-8") as f:
                f.write("existing\n")
            with patch.object(ocr_batch, "read_page") as read_page, \
                    redirect_stdout(StringIO()):
                result = ocr_page.main([image])
            read_page.assert_not_called()
            self.assertEqual(result, 0)

    def test_dictionary_mode_uses_two_column_ocr_prompt(self):
        with patch.object(ocr_page, "image_part", return_value={"image": "test"}), \
                patch.object(ocr_page, "generate", return_value={"finish": "STOP"}) as generate:
            result = ocr_page.read_page("page.jpg", mode="dictionary")
        self.assertEqual(result["finish"], "STOP")
        system = generate.call_args.kwargs["system"]
        self.assertIn("RIGHT column completely", system)
        self.assertIn("modern printed Urdu", system)

    def test_vertex_endpoint_defaults_to_global_location(self):
        import vertex
        with patch.object(vertex, "project_id", return_value="test-proj"), \
                patch.dict(os.environ, {}, clear=True):
            endpoint = vertex._endpoint("gemini-custom-flash")
            self.assertIn("https://aiplatform.googleapis.com/v1/projects/test-proj/locations/global/", endpoint)
            self.assertIn("/publishers/google/models/gemini-custom-flash:generateContent", endpoint)

    def test_vertex_endpoint_respects_custom_location(self):
        import vertex
        with patch.object(vertex, "project_id", return_value="test-proj"), \
                patch.dict(os.environ, {"VERTEX_LOCATION": "europe-west1"}):
            endpoint = vertex._endpoint("gemini-custom-flash")
            self.assertIn("https://europe-west1-aiplatform.googleapis.com/v1/projects/test-proj/locations/europe-west1/", endpoint)

    def test_vertex_generate_falls_back_when_primary_model_404s(self):
        import vertex
        import urllib.error
        from io import BytesIO

        called_models = []

        def fake_urlopen(req, timeout=300):
            url = req.full_url
            for m in ("gemini-3.8-flash", "gemini-3.7-flash"):
                if m in url:
                    called_models.append(m)
            if "gemini-3.8-flash" in url:
                fp = BytesIO(b"model not found")
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, fp)
            body = json.dumps({"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "fallback text"}]}}]}).encode()
            return BytesIO(body)

        with patch.dict(os.environ, {"HTR_MODEL_ACCESS": "1"}), \
                patch.object(vertex, "token", return_value="fake-token"), \
                patch.object(vertex, "project_id", return_value="test-proj"), \
                patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = vertex.generate([{"text": "test"}])
            self.assertEqual(res["text"], "fallback text")
            self.assertEqual(res["model"], "gemini-3.7-flash")
            self.assertEqual(called_models, ["gemini-3.8-flash", "gemini-3.7-flash"])

    def test_dictionary_extract_inherits_model_from_config_or_env(self):
        with tempfile.TemporaryDirectory() as directory:
            config = os.path.join(directory, "khusrau.toml")
            with open(config, "w", encoding="utf-8") as f:
                f.write('[ocr]\nmodel = "gemini-test-custom"\nthinking = "high"\n')
            with patch("dictionary_extract.extract") as mock_extract, \
                    redirect_stdout(StringIO()):
                khusrau.main([
                    "--config", config,
                    "dictionary", "extract",
                    "001.jpg", "-o", "out.jsonl", "--response", "resp.json",
                ])
            mock_extract.assert_called_once()
            self.assertEqual(mock_extract.call_args.args[4], "gemini-test-custom")
            self.assertEqual(mock_extract.call_args.args[5], "high")

    def test_status_is_local_and_reports_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            with patch.object(ganjoor, "_get") as network:
                with redirect_stdout(StringIO()):
                    result = khusrau.main(["status", image])
            network.assert_not_called()
            self.assertEqual(result, 1)

    def test_status_reports_explicit_artifact_states(self):
        with tempfile.TemporaryDirectory() as directory:
            images = [os.path.join(directory, f"{n}.jpg") for n in ("001", "002", "003")]
            with open(os.path.join(directory, "001.txt"), "w", encoding="utf-8") as f:
                f.write("hand corrected\n")
            with open(os.path.join(directory, "001.response.json"), "w", encoding="utf-8") as f:
                json.dump({"text": "model read", "finish": "STOP"}, f)
            with open(os.path.join(directory, "002.txt"), "w", encoding="utf-8") as f:
                f.write("supplied reading\n")
            out = StringIO()
            with redirect_stdout(out):
                result = khusrau.main(["status", *images[:2]])
            self.assertEqual(result, 0)
            self.assertIn("edited", out.getvalue())
            self.assertIn("unrecorded", out.getvalue())
            with redirect_stdout(StringIO()):
                self.assertEqual(khusrau.main(["status", *images]), 1)

    def test_ocr_force_is_explicit_and_passed_through(self):
        with patch.object(ocr_batch, "run", return_value=0) as run:
            khusrau.main(["ocr", "page.jpg"])
            khusrau.main(["ocr", "--force", "page.jpg"])
        self.assertFalse(run.call_args_list[0].kwargs["force"])
        self.assertTrue(run.call_args_list[1].kwargs["force"])

    def test_align_writes_data_and_metadata_atomically(self):
        class FakeIndex:
            poems = {1: {"title": "Test poem"}}
            lines = [(1, 0, set(), "line")]

        record = {
            "page": "page.txt",
            "poem_id": 1,
            "poem_title": "Test poem",
            "tier": "gold",
            "matched_lines": 1,
            "distinct_verses": 1,
            "cer_matched_span": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.txt")
            output = os.path.join(directory, "pairs.jsonl")
            with open(page, "w", encoding="utf-8") as f:
                f.write("line\n")
            with patch.object(manufacture, "load_index", return_value=FakeIndex()), \
                    patch.object(manufacture, "process", return_value=([record], 1)), \
                    redirect_stdout(StringIO()):
                result = manufacture.run([999], [page], output)
            self.assertEqual(result, 0)
            with open(output, encoding="utf-8") as f:
                self.assertEqual(json.loads(f.readline())["poem_id"], 1)
            with open(output + ".meta.json", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["categories"], [999])
            self.assertIn("retrieval", metadata["profiles"])

    def test_align_write_failure_cleans_up_temporary_files(self):
        class FakeIndex:
            poems = {1: {"title": "Test poem"}}
            lines = [(1, 0, set(), "line")]

        record = {
            "page": "page.txt",
            "poem_id": 1,
            "poem_title": "Test poem",
            "tier": "gold",
            "matched_lines": 1,
            "distinct_verses": 1,
            "cer_matched_span": 0.0,
        }
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.txt")
            output = os.path.join(directory, "pairs.jsonl")
            with open(page, "w", encoding="utf-8") as f:
                f.write("line\n")
            with patch.object(manufacture, "load_index", return_value=FakeIndex()), \
                    patch.object(manufacture, "process", return_value=([record], 1)), \
                    patch("json.dumps", side_effect=ValueError("serialization error")):
                with self.assertRaises(ValueError):
                    manufacture.run([999], [page], output)
            leftovers = [n for n in os.listdir(directory) if n.startswith(".khusrau-")]
            self.assertEqual(leftovers, [])

    def test_cli_requires_a_command(self):
        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit) as raised:
                khusrau.main([])
        self.assertEqual(raised.exception.code, 2)

    def test_manifest_drives_local_alignment_workflow(self):
        verses = [
            "دل نادان تجھے ہوا کیا ہے",
            "آخر اس درد کی دوا کیا ہے",
            "ہم ہیں مشتاق اور وہ بیزار",
            "یا الہی یہ ماجرا کیا ہے",
        ]
        with tempfile.TemporaryDirectory() as directory:
            cache = os.path.join(directory, "cache")
            page = os.path.join(directory, "page.txt")
            output = os.path.join(directory, "pairs.jsonl")
            config = os.path.join(directory, "khusrau.toml")
            with patch.object(ganjoor, "CACHE", cache):
                reference = ganjoor.cache_dir(999)
                with open(os.path.join(reference, "cat_999.json"), "w", encoding="utf-8") as f:
                    json.dump([{"id": 1, "title": "Test poem"}], f)
                with open(os.path.join(reference, "poem_1.json"), "w", encoding="utf-8") as f:
                    json.dump({"id": 1, "title": "Test poem", "verses": verses}, f)
                with open(page, "w", encoding="utf-8") as f:
                    f.write("\n".join(verses) + "\n")
                with open(config, "w", encoding="utf-8") as f:
                    f.write(
                        "[pages]\n"
                        'transcriptions = ["page.txt"]\n'
                        "[reference.ganjoor]\n"
                        "categories = [999]\n"
                        "[alignment]\n"
                        'output = "pairs.jsonl"\n')
                with redirect_stdout(StringIO()):
                    result = khusrau.main(["--config", config, "align"])
            self.assertEqual(result, 0)
            with open(output, encoding="utf-8") as f:
                record = json.loads(f.readline())
            self.assertEqual(record["tier"], "gold")
            self.assertEqual(record["distinct_verses"], 4)
            with open(output + ".meta.json", encoding="utf-8") as f:
                metadata = json.load(f)
            self.assertEqual(metadata["thresholds"]["min_jaccard"], 0.55)

    def test_rekhta_verify_is_local(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(os.path.join(directory, "poem.txt"), "w", encoding="utf-8") as f:
                f.write("first line\nsecond line\n")
            with patch("rekhta.get") as network, redirect_stdout(StringIO()):
                result = khusrau.main(
                    ["reference", "rekhta-verify", directory, "--poet", "test"])
            network.assert_not_called()
            self.assertEqual(result, 0)

    def test_server_rendered_story_parser_selects_largest_container(self):
        source = """
        <div class="pMC" data-pc="1"><div class="w">چھوٹا متن یہاں ہے</div></div>
        <div class="pMC" data-pc="2">
          <div class="w"><p><span>پہلا مکمل پیراگراف یہاں موجود ہے۔</span></p></div>
          <div class="w"><p><span>دوسرا مکمل پیراگراف یہاں موجود ہے۔</span></p></div>
        </div>
        """
        self.assertEqual(len(story_from_html(source)), 2)

    def test_prose_score_is_zero_for_identical_text(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.txt")
            reference = os.path.join(directory, "reference.txt")
            text = "یہ ایک مکمل آزمائشی عبارت ہے۔\n"
            for path in (page, reference):
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
            result = score_prose([page], reference)
            self.assertEqual(result["token_error_rate"], 0)
            self.assertEqual(result["folded_character_error_rate"], 0)

    def test_cli_scores_prose_and_writes_json(self):
        with tempfile.TemporaryDirectory() as directory:
            page = os.path.join(directory, "page.txt")
            reference = os.path.join(directory, "reference.txt")
            output = os.path.join(directory, "score.json")
            for path in (page, reference):
                with open(path, "w", encoding="utf-8") as f:
                    f.write("یہ ایک مکمل آزمائشی عبارت ہے۔\n")
            with redirect_stdout(StringIO()):
                result = khusrau.main([
                    "score", "prose", reference, page, "-o", output])
            self.assertEqual(result, 0)
            with open(output, encoding="utf-8") as f:
                score = json.load(f)
            self.assertEqual(score["token_error_rate"], 0)
            self.assertEqual(score["strict_character_error_rate"], 0)

    def test_page_manifest_and_candidate_export_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "168.jpg")
            text = os.path.join(directory, "168.txt")
            response = os.path.join(directory, "168.response.json")
            manifest = os.path.join(directory, "pages.jsonl")
            destination = os.path.join(directory, "export")
            with open(image, "wb") as f:
                f.write(b"image")
            with open(text, "w", encoding="utf-8") as f:
                f.write("صفحہ متن\n")
            with open(response, "w", encoding="utf-8") as f:
                json.dump({
                    "finish": "STOP",
                    "usage": {"totalTokenCount": 10},
                    "request": {"model": "test-model", "mode": "prose", "thinking": "low"},
                }, f)
            records = build_manifest(
                [image], manifest, "anandi", mode="prose",
                reference="rekhta:anandi")
            self.assertEqual(records[0]["id"], "anandi-168")
            self.assertEqual(records[0]["ocr"]["model"], "test-model")
            with self.assertRaisesRegex(ValueError, "not reviewed/gold"):
                export_pages(manifest, destination)
            exported = export_pages(manifest, destination, allow_candidate=True)
            self.assertEqual(exported[0]["export_image"], "anandi-168.jpg")
            self.assertTrue(os.path.isfile(os.path.join(destination, "anandi-168.txt")))

    def test_manifest_records_attempt_provenance(self):
        from ocr_artifacts import PageArtifacts

        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "168.jpg")
            with open(image, "wb") as f:
                f.write(b"image")
            page = PageArtifacts(image)
            page.record({"text": "read", "finish": "STOP"}, {"model": "m", "mode": "verse"})
            with open(page.response, encoding="utf-8") as f:
                attempt = json.load(f)["attempt"]
            records = build_manifest([image], os.path.join(directory, "m.jsonl"), "ds")
        self.assertEqual(records[0]["ocr"]["attempt"], attempt)
        self.assertEqual(records[0]["ocr"]["model"], "m")

    def test_page_export_detects_changed_text(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "001.jpg")
            text = os.path.join(directory, "001.txt")
            manifest = os.path.join(directory, "pages.jsonl")
            with open(image, "wb") as f:
                f.write(b"image")
            with open(text, "w", encoding="utf-8") as f:
                f.write("original\n")
            build_manifest([image], manifest, "test", status="reviewed")
            with open(text, "w", encoding="utf-8") as f:
                f.write("changed\n")
            with self.assertRaisesRegex(ValueError, "content hash mismatch"):
                export_pages(manifest, os.path.join(directory, "export"))

    def test_corpus_load_missing_directory_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            corpus.load("/nonexistent/corpus/path")
        self.assertIn("REKHTA_CORPUS", str(ctx.exception))

    def test_corpus_load_without_path_or_env_raises_value_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as ctx:
                corpus.load()
            self.assertIn("REKHTA_CORPUS", str(ctx.exception))

    def test_corpus_load_uses_rekhta_corpus_env(self):
        with tempfile.TemporaryDirectory() as directory:
            ghazal_path = os.path.join(directory, "sample-ghazal-mirza-ghalib-ghazals.txt")
            with open(ghazal_path, "w", encoding="utf-8") as f:
                f.write("نقش فریادی ہے کس کی شوخی تحریر کا\n" * 4)
            with patch.dict(os.environ, {"REKHTA_CORPUS": directory}):
                ghazals = corpus.load()
            self.assertEqual(len(ghazals), 1)

    def test_corpus_load_reads_valid_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            ghazal_path = os.path.join(directory, "sample-ghazal-mirza-ghalib-ghazals.txt")
            with open(ghazal_path, "w", encoding="utf-8") as f:
                f.write("نقش فریادی ہے کس کی شوخی تحریر کا\n")
                f.write("کاغذی ہے پیرہن ہر پیکر تصویر کا\n")
                f.write("کاو کاو سخت جانی ہائے تنہائی نہ پوچھ\n")
                f.write("صبح کرنا شام کا لانا ہے جوئے شیر کا\n")
            ghazals = corpus.load(directory)
            self.assertEqual(len(ghazals), 1)
            self.assertEqual(ghazals[0].gid, "sample-ghazal-mirza-ghalib-ghazals")
            self.assertEqual(ghazals[0].poet, "mirza-ghalib")
            self.assertEqual(len(ghazals[0].lines), 4)

    def test_selftest_align_without_corpus_exits_cleanly(self):
        import selftest_align
        err = StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(err):
            code = selftest_align.run([])
        self.assertEqual(code, 2)
        self.assertIn("REKHTA_CORPUS", err.getvalue())

    def test_reject_test_without_corpus_exits_cleanly(self):
        import reject_test
        err = StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(err):
            code = reject_test.main([])
        self.assertEqual(code, 2)
        self.assertIn("REKHTA_CORPUS", err.getvalue())

    def test_corpus_main_without_corpus_exits_cleanly(self):
        err = StringIO()
        with patch.dict(os.environ, {}, clear=True), redirect_stderr(err):
            code = corpus.main([])
        self.assertEqual(code, 2)
        self.assertIn("REKHTA_CORPUS", err.getvalue())

    def test_align_verse_best_offset_empty_or_short_inputs(self):
        from align_verse import best_offset
        self.assertEqual(best_offset(["a", "b"], []), (None, float("inf")))
        self.assertEqual(best_offset([], []), (None, float("inf")))
        self.assertEqual(best_offset([], ["a"]), (None, float("inf")))
        self.assertEqual(best_offset(["a"], ["a", "b"]), (None, float("inf")))

    def test_align_verse_cli_handles_empty_or_short_inputs_cleanly(self):
        import align_verse
        with tempfile.TemporaryDirectory() as directory:
            empty = os.path.join(directory, "empty.txt")
            with open(empty, "w", encoding="utf-8") as f:
                f.write("## only headings\n\n")
            one = os.path.join(directory, "one.txt")
            with open(one, "w", encoding="utf-8") as f:
                f.write("دل نادان تجھے ہوا کیا ہے\n")
            two = os.path.join(directory, "two.txt")
            with open(two, "w", encoding="utf-8") as f:
                f.write("دل نادان تجھے ہوا کیا ہے\nآخر اس درد کی دوا کیا ہے\n")

            err = StringIO()
            with redirect_stderr(err):
                code = align_verse.main([one, empty])
            self.assertEqual(code, 2)
            self.assertIn("reference has no usable verse lines", err.getvalue())

            err = StringIO()
            with redirect_stderr(err):
                code = align_verse.main([one, two])
            self.assertEqual(code, 1)
            self.assertIn("hypothesis has fewer lines than reference", err.getvalue())


if __name__ == "__main__":
    unittest.main()
