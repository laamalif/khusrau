#!/usr/bin/env python3
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from unittest.mock import patch

import ganjoor
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
                f.write("anchor one\nrecover me\nanchor two\n")
            with patch("discover_sequence.process", return_value=([record], 3)), \
                    patch("discover_sequence._candidate_score",
                          return_value=(0.4, "reference two")):
                rows = discover(FakeIndex(), page)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate"]["verse"], 11)
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
                _, output, status, finish = ocr_batch.one(image)
            read_page.assert_not_called()
            self.assertEqual(output, partial)
            self.assertEqual((status, finish), ("cached_partial", "MAX_TOKENS"))

    def test_dictionary_mode_uses_two_column_ocr_prompt(self):
        with patch.object(ocr_page, "image_part", return_value={"image": "test"}), \
                patch.object(ocr_page, "generate", return_value={"finish": "STOP"}) as generate:
            result = ocr_page.read_page("page.jpg", mode="dictionary")
        self.assertEqual(result["finish"], "STOP")
        system = generate.call_args.kwargs["system"]
        self.assertIn("RIGHT column completely", system)
        self.assertIn("modern printed Urdu", system)

    def test_status_is_local_and_reports_pending(self):
        with tempfile.TemporaryDirectory() as directory:
            image = os.path.join(directory, "page.jpg")
            with patch.object(ganjoor, "_get") as network:
                with redirect_stdout(StringIO()):
                    result = khusrau.main(["status", image])
            network.assert_not_called()
            self.assertEqual(result, 1)

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


if __name__ == "__main__":
    unittest.main()
