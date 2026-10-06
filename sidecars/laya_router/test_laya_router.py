"""Tests for the Laya capability router that need neither Laya nor torch.

Run from the repository root:

    python -m unittest sidecars/laya_router/test_laya_router.py
"""

import io
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import engine  # noqa: E402
import main  # noqa: E402

# `PromptIntent::to_capability_name` in src-tauri/src/capability/intent.rs.
# Adapters are bound under these keys in every model package's manifest, so a
# label Laya returns that is not one of them can never select an adapter.
RUST_CAPABILITY_KEYS = {"coding", "mathematics", "reasoning", "tool-calling", "research", "general"}


def word_tokenizer(text):
    """Stand-in tokenizer: one id per whitespace-separated word."""
    return list(range(len(text.split())))


class QuestionTests(unittest.TestCase):
    def test_labels_are_exactly_the_capability_keys(self):
        q = engine.load_question()
        criteria = q["questions"][q["question_id"]]["criteria"]
        self.assertEqual(set(criteria), RUST_CAPABILITY_KEYS)

    def test_the_instruction_names_the_state_field_it_reads(self):
        # Laya's convention: the field the question reads is named in backticks,
        # and the engine sends the prompt under that key.
        q = engine.load_question()
        self.assertIn("`request`", q["questions"][q["question_id"]]["instructions"])

    def test_every_description_fits_the_head_budget_with_room_to_spare(self):
        q = engine.load_question()
        spec = q["questions"][q["question_id"]]
        # Word counts understate subword tokens; doubling them is a generous
        # stand-in, and the real tokenizer is re-checked at load time.
        engine.check_option_budget(
            "english",
            lambda text: word_tokenizer(text) * 2,
            spec["instructions"],
            spec["criteria"],
            192,
        )

    def test_the_fingerprint_follows_the_wording(self):
        a = engine.load_question()
        self.assertEqual(a["fingerprint"], engine.load_question()["fingerprint"])
        self.assertEqual(len(a["fingerprint"]), 64)


class BudgetTests(unittest.TestCase):
    def test_an_option_over_the_token_cap_is_refused(self):
        with self.assertRaises(ValueError) as caught:
            engine.check_option_budget("english", word_tokenizer, "pick one", {"coding": "word " * 60}, 192)
        self.assertIn("coding", str(caught.exception))

    def test_options_that_crowd_out_the_instruction_are_refused(self):
        crit = {"o%d" % i: "word " * 40 for i in range(5)}
        with self.assertRaises(ValueError):
            engine.check_option_budget("english", word_tokenizer, "pick one", crit, 192)


class ThreadTests(unittest.TestCase):
    def test_an_explicit_count_wins(self):
        self.assertEqual(engine.thread_count("3", 32), 3)

    def test_half_the_logical_cpus_capped_at_eight(self):
        self.assertEqual(engine.thread_count("", 16), 8)
        self.assertEqual(engine.thread_count("", 32), 8)
        self.assertEqual(engine.thread_count("", 6), 3)
        self.assertEqual(engine.thread_count("", 1), 1)


class FakeEngine:
    def load(self):
        return {"loaded": True}

    def status(self):
        return {"loaded": True}

    def classify(self, prompt):
        if prompt == "boom":
            raise RuntimeError("inference failed")
        return {"choice": "coding", "probabilities": {"coding": 0.9, "general": 0.1}, "answerConfidence": 0.9}

    def choose(self, text, instructions, labels):
        return {"choice": labels[0], "probabilities": {labels[0]: 0.7}, "answerConfidence": 0.7}


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.methods = main.methods_for(FakeEngine())

    def call(self, payload):
        return main.handle(json.dumps(payload), self.methods)

    def test_ping(self):
        self.assertEqual(self.call({"jsonrpc": "2.0", "id": 1, "method": "laya.ping"})["result"], {"ok": True})

    def test_classify_returns_the_engines_answer_under_the_same_id(self):
        r = self.call({"jsonrpc": "2.0", "id": 7, "method": "laya.classify", "params": {"prompt": "fix this"}})
        self.assertEqual(r["id"], 7)
        self.assertEqual(r["result"]["choice"], "coding")

    def test_choose_passes_the_labels_through(self):
        r = self.call({
            "jsonrpc": "2.0", "id": 11, "method": "laya.choose",
            "params": {"text": "credit memos for NBFCs", "instructions": "What does `request` do?",
                       "labels": ["Finance", "Law"]},
        })
        self.assertEqual(r["result"]["choice"], "Finance")

    def test_choose_rejects_instructions_that_do_not_name_the_state_field(self):
        engine_ = engine.CapabilityEngine.__new__(engine.CapabilityEngine)
        engine_.router = object()  # loaded, as far as the guard is concerned
        engine_.installed = ["english"]
        with self.assertRaises(ValueError):
            engine_.choose("text", "What is this for?", ["Finance"])
        with self.assertRaises(ValueError):
            engine_.choose("text", "What does `request` do?", [])

    def test_an_engine_failure_is_an_error_response_not_a_crash(self):
        r = self.call({"jsonrpc": "2.0", "id": 8, "method": "laya.classify", "params": {"prompt": "boom"}})
        self.assertEqual(r["id"], 8)
        self.assertEqual(r["error"]["code"], -32603)

    def test_unknown_methods_and_bad_json_are_reported(self):
        self.assertEqual(self.call({"jsonrpc": "2.0", "id": 9, "method": "nope"})["error"]["code"], -32601)
        self.assertEqual(main.handle("{not json", self.methods)["error"]["code"], -32700)

    def test_serve_writes_one_line_per_request_and_skips_blanks(self):
        stdin = io.StringIO('{"id":1,"method":"laya.ping"}\n\n{"id":2,"method":"laya.status"}\n')
        out = io.StringIO()
        main.serve(stdin, out, self.methods)
        lines = out.getvalue().splitlines()
        self.assertEqual([json.loads(l)["id"] for l in lines], [1, 2])

    def test_non_ascii_prompts_survive_the_round_trip(self):
        stdin = io.StringIO(json.dumps({"id": 3, "method": "laya.classify", "params": {"prompt": "कोड ठीक करो"}}) + "\n")
        out = io.StringIO()
        main.serve(stdin, out, self.methods)
        line = out.getvalue().strip()
        line.encode("ascii")  # every response is plain ASCII on the wire
        self.assertEqual(json.loads(line)["id"], 3)


if __name__ == "__main__":
    unittest.main()
