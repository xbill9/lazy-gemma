import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lazy_eval
import train


def row(pid, k, n_think, n_ans, right=True, finish="eos"):
    gen = [7] * (n_think - 1) + [train.END_THINK] + [8] * n_ans
    return {"id": pid, "sample": k, "gen_ids": gen, "n_gen": len(gen), "right": right, "finish": finish,
            "prompt_ids": [2, 105]}


class Select(unittest.TestCase):
    def test_shortest_right_and_log(self):
        rows = [row("a", 0, 300, 20), row("a", 1, 100, 20), row("a", 2, 50, 10, right=False),
                row("b", 0, 90, 5, right=False)]
        chosen, log = train.select(rows)
        self.assertEqual([(r["id"], r["sample"]) for r in chosen], [("a", 1)])
        self.assertEqual(log[0], {"id": "a", "candidates": 3, "right": 2, "chosen_len": 120, "lens_right": [120, 320]})
        self.assertEqual(log[1]["chosen_len"], None)

    def test_truncated_never_chosen(self):
        chosen, _ = train.select([row("a", 0, 10, 2, finish="length"), row("a", 1, 40, 2)])
        self.assertEqual(chosen[0]["sample"], 1)

    def test_budget_pairs(self):
        rows = [row("a", 0, 100, 20), row("a", 1, 110, 20), row("a", 2, 400, 20), row("a", 3, 300, 20)]
        got = [(r["sample"], b) for r, b in train.select_budget(rows)]
        self.assertEqual(got, [(0, 128), (2, 512)])  # shortest, then longest right in another bucket

    def test_budget_single_bucket(self):
        got = train.select_budget([row("a", 0, 100, 20), row("a", 1, 120, 20)])
        self.assertEqual([(r["sample"], b) for r, b in got], [(0, 128)])

    def test_bucket_edges(self):
        self.assertEqual([train.bucket(n) for n in (1, 128, 129, 2048, 2049)], [128, 128, 256, 2048, None])


class Grade(unittest.TestCase):
    def test_answer_line_wins_over_last_number(self):
        self.assertTrue(lazy_eval.right("Answer: 42\n(checked 7 times)", "42"))

    def test_prompt_budget_text(self):
        self.assertIn("at most 256 tokens", lazy_eval.prompt("q?", 256))
        self.assertEqual(lazy_eval.prompt("q?"), lazy_eval.PROMPT.format(question="q?"))


if __name__ == "__main__":
    unittest.main()
