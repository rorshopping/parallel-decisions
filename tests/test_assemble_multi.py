"""Regression: multi-field inclusion must follow P(true), not winner confidence."""
import unittest

from parallel_decisions import Decider, Schema


class _Row:
    def __init__(self, true_prob):
        self.value = true_prob >= 0.5
        self.probability = max(true_prob, 1 - true_prob)
        self.distribution = {"true": true_prob, "false": 1 - true_prob}


class AssembleMultiTests(unittest.TestCase):
    def setUp(self):
        self.schema = Schema({"tags": {"type": "multi",
                                       "choices": ["invoice", "meeting"]}})

    def test_confident_false_rows_are_not_included(self):
        result = Decider._assemble_multi(self.schema.fields["tags"],
                                         {0: _Row(0.01), 1: _Row(0.99)})
        self.assertEqual(result.value, ["meeting"])
        self.assertEqual(result.distribution, {"invoice": 0.01, "meeting": 0.99})
        self.assertEqual(result.probability, 0.99)

    def test_missing_row_scores_zero(self):
        result = Decider._assemble_multi(self.schema.fields["tags"], {})
        self.assertEqual(result.value, [])
        self.assertEqual(result.distribution, {"invoice": 0.0, "meeting": 0.0})


if __name__ == "__main__":
    unittest.main()
