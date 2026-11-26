"""Hand-calculated unit fixtures, never scientific data or actual human reviews."""
import copy
import math
import unittest
from evaluation.metrics import evaluate


def record(key="a", value=0.25, split="Test-Comp", preserved=True):
    return {"id": key, "split": split, "y": [value, 0, 0, 0, 0, 0],
        "modified_mask": [int(value != 0), 0, 0, 0, 0, 0],
        "preserved_mask": [0, int(preserved), 0, 0, 0, 0], "source": "template",
        "semantic_group_id": "g", "ast": ([{"attribute": 0, "operation": "edit",
        "direction": 1, "magnitude": value}] if value else []) +
        ([{"attribute": 1, "operation": "preserve"}] if preserved else [])}


def prediction(row, y=None, mask=None):
    return {"id": row["id"], "split": row["split"], "y": list(row["y"] if y is None else y),
        "modified_mask": list(row["modified_mask"] if mask is None else mask)}


def paraphrase():
    return {"pair_id": "p", "split": "Test-Comp", "left_id": "a", "right_id": "b",
        "semantic_group_id": "g", "verification_status": "human-verified", "human_review_id": "r"}


def human_evidence():
    return {"r": {"pair_id": "p", "verification_status": "human-verified",
        "reviewer_id": "UNIT_TEST_FIXTURE_NOT_A_REAL_REVIEWER", "reviewed_at": "unit-test",
        "outcome": "approved", "endpoint_ids": ["a", "b"]}}


def intensity():
    pair = {"pair_id": "i", "split": "Test-Comp", "weak_id": "a", "strong_id": "b",
        "attribute_index": 0, "verification_status": "ast-verified", "other_controls_equal": True,
        "review_id": "i-review"}
    evidence = {"i-review": {"verification_status": "ast-verified", "weak_id": "a",
        "strong_id": "b", "attribute_index": 0, "generator_version": "unit-fixture"}}
    return pair, evidence


class MetricsTests(unittest.TestCase):
    def test_perfect_and_denominators(self):
        row = record()
        metrics = evaluate([row], [prediction(row)], split="Test-Comp")["metrics"]
        self.assertEqual(metrics["attribute_f1"]["value"], 1)
        self.assertEqual(metrics["mae"]["value"], 0)
        self.assertEqual(metrics["compositional_accuracy"]["value"], 1)
        self.assertEqual(metrics["negation_accuracy"]["eligible_count"], 1)
        self.assertEqual(metrics["negation_accuracy"]["value"], 1)
        self.assertIsNone(metrics["paraphrase_consistency"]["value"])
        self.assertIsNone(metrics["intensity_accuracy"]["value"])

    def test_hand_computed_f1_mae(self):
        row = record()
        pred = prediction(row, [0.5, 0, 0.25, 0, 0, 0], [1, 0, 1, 0, 0, 0])
        metrics = evaluate([row], [pred], split="Test-Comp")["metrics"]
        self.assertAlmostEqual(metrics["attribute_f1"]["value"], 2/3)
        self.assertAlmostEqual(metrics["mae"]["value"], 0.5/6)
        self.assertEqual(metrics["compositional_accuracy"]["value"], 0)

    def test_strict_epsilon_at_zero_target(self):
        row = record()
        for value, expected in [(math.nextafter(0.1, 0), 1), (0.1, 0), (math.nextafter(0.1, 1), 0)]:
            with self.subTest(value=value):
                pred = prediction(row, [0.25, value, 0, 0, 0, 0])
                metrics = evaluate([row], [pred], split="Test-Comp")["metrics"]
                self.assertEqual(metrics["compositional_accuracy"]["value"], expected)
                self.assertEqual(metrics["negation_accuracy"]["value"], expected)

    def test_unmentioned_is_not_preservation(self):
        row = record(preserved=False)
        metric = evaluate([row], [prediction(row)], split="Test-Comp")["metrics"]["negation_accuracy"]
        self.assertEqual(metric["eligible_count"], 0)
        self.assertIsNone(metric["value"])

    def test_all_negative_f1_and_false_positive(self):
        row = record(value=0)
        self.assertIsNone(evaluate([row], [prediction(row)], split="Test-Comp")["metrics"]["attribute_f1"]["value"])
        pred = prediction(row, mask=[1, 0, 0, 0, 0, 0])
        self.assertEqual(evaluate([row], [pred], split="Test-Comp")["metrics"]["attribute_f1"]["value"], 0)

    def test_empty_and_noncomp(self):
        result = evaluate([], [], split="Test-Human")
        self.assertTrue(all(v["value"] is None for v in result["metrics"].values()))
        row = record(split="Test-ID")
        self.assertIsNone(evaluate([row], [prediction(row)], split="Test-ID")["metrics"]["compositional_accuracy"]["value"])

    def test_constant_predictor_diagnostic(self):
        rows = [record("a"), record("b")]
        preds = [prediction(row, [0]*6, [0]*6) for row in rows]
        result = evaluate(rows, preds, split="Test-Comp", paraphrase_pairs=[paraphrase()], review_evidence=human_evidence())
        self.assertEqual(result["metrics"]["paraphrase_consistency"]["value"], 0)
        self.assertAlmostEqual(result["metrics"]["mae"]["value"], 0.25/6)

    def test_consistency_is_sum_not_dimension_mean(self):
        rows = [record("a"), record("b")]
        preds = [prediction(rows[0], [0]*6), prediction(rows[1], [1]*6)]
        result = evaluate(rows, preds, split="Test-Comp", paraphrase_pairs=[paraphrase()], review_evidence=human_evidence())
        self.assertEqual(result["metrics"]["paraphrase_consistency"]["value"], 6)

    def test_review_identifier_alone_is_not_evidence(self):
        rows = [record("a"), record("b")]
        result = evaluate(rows, [prediction(r) for r in rows], split="Test-Comp", paraphrase_pairs=[paraphrase()])
        metric = result["metrics"]["paraphrase_consistency"]
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["exclusion_count"], 1)
        bad = human_evidence()
        bad["r"]["endpoint_ids"] = ["a", "WRONG"]
        result = evaluate(rows, [prediction(r) for r in rows], split="Test-Comp", paraphrase_pairs=[paraphrase()], review_evidence=bad)
        self.assertIsNone(result["metrics"]["paraphrase_consistency"]["value"])

    def test_intensity_absolute_order_and_tie(self):
        rows = [record("a", 0.25), record("b", 0.75)]
        pair, evidence = intensity()
        for values, expected in [((-0.25, -0.75), 1), ((0.5, 0.5), 0), ((0.75, 0.25), 0)]:
            preds = [prediction(r, [v, 0, 0, 0, 0, 0]) for r, v in zip(rows, values)]
            result = evaluate(rows, preds, split="Test-Comp", intensity_pairs=[pair], review_evidence=evidence)
            self.assertEqual(result["metrics"]["intensity_accuracy"]["value"], expected)

    def test_ast_evidence_cannot_verify_human_text(self):
        rows = [record("a", 0.25), record("b", 0.75)]
        rows[1]["source"] = "human"
        pair, evidence = intensity()
        result = evaluate(rows, [prediction(r) for r in rows], split="Test-Comp", intensity_pairs=[pair], review_evidence=evidence)
        self.assertIsNone(result["metrics"]["intensity_accuracy"]["value"])

    def test_invalid_prediction_inputs(self):
        row = record()
        for field, bad in [("y", [float("nan")]*6), ("y", [2]*6), ("y", [0]*5),
                           ("modified_mask", [2]*6), ("modified_mask", [True]*6),
                           ("id", "wrong"), ("split", "Train")]:
            with self.subTest(field=field, bad=bad):
                pred = prediction(row)
                pred[field] = bad
                with self.assertRaises(ValueError):
                    evaluate([row], [pred], split="Test-Comp")
        for preds in [[], [prediction(row), prediction(row)]]:
            with self.assertRaises(ValueError):
                evaluate([row], preds, split="Test-Comp")

    def test_invalid_gold_and_pairs(self):
        rows = [record("a"), record("b")]
        badrow = copy.deepcopy(rows[0])
        badrow["preserved_mask"][0] = 1
        with self.assertRaises(ValueError):
            evaluate([badrow], [prediction(badrow)], split="Test-Comp")
        for change in [{"right_id": "missing"}, {"split": "Train"}, {"right_id": "a"}, {"semantic_group_id": "wrong"}]:
            pair = paraphrase()
            pair.update(change)
            with self.assertRaises(ValueError):
                evaluate(rows, [prediction(r) for r in rows], split="Test-Comp", paraphrase_pairs=[pair])
        with self.assertRaises(ValueError):
            evaluate(rows, [prediction(r) for r in rows], split="Test-Comp", paraphrase_pairs=[paraphrase(), paraphrase()])


if __name__ == "__main__":
    unittest.main()
