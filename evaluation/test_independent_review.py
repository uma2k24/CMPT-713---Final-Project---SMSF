"""Independent coordinator/Eval cases, separate from Agent C's metric tests."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from evaluation.metrics import evaluate


def gold(key, y, preserved=None, split='Test-Comp'):
    return {'id': key, 'split': split, 'y': y, 'modified_mask': [int(v != 0) for v in y],
            'preserved_mask': preserved or [0] * 6, 'schema_version': '1.0'}


def pred(row, y, modified):
    return {'id': row['id'], 'split': row['split'], 'y': y, 'modified_mask': modified,
            'schema_version': '1.0', 'model_version': 'independent-test-only'}


class IndependentReview(unittest.TestCase):
    def test_delivered_candidates_independent_holdout_and_hash_audit(self):
        directory = Path(__file__).resolve().parents[1] / 'data/candidates/v1'
        manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['release_eligible'])
        for name, expected_hash in manifest['files_sha256'].items():
            self.assertEqual(hashlib.sha256((directory / name).read_bytes()).hexdigest(), expected_hash)
        def read(name):
            return [json.loads(line) for line in (directory / name).read_text(encoding='utf-8').splitlines() if line.strip()]
        splits = {name: read(filename) for name, filename in [('Train', 'train.jsonl'), ('Val', 'val.jsonl'),
                  ('Test-ID', 'test_id.jsonl'), ('Test-Comp', 'test_comp.jsonl')]}
        self.assertEqual([len(x) for x in splits.values()], [800, 100, 100, 100])
        holdouts = [{0, 2}, {1, 4}, {3, 5}]
        by_id, semantics = {}, set()
        for split, records in splits.items():
            for row in records:
                self.assertEqual(row['split'], split)
                self.assertEqual(row['schema_version'], '1.0')
                self.assertEqual(row['source'], 'template')
                self.assertNotIn(row['id'], by_id)
                by_id[row['id']] = row
                signature = (tuple(row['y']), tuple(row['modified_mask']), tuple(row['preserved_mask']))
                self.assertNotIn(signature, semantics)
                semantics.add(signature)
                modified = {i for i, value in enumerate(row['modified_mask']) if value}
                if split == 'Test-Comp':
                    self.assertIn(modified, holdouts)
                else:
                    self.assertFalse(any(heldout <= modified for heldout in holdouts))
        primitives = {(i, value) for row in splits['Train'] for i, value in enumerate(row['y']) if value}
        self.assertEqual(primitives, {(i, sign * level) for i in range(6) for sign in (-1, 1) for level in (.25, .5, .75, 1.)})
        pairs = read('intensity_pairs.jsonl')
        self.assertEqual(len(pairs), 1650)
        for pair in pairs:
            a, b, index = by_id[pair['weak_id']], by_id[pair['strong_id']], pair['attribute_index']
            self.assertEqual(a['split'], b['split'])
            self.assertEqual(a['split'], pair['split'])
            self.assertEqual(pair['verification_status'], 'ast-verified')
            self.assertEqual(a['preserved_mask'], b['preserved_mask'])
            self.assertTrue(a['y'][index] * b['y'][index] > 0)
            self.assertLess(abs(a['y'][index]), abs(b['y'][index]))
            self.assertTrue(all(a['y'][i] == b['y'][i] for i in range(6) if i != index))
        self.assertEqual(read('paraphrase_pairs.jsonl'), [])

    def test_hand_calculation_with_missed_and_spurious_edits(self):
        a = gold('a', [.5, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0])
        b = gold('b', [0, 0, -.75, 0, 0, 0], [0, 0, 0, 0, 1, 0])
        predictions = [pred(a, [.25, .2, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0]),
                       pred(b, [0, 0, -.5, 0, 0, 0], [0] * 6)]
        metrics = evaluate([a, b], predictions, split='Test-Comp')['metrics']
        # TP=1, FP=1, FN=1. Absolute error=.25+.20+.25=.70, 12 dimensions.
        self.assertEqual(metrics['attribute_f1']['value'], .5)
        self.assertAlmostEqual(metrics['mae']['value'], .7 / 12)
        self.assertEqual(metrics['mae']['eligible_count'], 12)
        self.assertEqual(metrics['negation_accuracy']['value'], .5)
        self.assertEqual(metrics['negation_accuracy']['eligible_count'], 2)
        self.assertEqual(metrics['compositional_accuracy']['value'], 0.)

    def test_exact_epsilon_fails_both_tolerance_metrics(self):
        a = gold('epsilon', [0] * 6, [1, 0, 0, 0, 0, 0])
        p = pred(a, [.1, 0, 0, 0, 0, 0], [0] * 6)
        metrics = evaluate([a], [p], split='Test-Comp')['metrics']
        self.assertEqual(metrics['compositional_accuracy']['value'], 0.)
        self.assertEqual(metrics['negation_accuracy']['value'], 0.)
        self.assertIsNone(metrics['attribute_f1']['value'])
        self.assertIsNone(metrics['paraphrase_consistency']['value'])

    def test_id_coverage_cannot_silently_drop_or_duplicate(self):
        a = gold('expected', [.25, 0, 0, 0, 0, 0])
        p = pred(a, a['y'], a['modified_mask'])
        for predictions in ([], [p, p], [dict(p, id='unexpected')]):
            with self.subTest(predictions=predictions), self.assertRaises(ValueError):
                evaluate([a], predictions, split='Test-Comp')

    def test_nonfinite_prediction_and_wrong_split_rejected(self):
        a = gold('valid', [.25, 0, 0, 0, 0, 0])
        p = pred(a, a['y'], a['modified_mask'])
        for value in (float('nan'), float('inf'), 1.01):
            invalid = copy.deepcopy(p)
            invalid['y'][0] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                evaluate([a], [invalid], split='Test-Comp')
        with self.assertRaises(ValueError):
            evaluate([a], [dict(p, split='Train')], split='Test-Comp')


if __name__ == '__main__':
    unittest.main()
