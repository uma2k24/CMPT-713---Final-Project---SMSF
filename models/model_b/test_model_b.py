"""Offline implementation checks using random tiny T5; never experimental results."""
import copy
import json
import math
import tempfile
import unittest
from pathlib import Path
import torch
from transformers import T5Config, T5EncoderModel
from models.model_b.data import group_weights, load_release, sha256, validate_rows, require_review
from models.model_b.model import GroundingModel, load_checkpoint, masked_mean, save_checkpoint, supervised_loss
from models.model_b.train import select_threshold


def fixture(identifier='a', split='Train', group='g1'):
    return {'id': identifier, 'text': 'raise the eyebrows slightly', 'y': [.25, 0, 0, 0, 0, 0],
            'modified_mask': [1, 0, 0, 0, 0, 0], 'preserved_mask': [0] * 6,
            'semantic_group_id': group, 'paraphrase_group_id': group,
            'split': split, 'schema_version': '1.0', 'source': 'template', 'review_status': 'generated'}


class ModelChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        torch.set_num_threads(1)
        config = T5Config(vocab_size=32, d_model=16, d_kv=8, d_ff=32, num_layers=1,
                          num_heads=2, dropout_rate=0., pad_token_id=0, eos_token_id=1)
        self.model = GroundingModel(T5EncoderModel(config))

    def test_masked_pool_excludes_padding_and_rejects_empty(self):
        states = torch.tensor([[[1., 3.], [3., 5.], [100., 200.]]])
        torch.testing.assert_close(masked_mean(states, torch.tensor([[1, 1, 0]])), torch.tensor([[2., 4.]]))
        with self.assertRaises(ValueError):
            masked_mean(states, torch.zeros(1, 3))

    def test_encoder_predictions_padding_invariant_and_bounded(self):
        self.model.eval()
        short = self.model(torch.tensor([[2, 3, 1]]), torch.tensor([[1, 1, 1]]))
        padded = self.model(torch.tensor([[2, 3, 1, 0, 0]]), torch.tensor([[1, 1, 1, 0, 0]]))
        for key in ('y', 'logits', 'probabilities'):
            self.assertEqual(short[key].shape, (1, 6))
            torch.testing.assert_close(short[key], padded[key], atol=1e-6, rtol=1e-5)
        self.assertTrue((short['y'].abs() <= 1).all())
        self.assertTrue(((short['probabilities'] >= 0) & (short['probabilities'] <= 1)).all())

    def test_loss_scaling_and_gradient_paths(self):
        zeros = torch.zeros(2, 6)
        loss = supervised_loss(zeros, torch.ones(2, 6), zeros, zeros)
        self.assertAlmostEqual(float(loss), math.log(2) + 6, places=5)
        out = self.model(torch.tensor([[2, 3, 1], [5, 4, 1]]), torch.ones(2, 3, dtype=torch.long))
        supervised_loss(out['logits'], out['y'], zeros, zeros).backward()
        for module in (self.model.encoder, self.model.classifier, self.model.regressor):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def test_group_weights_have_equal_total_per_group(self):
        rows = [fixture(), fixture('b'), fixture('c', group='g2')]
        weights = group_weights(rows)
        self.assertAlmostEqual(sum(weights) / 3, 1)
        self.assertAlmostEqual(weights[0] + weights[1], weights[2])
        zeros = torch.zeros(3, 6)
        actual = supervised_loss(zeros, zeros, zeros, zeros, torch.tensor(weights))
        self.assertAlmostEqual(float(actual), math.log(2), places=6)

    def test_checkpoint_roundtrip(self):
        self.model.eval()
        ids, mask = torch.tensor([[2, 3, 1]]), torch.ones(1, 3, dtype=torch.long)
        expected = self.model(ids, mask)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            save_checkpoint(tmp, self.model, {'model_version': 'tiny-random-test-only'})
            loaded, metadata = load_checkpoint(tmp)
            actual = loaded(ids, mask)
            self.assertEqual(metadata['model_version'], 'tiny-random-test-only')
            for key in expected:
                torch.testing.assert_close(expected[key], actual[key], atol=0, rtol=0)

    def test_threshold_selection_and_tie(self):
        threshold, score = select_threshold(torch.tensor([[.8, .1, .1, .1, .1, .1]]), torch.tensor([[1, 0, 0, 0, 0, 0]]))
        self.assertEqual(threshold, .5)
        self.assertEqual(score, 1.)

    def test_record_validation_rejects_wrong_split_and_review(self):
        with self.assertRaises(ValueError):
            validate_rows([fixture(split='Test-ID')], 'Train')
        row = fixture()
        row['source'] = 'llm_paraphrase'
        with self.assertRaises(ValueError):
            validate_rows([row], 'Train')
        row = fixture()
        row['preserved_mask'][0] = 1
        with self.assertRaises(ValueError):
            validate_rows([row], 'Train')

    def test_candidate_release_gate(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path = Path(tmp) / 'manifest.json'
            path.write_text(json.dumps({'release_status': 'candidate', 'g2_complete': False}), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'G2 incomplete'):
                load_release(path)

    def test_release_hash_and_overlap_guards(self):
        # Artificial evidence fixture tests plumbing; it is not a real review record.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            train, val = fixture(), fixture('v', 'Val', 'g2')
            val['text'] = 'slightly raise the eyebrows'
            evidence = root / 'evidence.json'
            evidence.write_text('[{"review_id": "fixture-only-unrelated-evidence", "fixture_only": true}]', encoding='utf-8')
            manifest = {'release_status': 'released', 'g2_complete': True, 'human_review_complete': True,
                        'test_human_collected': True, 'files': {},
                        'review_evidence': [{'path': evidence.name, 'sha256': sha256(evidence)}]}
            for split, row in [('Train', train), ('Val', val)]:
                path = root / f'{split}.jsonl'
                path.write_text(json.dumps(row) + '\n', encoding='utf-8')
                manifest['files'][split] = {'path': path.name, 'sha256': sha256(path)}
            path = root / 'manifest.json'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            _, splits, pairs = load_release(path)
            self.assertEqual(len(splits['Train']), 1)
            with self.assertRaisesRegex(ValueError, 'requires human-verified'):
                load_release(path, require_pairs=True)
            manifest['files']['Train']['sha256'] = 'invalid'
            path.write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                load_release(path)

    def test_human_review_must_resolve_exact_evidence(self):
        record = {'review_id': 'r', 'endpoint_ids': ['a', 'b'], 'pair_id': 'p',
                  'verification_status': 'human-verified', 'reviewer_id': 'fixture-human',
                  'reviewed_at': '2026-11-23', 'outcome': 'approved', 'fixture_only': True}
        require_review({'r': record}, 'r', 'human-verified', ['a', 'b'], 'p')
        with self.assertRaisesRegex(ValueError, 'Missing linked'):
            require_review({'unrelated': record}, 'r', 'human-verified', ['a', 'b'], 'p')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            require_review({'r': record}, 'r', 'human-verified', ['a', 'c'], 'p')
        bad = dict(record, reviewer_id='')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            require_review({'r': bad}, 'r', 'human-verified', ['a', 'b'], 'p')

    def test_malformed_labels_rejected_with_valueerror(self):
        for key, value in [('y', None), ('y', [True, 0, 0, 0, 0, 0]),
                           ('y', [float('nan'), 0, 0, 0, 0, 0]),
                           ('modified_mask', [True, 0, 0, 0, 0, 0])]:
            row = fixture()
            row[key] = value
            with self.assertRaises(ValueError):
                validate_rows([row], 'Train')


if __name__ == '__main__':
    unittest.main(verbosity=2)
