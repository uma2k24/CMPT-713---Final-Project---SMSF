"""Check AI challenge integrity and that it cannot manufacture verified metrics."""
import unittest
from pathlib import Path
from run_ai_validation import rows, load, norm, digest
from evaluation.metrics import evaluate
from models.model_b.data import load_release

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'experiments/ai_validation_v1'


class AIValidationTests(unittest.TestCase):
    def test_frozen_dataset_shapes_and_provenance(self):
        gold=rows(DEST/'challenge.jsonl')
        self.assertEqual(len(gold),150)
        self.assertEqual(len({norm(r['text']) for r in gold}),150)
        for row in gold:
            self.assertIsNone(row['human_review_id'])
            self.assertEqual(row['source'],'llm_synthetic')
            self.assertFalse(row['provenance']['independent_annotation'])
            self.assertEqual(row['modified_mask'],[int(v!=0) for v in row['y']])
            self.assertFalse(any(m and p for m,p in zip(row['modified_mask'],row['preserved_mask'])))
        plan=load(DEST/'plan.json')
        for name,expected in plan['input_sha256'].items():
            self.assertEqual(digest(DEST/name),expected)

    def test_unreviewed_pairs_cannot_become_official_consistency(self):
        gold=rows(DEST/'challenge.jsonl')
        pairs=rows(DEST/'challenge_pairs.jsonl')
        # Exact copies are deliberately synthetic predictions for an invariant test,
        # never reported as trained model results.
        predicted=[dict(id=r['id'],split=r['split'],y=r['y'],modified_mask=r['modified_mask']) for r in gold]
        result=evaluate(gold,predicted,split='Test-AI',paraphrase_pairs=pairs)
        self.assertIsNone(result['metrics']['paraphrase_consistency']['value'])
        self.assertEqual(result['metrics']['paraphrase_consistency']['exclusion_count'],75)
        self.assertIsNone(result['metrics']['compositional_accuracy']['value'])
        self.assertEqual(result['metrics']['negation_accuracy']['eligible_count'],112)
        self.assertEqual(result['metrics']['mae']['value'],0)
        with self.assertRaises(ValueError):
            load_release(DEST/'plan.json')

    def test_review_sample_keeps_human_evidence_absent(self):
        sample=rows(DEST/'ai_semantic_reviews.jsonl')
        self.assertEqual(len({r['paraphrase_id'] for r in sample}),150)
        self.assertEqual(sum(r['assessment']=='needs_human_attention' for r in sample),6)
        self.assertTrue(all(r['human_review_id'] is None and r['human_verified'] is False for r in sample))


if __name__=='__main__':
    unittest.main()
