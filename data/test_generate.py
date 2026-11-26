import copy
import json
from pathlib import Path
import unittest
from data import generate

class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((Path(__file__).parent/'specs'/'FROZEN_V1.json').read_text())
        cls.records, cls.pairs, cls.evidence = generate.generate(cls.spec)

    def test_release_invariants_and_counts(self):
        report = generate.audit(self.records, self.pairs, self.spec)
        self.assertEqual(report['status'], 'pass')
        self.assertEqual([len(self.records[s]) for s in ('Train','Val','Test-ID','Test-Comp')], [800,100,100,100])
        self.assertEqual(len(self.pairs), 1650)
        self.assertFalse(report['release_eligible'])

    def test_determinism(self):
        self.assertEqual(generate.generate(self.spec), (self.records,self.pairs,self.evidence))

    def test_manual_ast_ground_truth(self):
        ast = [{'attribute': 0,'operation':'edit','direction':-1,'magnitude':0.75},
               {'attribute':4,'operation':'preserve','direction':None,'magnitude':None}]
        self.assertEqual(generate.ast_labels(ast), ([-0.75,0,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,1,0]))

    def test_conflicts_and_invalid_ast_rejected(self):
        node = {'attribute':0,'operation':'edit','direction':1,'magnitude':0.25}
        with self.assertRaises(ValueError): generate.ast_labels([node,node])
        with self.assertRaises(ValueError): generate.ast_labels([{**node,'magnitude':2}])
        with self.assertRaises(ValueError): generate.ast_labels([{**node,'direction':0}])

    def test_corruption_detected(self):
        rows = copy.deepcopy(self.records)
        rows['Val'][0]['y'][0] = 7
        self.assertEqual(generate.audit(rows,self.pairs,self.spec)['status'],'fail')

    def test_cross_split_pair_detected(self):
        pairs = copy.deepcopy(self.pairs)
        pairs[0]['strong_id'] = self.records['Val'][0]['id']
        self.assertFalse(generate.audit(self.records,pairs,self.spec)['checks']['intensity_pairs'])

    def test_evidence_resolves(self):
        evidence = {e['review_id']: e for e in self.evidence}
        for pair in self.pairs:
            entry = evidence[pair['review_id']]
            for key in ('weak_id','strong_id','attribute_index','verification_status'):
                self.assertEqual(entry[key],pair[key])
            self.assertFalse(entry['human_review'])

if __name__ == '__main__': unittest.main()
