import copy
import json
from pathlib import Path
import tempfile
import unittest
from .data import load_release, validate_rows, sha256
from .provisional_data import load_provisional

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiments/provisional_v1/data/manifest.json'


class ProvisionalDataTests(unittest.TestCase):
    def test_snapshot_counts_and_official_rejection(self):
        _, splits, pairs = load_provisional(MANIFEST, require_pairs=True)
        self.assertEqual([len(splits[s]) for s in ('Train', 'Val')], [1600, 200])
        self.assertEqual(sum(p['split'] == 'Train' for p in pairs), 800)
        with self.assertRaises(ValueError):
            load_release(MANIFEST)
        with self.assertRaises(ValueError):
            validate_rows(splits['Train'], 'Train')

    def test_provisional_does_not_allow_unreviewed_human_rows(self):
        _, splits, _ = load_provisional(MANIFEST)
        row = copy.deepcopy(next(r for r in splits['Train'] if r['source'] == 'llm_paraphrase'))
        row['source'] = 'human'
        with self.assertRaises(ValueError):
            validate_rows([row], 'Train', allow_generated=True)

    def test_tamper_and_pair_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)
            for path in MANIFEST.parent.iterdir():
                (target/path.name).write_bytes(path.read_bytes())
            pair_path = target/'pairs.jsonl'
            original = pair_path.read_text()
            rows = [json.loads(s) for s in original.splitlines()]
            rows[0]['right_id'] = rows[1]['right_id']
            pair_path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            with self.assertRaises(ValueError):
                load_provisional(target/'manifest.json')
            manifest = json.loads((target/'manifest.json').read_text())
            manifest['paraphrase_pairs']['sha256'] = sha256(pair_path)
            (target/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_provisional(target/'manifest.json')


if __name__ == '__main__':
    unittest.main()
