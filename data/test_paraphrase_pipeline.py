"""Independent provenance tests. All named reviewers here are fictional test fixtures."""
import copy
import contextlib
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from data import assemble_paraphrases
from data.assemble_paraphrases import validate_candidates
from data.import_paraphrase_reviews import validate_review, verify_inputs
from models.model_b.data import require_review, validate_rows


def fixture():
    base = {'id': 'test-only-base', 'text': 'raise the eyebrows slightly.', 'split': 'Train',
            'schema_version': '1.0', 'source': 'template', 'review_status': 'generated',
            'semantic_group_id': 'test-semantic', 'paraphrase_group_id': 'test-paraphrase',
            'y': [.25, 0, 0, 0, 0, 0], 'modified_mask': [1, 0, 0, 0, 0, 0], 'preserved_mask': [0]*6}
    candidate = dict(base, id='test-only-candidate', text='Lift the eyebrows a little.', source='llm_paraphrase', human_review_id=None)
    pair = {'pair_id': 'test-only-pair', 'split': 'Train', 'left_id': base['id'], 'right_id': candidate['id'],
            'semantic_group_id': base['semantic_group_id'], 'verification_status': 'generated', 'human_review_id': None}
    row = {'pair_id': pair['pair_id'], 'base_id': base['id'], 'paraphrase_id': candidate['id'], 'split': 'Train',
           'original_text': base['text'], 'paraphrase_text': candidate['text'], 'proposed_y': json.dumps(base['y']),
           'proposed_modified_mask': json.dumps(base['modified_mask']), 'proposed_preserved_mask': json.dumps(base['preserved_mask']),
           'reviewer_id': '', 'reviewed_at': '', 'outcome': '', 'meaning_preserved': '', 'notes': ''}
    return base, candidate, pair, row


class ParaphrasePipelineChecks(unittest.TestCase):
    def test_immutable_hash_check_allows_review_edits_but_rejects_data_edits(self):
        with tempfile.TemporaryDirectory(prefix='paraphrase_hash_fixture_') as temporary:
            base_dir, para_dir = Path(temporary)/'base', Path(temporary)/'para'
            for directory, names in [(base_dir, ('train.jsonl', 'val.jsonl', 'test_id.jsonl', 'test_comp.jsonl')),
                                     (para_dir, ('candidates.jsonl', 'paraphrase_pairs.jsonl'))]:
                directory.mkdir()
                hashes = {}
                for name in names:
                    path = directory/name; path.write_text('[]\n', encoding='utf-8')
                    hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
                (directory/'manifest.json').write_text(json.dumps({'files_sha256': hashes}), encoding='utf-8')
            (para_dir/'review_train.csv').write_text('Reviewer-edit-is-allowed\n', encoding='utf-8')
            verify_inputs(base_dir, para_dir)
            (para_dir/'candidates.jsonl').write_text('changed data\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                verify_inputs(base_dir, para_dir)

    def test_any_review_sheet_edit_is_preserved_on_reassembly(self):
        base, candidate, _, _ = fixture()
        base = dict(base, composition_key='0')
        with tempfile.TemporaryDirectory(prefix='paraphrase_review_fixture_') as temporary:
            directory = Path(temporary)
            (directory/'candidates/v1').mkdir(parents=True)
            (directory/'paraphrases/shards').mkdir(parents=True)
            for name in ('train', 'val', 'test_id', 'test_comp'):
                (directory/f'candidates/v1/{name}.jsonl').write_text(json.dumps(base)+'\n' if name == 'train' else '', encoding='utf-8')
            for name in ('train_a', 'train_b', 'heldout'):
                record = {'base_id': base['id'], 'text': candidate['text']}
                (directory/f'paraphrases/shards/{name}_generated.jsonl').write_text(json.dumps(record)+'\n' if name == 'train_a' else '', encoding='utf-8')
            with patch.object(assemble_paraphrases, 'ROOT', directory), contextlib.redirect_stdout(io.StringIO()):
                assemble_paraphrases.main()
                review_path = directory/'paraphrases/v1/review_train.csv'
                data_path = directory/'paraphrases/v1/candidates.jsonl'
                original_review, original_data = review_path.read_bytes(), data_path.read_bytes()
                for field in ('meaning_preserved', 'reviewed_at', 'paraphrase_text', 'proposed_y'):
                    review_path.write_bytes(original_review)
                    with review_path.open(encoding='utf-8-sig', newline='') as stream:
                        reader = csv.DictReader(stream); names = reader.fieldnames; review_rows = list(reader)
                    review_rows[0][field] = 'fictional-edit-test-only'
                    with review_path.open('w', encoding='utf-8-sig', newline='') as stream:
                        writer = csv.DictWriter(stream, fieldnames=names); writer.writeheader(); writer.writerows(review_rows)
                    edited = review_path.read_bytes()
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        assemble_paraphrases.main()
                    self.assertEqual(review_path.read_bytes(), edited)
                    self.assertEqual(data_path.read_bytes(), original_data)

    def test_complete_distinct_candidate_preserves_source_map(self):
        base, candidate, _, _ = fixture()
        result = validate_candidates([base], [{'base_id': base['id'], 'text': candidate['text']}])
        self.assertIs(result[base['id']], base)

    def test_incomplete_unknown_duplicate_unchanged_and_empty_rejected(self):
        base, candidate, _, _ = fixture()
        valid = {'base_id': base['id'], 'text': candidate['text']}
        for records in ([], [dict(valid, base_id='unknown')], [valid, valid],
                        [dict(valid, text=base['text'].upper())], [dict(valid, text='  ')]):
            with self.subTest(records=records), self.assertRaises(ValueError):
                validate_candidates([base], records)

    def test_cross_meaning_and_split_collision_rejected(self):
        base, candidate, _, _ = fixture()
        second = dict(base, id='test-base2', text='Lower the eyebrows slightly.',
                      split='Val', semantic_group_id='different-semantic', paraphrase_group_id='different-para')
        records = [{'base_id': base['id'], 'text': candidate['text']},
                   {'base_id': second['id'], 'text': candidate['text'].upper()}]
        with self.assertRaises(ValueError):
            validate_candidates([base, second], records)

    def test_blank_and_rejected_reviews_create_no_approval(self):
        base, candidate, pair, row = fixture()
        self.assertEqual(validate_review(row, pair, base, candidate), ([], None, 'pending'))
        rejected = dict(row, outcome='rejected', reviewer_id='fictional-test-fixture-only',
                        reviewed_at='2026-11-23T12:00:00-08:00')
        self.assertEqual(validate_review(rejected, pair, base, candidate), ([], None, 'rejected'))
        with self.assertRaises(ValueError):
            validate_rows([candidate], 'Train')

    def test_approval_requires_exact_record_and_provenance(self):
        base, candidate, pair, row = fixture()
        approved = dict(row, outcome='approved', reviewer_id='fictional-test-fixture-only',
                        reviewed_at='2026-11-23T12:00:00-08:00', meaning_preserved='true')
        changes = [('reviewer_id', ''), ('reviewed_at', ''), ('reviewed_at', 'not-a-date'),
                   ('reviewed_at', '2026-11-23T12:00:00'), ('meaning_preserved', 'false'),
                   ('base_id', 'unlinked'), ('paraphrase_id', 'unlinked'), ('split', 'Val'),
                   ('original_text', 'changed'), ('paraphrase_text', 'changed'),
                   ('proposed_y', '[0,0,0,0,0,0]'), ('outcome', 'maybe')]
        for key, value in changes:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_review(dict(approved, **{key: value}), pair, base, candidate)

    def test_test_only_approved_evidence_matches_training_loader(self):
        base, candidate, pair, row = fixture()
        approved = dict(row, outcome='approved', reviewer_id='fictional-test-fixture-only',
                        reviewed_at='2026-11-23T12:00:00-08:00', meaning_preserved='true')
        evidence, reviewed, status = validate_review(approved, pair, base, candidate)
        self.assertEqual(status, 'approved')
        self.assertEqual(len(evidence), 2)
        reviewed_candidate, reviewed_pair = reviewed
        indexed = {r['review_id']: r for r in evidence}
        validate_rows([base, reviewed_candidate], 'Train')
        require_review(indexed, reviewed_candidate['human_review_id'], 'human-verified', [candidate['id']])
        require_review(indexed, reviewed_pair['human_review_id'], 'human-verified', [base['id'], candidate['id']], pair['pair_id'])
        self.assertEqual(candidate['review_status'], 'generated')
        self.assertEqual(pair['verification_status'], 'generated')
        tampered = copy.deepcopy(indexed)
        tampered[reviewed_pair['human_review_id']]['endpoint_ids'] = ['unrelated', candidate['id']]
        with self.assertRaises(ValueError):
            require_review(tampered, reviewed_pair['human_review_id'], 'human-verified', [base['id'], candidate['id']], pair['pair_id'])


if __name__ == '__main__':
    unittest.main()
