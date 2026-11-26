"""Validate supplied human review CSVs and export linked evidence; never release G2."""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from data.assemble_paraphrases import ROOT, read_rows


def text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def verify_inputs(base_root=None, paraphrase_root=None):
    for directory, names in [(base_root or ROOT/'candidates/v1', ('train.jsonl', 'val.jsonl', 'test_id.jsonl', 'test_comp.jsonl')),
                             (paraphrase_root or ROOT/'paraphrases/v1', ('candidates.jsonl', 'paraphrase_pairs.jsonl'))]:
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text(encoding='utf-8-sig'))
        for name in names:
            actual = hashlib.sha256((directory/name).read_bytes()).hexdigest()
            if manifest.get('files_sha256', {}).get(name) != actual:
                raise ValueError(f'Immutable data hash mismatch: {directory/name}')


def validate_review(row, pair, base, candidate):
    required = {'pair_id': pair['pair_id'], 'base_id': base['id'], 'paraphrase_id': candidate['id'],
                'split': pair['split'], 'original_text': base['text'], 'paraphrase_text': candidate['text']}
    if any(row.get(k) != value for k, value in required.items()):
        raise ValueError('Review source/text metadata changed; corrected candidates need a new version and review')
    for key, label in [('proposed_y', 'y'), ('proposed_modified_mask', 'modified_mask'), ('proposed_preserved_mask', 'preserved_mask')]:
        if json.loads(row[key]) != base[label]:
            raise ValueError('Proposed labels changed in review sheet')
    outcome = row.get('outcome', '').strip()
    if not outcome:
        return [], None, 'pending'
    if outcome not in ('approved', 'rejected', 'needs_adjudication'):
        raise ValueError('Unknown review outcome')
    if not row.get('reviewer_id', '').strip() or not row.get('reviewed_at', '').strip():
        raise ValueError('Completed review requires actual reviewer identity and date/time')
    reviewed_at = datetime.fromisoformat(row['reviewed_at'].replace('Z', '+00:00'))
    if reviewed_at.tzinfo is None:
        raise ValueError('Review timestamp must include timezone')
    if outcome != 'approved':
        return [], None, outcome
    if row.get('meaning_preserved', '').strip().lower() != 'true':
        raise ValueError('Approved review requires explicit meaning_preserved=true')
    common = {'verification_status': 'human-verified', 'reviewer_id': row['reviewer_id'].strip(),
              'reviewed_at': row['reviewed_at'], 'outcome': 'approved', 'notes': row.get('notes', ''),
              'original_text_sha256': text_hash(base['text']), 'paraphrase_text_sha256': text_hash(candidate['text'])}
    row_review = dict(common, review_id='review-row-' + candidate['id'], endpoint_ids=[candidate['id']])
    pair_review = dict(common, review_id='review-pair-' + pair['pair_id'], pair_id=pair['pair_id'],
                       endpoint_ids=[base['id'], candidate['id']])
    reviewed_candidate = dict(candidate, review_status='human-verified', human_review_id=row_review['review_id'])
    reviewed_pair = dict(pair, verification_status='human-verified', human_review_id=pair_review['review_id'])
    return [row_review, pair_review], (reviewed_candidate, reviewed_pair), 'approved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reviews', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New directory for validated evidence; never overwrites')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Use a new output directory to retain review history')
    verify_inputs()
    bases = {r['id']: r for name in ('train', 'val', 'test_id', 'test_comp') for r in read_rows(ROOT/f'candidates/v1/{name}.jsonl')}
    candidates = {r['id']: r for r in read_rows(ROOT/'paraphrases/v1/candidates.jsonl')}
    pairs = {r['pair_id']: r for r in read_rows(ROOT/'paraphrases/v1/paraphrase_pairs.jsonl')}
    evidence, approved_rows, approved_pairs, decisions, seen = [], [], [], [], set()
    counts = {'pending': 0, 'approved': 0, 'rejected': 0, 'needs_adjudication': 0}
    for path in args.reviews:
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                key = row['pair_id']
                if key in seen or key not in pairs:
                    raise ValueError('Duplicate or unknown reviewed pair')
                seen.add(key)
                pair = pairs[key]
                records, approved, outcome = validate_review(row, pair, bases[pair['left_id']], candidates[pair['right_id']])
                counts[outcome] += 1
                evidence.extend(records); decisions.append(row)
                if approved:
                    approved_rows.append(approved[0]); approved_pairs.append(approved[1])
    args.output.mkdir(parents=True)
    for name, records in [('review_evidence', evidence), ('reviewed_candidates', approved_rows),
                           ('reviewed_pairs', approved_pairs), ('review_decisions', decisions)]:
        (args.output/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records), encoding='utf-8')
    summary = {'counts': counts, 'pairs_not_in_supplied_files': len(pairs)-len(seen),
               'g2_complete': False, 'release_status': 'not_released',
               'remaining': 'Test-Human authorship, dual annotations/adjudication and full release audit are separate requirements',
               'evidence_caveat': 'This validates recorded assertions; it cannot prove that a reviewer actually performed the review.'}
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
