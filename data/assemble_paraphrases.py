"""Validate and package actual assistant-authored candidates for real human review."""
import csv
import hashlib
import json
from pathlib import Path
import re
from collections import Counter

ROOT = Path(__file__).resolve().parent


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ' '.join(re.findall(r"[a-z0-9']+", text.casefold()))


def validate_candidates(base_rows, generated):
    bases = {r['id']: r for r in base_rows}
    if len(bases) != len(base_rows):
        raise ValueError('Duplicate base IDs')
    seen = set()
    for row in generated:
        key = row.get('base_id')
        if key not in bases or key in seen:
            raise ValueError(f'Duplicate/unknown paraphrase source: {key}')
        seen.add(key)
        text = row.get('text')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'Missing candidate text: {key}')
        if normalized(text) == normalized(bases[key]['text']):
            raise ValueError(f'Candidate repeats original verbatim: {key}')
    if seen != set(bases):
        raise ValueError(f'Incomplete paraphrases: {len(seen)} of {len(bases)}')
    # Exact/normalized collisions across different intended meanings cannot be released.
    text_to_group = {}
    for row in base_rows + [dict(bases[r['base_id']], text=r['text']) for r in generated]:
        text = normalized(row['text'])
        signature = (row['semantic_group_id'], row['split'])
        if text in text_to_group and text_to_group[text] != signature:
            raise ValueError(f'Cross-meaning/split text collision: {row["id"]}')
        text_to_group[text] = signature
    return bases


def main():
    base_rows = []
    for name in ('train', 'val', 'test_id', 'test_comp'):
        base_rows += read_rows(ROOT / f'candidates/v1/{name}.jsonl')
    generated, source_files = [], []
    for name in ('train_a', 'train_b', 'heldout'):
        path = ROOT / f'paraphrases/shards/{name}_generated.jsonl'
        generated += [dict(r, authoring_agent=name) for r in read_rows(path)]
        source_files.append({'path': str(path.relative_to(ROOT)), 'sha256': fingerprint(path)})
    bases = validate_candidates(base_rows, generated)
    destination = ROOT / 'paraphrases/v1'
    destination.mkdir(parents=True, exist_ok=True)
    prior_manifest = json.loads((destination/'manifest.json').read_text(encoding='utf-8')) if (destination/'manifest.json').exists() else {}
    for path in destination.glob('review_*.csv'):
        if prior_manifest.get('files_sha256', {}).get(path.name) != fingerprint(path):
            raise ValueError(f'Existing review sheet changed or lacks original hash: {path}; refusing to erase review work')
        with path.open(encoding='utf-8-sig', newline='') as stream:
            prior = list(csv.DictReader(stream))
        if any(r.get('reviewer_id') or r.get('outcome') or r.get('notes') for r in prior):
            raise ValueError(f'Existing review work present: {path}; preserve it before reassembly')
    output_rows, pairs, review_rows = [], [], []
    for authored in generated:
        base = bases[authored['base_id']]
        pid = 'llm-' + base['id'][5:]
        pair_id = 'pp-' + base['id'][5:]
        record = {k: base[k] for k in ('y', 'modified_mask', 'preserved_mask', 'semantic_group_id',
                                      'paraphrase_group_id', 'composition_key', 'split', 'schema_version')}
        record.update(id=pid, text=authored['text'].strip(), source='llm_paraphrase', template_family_id=None,
                      review_status='generated', human_review_id=None,
                      provenance={'base_id': base['id'], 'generator': 'Codex assistant direct LLM authoring',
                                  'model_version': 'not_exposed_by_session', 'authoring_agent': authored['authoring_agent'],
                                  'generation_date': '2026-11-23', 'method': 'individually_authored_output_strings',
                                  'label_status': 'inherited_proposed_labels_pending_real_meaning_review',
                                  'human_review': None})
        output_rows.append(record)
        pairs.append({'pair_id': pair_id, 'split': base['split'], 'left_id': base['id'], 'right_id': pid,
                      'semantic_group_id': base['semantic_group_id'], 'verification_status': 'generated',
                      'human_review_id': None})
        review_rows.append({'pair_id': pair_id, 'base_id': base['id'], 'paraphrase_id': pid,
                            'split': base['split'], 'original_text': base['text'], 'paraphrase_text': record['text'],
                            'proposed_y': json.dumps(base['y']), 'proposed_modified_mask': json.dumps(base['modified_mask']),
                            'proposed_preserved_mask': json.dumps(base['preserved_mask']),
                            'reviewer_id': '', 'reviewed_at': '', 'outcome': '', 'meaning_preserved': '', 'notes': ''})
    for name, records in [('candidates.jsonl', output_rows), ('paraphrase_pairs.jsonl', pairs)]:
        (destination / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
    for split in ('Train', 'Val', 'Test-ID', 'Test-Comp'):
        path = destination / f'review_{split.lower().replace("-", "_")}.csv'
        # Filled reviews are never overwritten by rerunning assembly.
        if path.exists():
            with path.open(encoding='utf-8-sig', newline='') as stream:
                prior = list(csv.DictReader(stream))
            if any(r.get('reviewer_id') or r.get('outcome') or r.get('notes') for r in prior):
                raise ValueError(f'Existing review work present: {path}; preserve it before reassembly')
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(review_rows[0]))
            writer.writeheader(); writer.writerows(r for r in review_rows if r['split'] == split)
    summary = {'data_version': 'paraphrase-candidates-v1', 'schema_version': '1.0',
               'candidate_count': len(output_rows), 'split_counts': dict(Counter(r['split'] for r in output_rows)),
               'pair_count': len(pairs), 'human_verified_count': 0, 'release_eligible': False,
               'structural_audit': 'pass', 'meaning_equivalence': 'unverified_pending_human_review',
               'checks': ['exactly_one_candidate_per_base', 'ids_complete_unique', 'nonempty_changed_text',
                          'no_cross_meaning_or_split_normalized_text_collisions', 'inherited_groups_and_splits'],
               'source_files': source_files, 'files_sha256': {p.name: fingerprint(p) for p in sorted(destination.iterdir()) if p.name != 'manifest.json'},
               'notes': 'Structural checks do not certify semantic equivalence. No paraphrase enters official training/evaluation before real review.'}
    (destination/'manifest.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('candidate_count', 'split_counts', 'pair_count', 'human_verified_count', 'release_eligible')}, indent=2))


if __name__ == '__main__':
    main()
