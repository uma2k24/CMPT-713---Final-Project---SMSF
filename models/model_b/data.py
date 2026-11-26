"""Fail-closed official release loading; candidate files cannot authorize fitting."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math

HUMAN_REVIEWED = {'human-verified', 'human-adjudicated'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validate_rows(rows, split, allow_generated=False):
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Records must be objects')
        if not isinstance(row.get('id'), str) or not row['id'] or row['id'] in seen:
            raise ValueError('Missing or duplicate example ID')
        seen.add(row['id'])
        if row.get('split') != split or row.get('schema_version') != '1.0':
            raise ValueError('Split/schema mismatch')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Missing text')
        if any(not isinstance(row.get(k), str) or not row[k] for k in ('semantic_group_id', 'paraphrase_group_id')):
            raise ValueError('Group identifiers required')
        y, mod, pre = (row.get(key, []) for key in ('y', 'modified_mask', 'preserved_mask'))
        if any(not isinstance(v, (list, tuple)) or len(v) != 6 for v in (y, mod, pre)):
            raise ValueError('Expected six-dimensional labels')
        for value, modified, preserved in zip(y, mod, pre):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or abs(value) > 1:
                raise ValueError('Invalid semantic value')
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or x not in (0, 1) for x in (modified, preserved)) or modified + preserved > 1:
                raise ValueError('Invalid masks')
            if bool(value) != bool(modified):
                raise ValueError('Modified mask must match nonzero edit')
        if row.get('source') in {'llm_paraphrase', 'human'}:
            if (allow_generated and row['source'] == 'llm_paraphrase'
                    and row.get('review_status') == 'generated' and row.get('human_review_id') is None):
                continue
            if row.get('review_status') not in HUMAN_REVIEWED or not row.get('human_review_id'):
                raise ValueError('Human/paraphrase row needs actual review evidence')
        elif row.get('source') != 'template':
            raise ValueError('Unsupported source')
    return rows


def group_weights(rows):
    counts = Counter(row['semantic_group_id'] for row in rows)
    return [len(rows) / (len(counts) * counts[row['semantic_group_id']]) for row in rows]


def _verified_file(root, entry):
    path = (root / entry['path']).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Release file must exist within release directory')
    if sha256(path) != entry['sha256']:
        raise ValueError('Release file hash mismatch')
    return path


def load_review_evidence(root, entries):
    """Resolve hash-verified evidence records, not merely evidence filenames."""
    evidence = {}
    for item in entries:
        path = _verified_file(root, item)
        records = read_jsonl(path) if path.suffix == '.jsonl' else json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(records, list):
            raise ValueError('Review evidence must be a JSON array or JSONL records')
        for record in records:
            if not isinstance(record, dict):
                raise ValueError('Review evidence record must be an object')
            key = record.get('review_id')
            if not isinstance(key, str) or not key or key in evidence:
                raise ValueError('Missing or duplicate evidence review_id')
            evidence[key] = record
    return evidence


def require_review(evidence, review_id, status, endpoint_ids, pair_id=None):
    record = evidence.get(review_id)
    if not isinstance(record, dict):
        raise ValueError('Missing linked human review evidence')
    actual = record.get('endpoint_ids', [])
    if (not isinstance(actual, list) or not all(isinstance(x, str) for x in actual)
            or set(actual) != set(endpoint_ids) or len(actual) != len(endpoint_ids)
            or record.get('verification_status') != status
            or status not in HUMAN_REVIEWED or record.get('outcome') != 'approved'
            or not isinstance(record.get('reviewer_id'), str) or not record['reviewer_id'].strip()
            or not isinstance(record.get('reviewed_at'), str) or not record['reviewed_at'].strip()
            or (pair_id is not None and record.get('pair_id') != pair_id)):
        raise ValueError('Human review evidence does not match approved endpoints/status/provenance')


def load_release(manifest_path, require_pairs=False):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    if not (manifest.get('release_status') == 'released' and manifest.get('g2_complete') is True
            and manifest.get('human_review_complete') is True and manifest.get('test_human_collected') is True):
        raise ValueError('G2 incomplete: candidate data cannot be used for official neural fitting')
    root = manifest_path.parent
    evidence = manifest.get('review_evidence', [])
    if not evidence:
        raise ValueError('G2 release requires review-evidence files')
    evidence_by_id = load_review_evidence(root, evidence)
    splits = {}
    for split in ('Train', 'Val'):
        splits[split] = validate_rows(read_jsonl(_verified_file(root, manifest['files'][split])), split)
        if not splits[split]:
            raise ValueError(f'{split} is empty')
        for row in splits[split]:
            if row['source'] in {'human', 'llm_paraphrase'}:
                require_review(evidence_by_id, row['human_review_id'], row['review_status'], [row['id']])
    for key in ('id', 'semantic_group_id', 'paraphrase_group_id'):
        if {r[key] for r in splits['Train']} & {r[key] for r in splits['Val']}:
            raise ValueError(f'Train/Val overlap: {key}')
    normalized = lambda text: ' '.join(text.lower().split())
    if {normalized(r['text']) for r in splits['Train']} & {normalized(r['text']) for r in splits['Val']}:
        raise ValueError('Train/Val duplicate text')
    pairs = []
    if manifest.get('paraphrase_pairs'):
        pairs = read_jsonl(_verified_file(root, manifest['paraphrase_pairs']))
    seen_pairs = set()
    by_id = {r['id']: r for rows in splits.values() for r in rows}
    for pair in pairs:
        if not isinstance(pair, dict):
            raise ValueError('Pair records must be objects')
        if pair.get('split') not in ('Train', 'Val'):
            raise ValueError('Training pair file may contain only Train/Val')
        if not pair.get('pair_id') or pair['pair_id'] in seen_pairs:
            raise ValueError('Duplicate or missing pair ID')
        seen_pairs.add(pair['pair_id'])
        if pair.get('verification_status') not in HUMAN_REVIEWED or not pair.get('human_review_id'):
            raise ValueError('Pair needs genuine human verification evidence')
        left, right = by_id.get(pair.get('left_id')), by_id.get(pair.get('right_id'))
        if left is None or right is None or left['id'] == right['id']:
            raise ValueError('Pair endpoints missing or identical')
        for key in ('split', 'semantic_group_id', 'y', 'modified_mask', 'preserved_mask'):
            if left[key] != right[key]:
                raise ValueError(f'Pair is not equivalent: {key}')
        if left['split'] != pair['split'] or left['semantic_group_id'] != pair['semantic_group_id']:
            raise ValueError('Pair metadata mismatch')
        require_review(evidence_by_id, pair['human_review_id'], pair['verification_status'],
                       [left['id'], right['id']], pair['pair_id'])
    if require_pairs and not any(p['split'] == 'Train' for p in pairs):
        raise ValueError('Model C requires human-verified Train paraphrase pairs')
    return manifest, splits, pairs
