"""Explicitly unreleased generated-data experiments; no human review is asserted."""
import json
from pathlib import Path
from .data import _verified_file, read_jsonl, validate_rows


def load_provisional(manifest_path, require_pairs=False):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('release_status') != 'unreleased_provisional'
            or any(manifest.get(k) is not False for k in
                   ('g2_complete', 'human_review_complete', 'test_human_collected'))):
        raise ValueError('Expected explicit unreleased, unreviewed provisional manifest')
    splits = {s: validate_rows(read_jsonl(_verified_file(path.parent, manifest['files'][s])),
                              s, allow_generated=True) for s in ('Train', 'Val')}
    if not all(splits.values()):
        raise ValueError('Empty training/validation split')
    for key in ('id', 'semantic_group_id', 'paraphrase_group_id'):
        if {r[key] for r in splits['Train']} & {r[key] for r in splits['Val']}:
            raise ValueError('Train/Val group leakage')
    norm = lambda t: ' '.join(t.lower().split())
    if {norm(r['text']) for r in splits['Train']} & {norm(r['text']) for r in splits['Val']}:
        raise ValueError('Train/Val text leakage')
    pairs = read_jsonl(_verified_file(path.parent, manifest['paraphrase_pairs']))
    by_id = {r['id']: r for rows in splits.values() for r in rows}
    seen = set()
    for p in pairs:
        if not p.get('pair_id') or p['pair_id'] in seen:
            raise ValueError('Missing/duplicate pair ID')
        seen.add(p['pair_id'])
        if p.get('verification_status') != 'generated' or p.get('human_review_id') is not None:
            raise ValueError('Provisional pairs must retain generated status')
        left, right = by_id.get(p.get('left_id')), by_id.get(p.get('right_id'))
        if not left or not right or left['id'] == right['id']:
            raise ValueError('Invalid pair endpoints')
        for key in ('split', 'semantic_group_id', 'paraphrase_group_id', 'y', 'modified_mask', 'preserved_mask'):
            if left[key] != right[key]:
                raise ValueError('Pair label/group mismatch')
        if p['split'] != left['split'] or p['semantic_group_id'] != left['semantic_group_id']:
            raise ValueError('Pair metadata mismatch')
    if require_pairs and not any(p['split'] == 'Train' for p in pairs):
        raise ValueError('Model C requires Train pairs')
    return manifest, splits, pairs
