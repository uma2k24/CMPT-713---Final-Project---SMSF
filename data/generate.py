"""Deterministic candidate generation. No human reviews or released gold are invented."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NAMES = ('eyebrows', 'eyes', 'mouth', 'smile', 'jaw', 'cheeks')
VERBS = (('lower', 'raise'), ('close', 'open'), ('narrow', 'widen'),
         ('reduce', 'strengthen'), ('close', 'open'), ('lower', 'raise'))
MODIFIERS = {0.25: 'slightly', 0.5: 'moderately', 0.75: 'much', 1.0: 'maximally'}
FILES = {'Train': 'train', 'Val': 'val', 'Test-ID': 'test_id', 'Test-Comp': 'test_comp'}

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def ast_labels(ast):
    y, modified, preserved = [0.0]*6, [0]*6, [0]*6
    seen = set()
    for node in ast:
        i = node['attribute']
        if type(i) is not int or not 0 <= i < 6 or i in seen:
            raise ValueError('Invalid or repeated AST attribute')
        seen.add(i)
        if node['operation'] == 'preserve':
            if node['direction'] is not None or node['magnitude'] is not None:
                raise ValueError('Preservation cannot carry direction/intensity')
            preserved[i] = 1
        elif node['operation'] == 'edit':
            if node['direction'] not in (-1, 1) or node['magnitude'] not in MODIFIERS:
                raise ValueError('Unsupported direction/intensity')
            y[i] = node['direction'] * node['magnitude']
            modified[i] = 1
        else:
            raise ValueError('Unsupported operation')
    return y, modified, preserved

def render(ast, family):
    clauses = []
    for node in ast:
        i = node['attribute']
        if node['operation'] == 'preserve':
            clause = (f'keep the {NAMES[i]} unchanged' if family != 1
                      else f'do not change the {NAMES[i]}')
        else:
            verb = VERBS[i][node['direction'] == 1]
            modifier = MODIFIERS[node['magnitude']]
            clause = f'{verb} the {NAMES[i]} {modifier}'
            # Prefix "much" is ungrammatical; use its frozen alias in prefix form.
            if family == 1 and modifier != 'much':
                clause = f'{modifier} {verb} the {NAMES[i]}'
            if family == 2 and node['magnitude'] == 0.5:
                clause = f'{verb} the {NAMES[i]}'
        clauses.append(clause)
    return ('please ' if family == 2 else '') + ' and '.join(clauses) + '.'

def blocks_for(compositions, spec):
    """A block varies the first modified attribute; blocks have disjoint semantics."""
    blocks = []
    for comp in compositions:
        for signs in itertools.product((-1, 1), repeat=len(comp)):
            for fixed in itertools.product(spec['magnitudes'], repeat=len(comp)-1):
                for preserve in [None] + [i for i in range(6) if i not in comp]:
                    blocks.append((tuple(comp), signs, fixed, preserve))
    return blocks

def make_rows(block, split, spec):
    comp, signs, fixed, preserve = block
    block_id = digest(block)[:20]
    family = int(block_id[:8], 16) % 3
    rows = []
    for magnitude in spec['magnitudes']:
        ast = [{'attribute': i, 'operation': 'edit', 'direction': sign, 'magnitude': mag}
               for i, sign, mag in zip(comp, signs, (magnitude,) + fixed)]
        if preserve is not None:
            ast.append({'attribute': preserve, 'operation': 'preserve', 'direction': None, 'magnitude': None})
        ast.sort(key=lambda x: x['attribute'])
        y, modified, preserved = ast_labels(ast)
        semantic = digest(ast)[:24]
        rows.append({'id': 'base-' + semantic, 'text': render(ast, family), 'y': y,
                     'modified_mask': modified, 'preserved_mask': preserved,
                     'semantic_group_id': 'sem-' + semantic, 'paraphrase_group_id': 'para-' + semantic,
                     'template_family_id': f'canonical-{family}',
                     'composition_key': '+'.join(map(str, comp)),
                     'preservation_key': '' if preserve is None else str(preserve),
                     'source': 'template', 'split': split, 'schema_version': spec['schema_version'],
                     'review_status': 'generated', 'ast': ast,
                     'provenance': {'generator': 'data/generate.py', 'version': '1.0', 'seed': spec['seed'],
                                    'intensity_block_id': block_id, 'ast_sha256': digest(ast),
                                    'label_method': 'deterministic_ast', 'human_review': None}})
    return rows

def generate(spec):
    holdouts = [set(x) for x in spec['holdout_modified_sets']]
    allowed = [c for k in (1, 2, 3) for c in itertools.combinations(range(6), k)
               if not any(h <= set(c) for h in holdouts)]
    pool = blocks_for(allowed, spec)
    rng = random.Random(spec['seed'])
    rng.shuffle(pool)
    # Predeclared coverage: all single-attribute directions and every allowed composition.
    forced = [((i,), (s,), (), None) for i in range(6) for s in (-1, 1)]
    for comp in allowed:
        if not any(b[0] == comp for b in forced):
            forced.append(next(b for b in pool if b[0] == comp))
    forced_set = set(forced)
    pool = forced + [b for b in pool if b not in forced_set]
    assigned = {}
    offset = 0
    for split in ('Train', 'Val', 'Test-ID'):
        count = spec['base_counts'][split] // 4
        assigned[split] = pool[offset:offset+count]
        offset += count
    held = blocks_for([tuple(h) for h in spec['holdout_modified_sets']], spec)
    rng.shuffle(held)
    # Ensure every heldout composition appears, before random fill.
    forced_held = [next(b for b in held if b[0] == tuple(h)) for h in spec['holdout_modified_sets']]
    assigned['Test-Comp'] = (forced_held + [b for b in held if b not in forced_held])[:25]
    records, pairs, evidence = {}, [], []
    for split, blocks in assigned.items():
        records[split] = []
        for block in blocks:
            rows = make_rows(block, split, spec)
            records[split].extend(rows)
            for weak, strong in itertools.combinations(rows, 2):
                pair_id = 'int-' + digest([weak['id'], strong['id']])[:24]
                review_id = 'ast-' + pair_id
                pairs.append({'pair_id': pair_id, 'split': split, 'weak_id': weak['id'],
                              'strong_id': strong['id'], 'attribute_index': block[0][0],
                              'verification_status': 'ast-verified', 'other_controls_equal': True,
                              'review_id': review_id})
                evidence.append({'review_id': review_id, 'method': 'exact_ast_comparison',
                                 'verification_status': 'ast-verified', 'weak_id': weak['id'],
                                 'strong_id': strong['id'], 'attribute_index': block[0][0],
                                 'generator': 'data/generate.py', 'generator_version': '1.0',
                                 'weak_ast_sha256': weak['provenance']['ast_sha256'],
                                 'strong_ast_sha256': strong['provenance']['ast_sha256'],
                                 'human_review': False})
    return records, pairs, evidence

def audit(records, pairs, spec):
    all_rows = [r for rows in records.values() for r in rows]
    checks = {}
    checks['counts'] = all(len(records[s]) == n for s, n in spec['base_counts'].items())
    checks['unique_ids'] = len({r['id'] for r in all_rows}) == len(all_rows)
    checks['unique_semantic_groups'] = len({r['semantic_group_id'] for r in all_rows}) == len(all_rows)
    checks['unique_paraphrase_groups'] = len({r['paraphrase_group_id'] for r in all_rows}) == len(all_rows)
    normalized = [' '.join(r['text'].lower().split()).rstrip('.') for r in all_rows]
    checks['unique_normalized_text'] = len(set(normalized)) == len(normalized)
    checks['labels_from_ast'] = all(ast_labels(r['ast']) == (r['y'], r['modified_mask'], r['preserved_mask']) for r in all_rows)
    checks['honest_review_status'] = all(r['review_status'] == 'generated' and r['source'] == 'template' for r in all_rows)
    holdouts = [set(h) for h in spec['holdout_modified_sets']]
    checks['heldout_containment'] = all(not any(h <= {i for i, m in enumerate(r['modified_mask']) if m} for h in holdouts)
                                           for s in ('Train', 'Val', 'Test-ID') for r in records[s])
    checks['test_comp_exact_holdouts'] = all({i for i, m in enumerate(r['modified_mask']) if m} in holdouts for r in records['Test-Comp'])
    train_comps = {r['composition_key'] for r in records['Train']}
    checks['id_val_compositions_in_train'] = all(r['composition_key'] in train_comps for s in ('Val', 'Test-ID') for r in records[s])
    primitives = {(i, r['y'][i]) for r in records['Train'] for i in range(6) if r['modified_mask'][i]}
    checks['train_all_primitives'] = all((i, s*m) in primitives for i in range(6) for s in (-1, 1) for m in spec['magnitudes'])
    by_id = {r['id']: r for r in all_rows}
    valid_pairs = True
    for p in pairs:
        a, b, j = by_id[p['weak_id']], by_id[p['strong_id']], p['attribute_index']
        valid_pairs &= a['split'] == b['split'] == p['split']
        valid_pairs &= a['modified_mask'] == b['modified_mask'] and a['preserved_mask'] == b['preserved_mask']
        valid_pairs &= 0 < abs(a['y'][j]) < abs(b['y'][j]) and a['y'][j]*b['y'][j] > 0
        valid_pairs &= all(a['y'][i] == b['y'][i] for i in range(6) if i != j)
    checks['intensity_pairs'] = bool(valid_pairs)
    family_sets = {s: {r['template_family_id'] for r in rows} for s, rows in records.items()}
    overlap = {a+' / '+b: sorted(family_sets[a] & family_sets[b]) for a, b in itertools.combinations(records, 2)}
    coverage = {s: {'composition_counts': {c: sum(r['composition_key'] == c for r in rows) for c in sorted({r['composition_key'] for r in rows})},
                    'explicit_preservation_targets': sum(sum(r['preserved_mask']) for r in rows),
                    'attribute_modified_counts': [sum(r['modified_mask'][i] for r in rows) for i in range(6)]}
                for s, rows in records.items()}
    return {'status': 'pass' if all(checks.values()) else 'fail', 'checks': checks,
            'template_family_overlap': overlap, 'coverage': coverage,
            'near_duplicate_note': 'AST-equivalent meanings and normalized exact text audited; lexical near-duplicates are intentional related intensity blocks, not a claim of independent templates.',
            'human_review_complete': False, 'release_eligible': False,
            'blockers': ['LLM paraphrase generation and real manual verification pending', '150-200 real Test-Human instructions and two independent annotations/adjudication pending']}

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')

def write_jsonl(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rows), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'candidates'/'v1')
    args = parser.parse_args()
    spec = json.loads((ROOT/'specs'/'FROZEN_V1.json').read_text(encoding='utf-8'))
    records, pairs, evidence = generate(spec)
    report = audit(records, pairs, spec)
    if report['status'] != 'pass':
        raise ValueError(canonical(report['checks']))
    args.output.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        write_jsonl(args.output/(FILES[split]+'.jsonl'), rows)
    write_jsonl(args.output/'intensity_pairs.jsonl', pairs)
    write_jsonl(args.output/'paraphrase_pairs.jsonl', [])
    write_jsonl(args.output/'ast_verification.jsonl', evidence)
    write_json(args.output/'audit.json', report)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(args.output.iterdir()) if p.is_file() and p.name != 'manifest.json'}
    write_json(args.output/'manifest.json', {'data_version': spec['data_version'], 'schema_version': spec['schema_version'],
               'seed': spec['seed'], 'release_eligible': False, 'status': 'synthetic_candidates_only',
               'spec_sha256': hashlib.sha256((ROOT/'specs'/'FROZEN_V1.json').read_bytes()).hexdigest(),
               'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'splits': {s: {'base_count': len(rows), 'variant_count': 0, 'final_row_count': len(rows),
                            'intensity_pair_count': sum(p['split'] == s for p in pairs), 'paraphrase_pair_count': 0}
                          for s, rows in records.items()},
               'test_human': {'collected_count': 0, 'status': 'pending_real_authorship_and_annotation'}, 'files_sha256': hashes})
    print(json.dumps({'output': str(args.output), 'audit': report['status'], 'bases': sum(map(len, records.values())),
                      'intensity_pairs': len(pairs), 'release_eligible': False}))

if __name__ == '__main__':
    main()
