"""Frozen Model A and zero diagnostic on unreleased synthetic candidates only."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from evaluation.metrics import evaluate
from models.model_a.model import predict

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    source = ROOT / 'data/candidates/v1'
    manifest = load(source / 'manifest.json')
    if manifest['release_eligible'] or manifest['status'] != 'synthetic_candidates_only':
        raise ValueError('This command is only for explicitly unreleased synthetic candidates')
    for name, sha in manifest['files_sha256'].items():
        if digest(source / name) != sha:
            raise ValueError(f'Candidate hash mismatch: {name}')
    if load(source / 'audit.json')['status'] != 'pass':
        raise ValueError('Candidate leakage audit has not passed')
    code_hash = digest(ROOT / 'models/model_a/model.py')
    if load(ROOT / 'coordination/MODEL_A_FREEZE.json')['code_sha256'] != code_hash:
        raise ValueError('Model A changed after freeze; do not silently reuse heldout evaluation')
    pairs = rows(source / 'intensity_pairs.jsonl')
    evidence = rows(source / 'ast_verification.jsonl')
    results = []
    destination = ROOT / 'experiments/provisional'
    destination.mkdir(parents=True, exist_ok=True)
    for split, name in [('Test-ID', 'test_id'), ('Test-Comp', 'test_comp')]:
        gold = rows(source / f'{name}.jsonl')
        outputs = predict([r['text'] for r in gold])
        for row, output in zip(gold, outputs):
            output.update(id=row['id'], split=split, code_sha256=code_hash)
        (destination / f'model_a_{name}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in outputs), encoding='utf-8')
        zero = [{'id': r['id'], 'split': split, 'y': [0.] * 6, 'modified_mask': [0] * 6,
                 'schema_version': '1.0', 'model_version': 'constant-zero-diagnostic-1.0'} for r in gold]
        (destination / f'zero_{name}.jsonl').write_text(
            ''.join(json.dumps(r) + '\n' for r in zero), encoding='utf-8')
        for model_name, predictions in [('A', outputs), ('constant_zero', zero)]:
            report = evaluate(gold, predictions, split=split,
                              intensity_pairs=[p for p in pairs if p['split'] == split], review_evidence=evidence)
            report.update(model=model_name, seed=None, data_version=manifest['data_version'],
                          status='provisional_unreviewed_synthetic_diagnostic',
                          unsupported_count=sum(p.get('status') == 'unsupported' for p in predictions))
            results.append(report)
    record = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'status': 'provisional_unreviewed_synthetic_diagnostic',
              'official_evaluation': False, 'data_manifest_sha256': digest(source / 'manifest.json'),
              'model_a_code_sha256': code_hash, 'metric_code_sha256': digest(ROOT / 'evaluation/metrics.py'),
              'results': results,
              'unavailable': {'Test-Human': 'No real human-authored/dual-annotated data collected',
                              'B': 'Not fitted; reviewed data release required',
                              'C': 'Not fitted; reviewed data and verified Train paraphrases required',
                              'paraphrase_consistency': 'No human-verified paraphrase pairs'},
              'interpretation': 'Grammar diagnostics only. Overlapping template families and no independently authored language; no evidence of neural or paraphrase gains.'}
    save(destination / 'results.json', record)
    lines = ['# Provisional synthetic diagnostics', '',
             'Actual Model A inference and metric computation on the unreleased synthetic candidates. '
             'These are software/grammar diagnostics, not the proposal\'s completed model comparison.', '',
             '| Model | Split | N | Attribute F1 | MAE | Composition | Preservation | Intensity | Unsupported |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for item in results:
        def number(key):
            metric = item['metrics'][key]
            return 'unavailable' if metric['value'] is None else f"{metric['value']:.6f} (n={metric['eligible_count']})"
        lines.append(f"| {item['model']} | {item['split']} | {item['instruction_count']} | " +
                     ' | '.join(number(k) for k in ['attribute_f1', 'mae', 'compositional_accuracy',
                                                    'negation_accuracy', 'intensity_accuracy']) +
                     f" | {item['unsupported_count']} |")
    lines += ['', 'Paraphrase consistency is unavailable: zero verified pairs. Test-Human and neural B/C results are unavailable. '
              'The zero predictor illustrates that perfect preservation can coexist with poor grounding and tied intensity outputs.', '',
              'All original float values, denominators, exclusions, hashes and null reasons are in `experiments/provisional/results.json`. '
              'Each scored ID has a saved prediction. A was frozen before scoring; no test-derived rule adjustment was made.', '',
              'G2 remains pending real human review and collection. Template families overlap, so this result does not establish robustness '
              'to unfamiliar phrasing. Official comparison and the three-seed sweep have not run.']
    (ROOT / 'reports/PROVISIONAL_DIAGNOSTICS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'result_file': str(destination / 'results.json'), 'scored_splits': ['Test-ID', 'Test-Comp'],
                      'official_evaluation': False, 'test_human': 'unavailable'}))


if __name__ == '__main__':
    main()
