"""Freeze and execute the explicitly authorized provisional B/C sweep."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from data.import_paraphrase_reviews import verify_inputs
from models.model_b.data import read_jsonl, sha256
from models.model_b.provisional_data import load_provisional

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / 'experiments/provisional_v1'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def prepare():
    if EXPERIMENT.exists():
        raise ValueError('Experiment already exists; use --run to continue completed-run-aware execution')
    verify_inputs()
    EXPERIMENT.mkdir(parents=True)
    dataset = EXPERIMENT / 'data'
    dataset.mkdir()
    manifest = dict(release_status='unreleased_provisional', g2_complete=False,
                    human_review_complete=False, test_human_collected=False,
                    label_quality='AST bases and unreviewed inherited paraphrase labels', files={},
                    created_at_utc=datetime.now(timezone.utc).isoformat(), source_hashes={})
    def export(name, rows):
        path = dataset / name
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
        return dict(path=name, sha256=sha256(path), count=len(rows))
    paras_path = ROOT / 'data/paraphrases/v1/candidates.jsonl'
    pair_path = ROOT / 'data/paraphrases/v1/paraphrase_pairs.jsonl'
    paras = read_jsonl(paras_path)
    for split in ('Train', 'Val'):
        base_path = ROOT / f'data/candidates/v1/{split.lower()}.jsonl'
        rows = read_jsonl(base_path) + [r for r in paras if r['split'] == split]
        manifest['files'][split] = export(split.lower() + '.jsonl', rows)
        manifest['source_hashes'][str(base_path.relative_to(ROOT))] = sha256(base_path)
    for p in (paras_path, pair_path):
        manifest['source_hashes'][str(p.relative_to(ROOT))] = sha256(p)
    pairs = [p for p in read_jsonl(pair_path) if p['split'] in ('Train', 'Val')]
    manifest['paraphrase_pairs'] = export('pairs.jsonl', pairs)
    write_json(dataset / 'manifest.json', manifest)
    load_provisional(dataset / 'manifest.json', require_pairs=True)
    review_counts = {}
    for path in (ROOT / 'data/paraphrases/v1').glob('review_*.csv'):
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                outcome = row.get('outcome') or 'pending'
                review_counts[outcome] = review_counts.get(outcome, 0) + 1
    base = json.loads((ROOT / 'models/model_b/config.json').read_text())
    configs = EXPERIMENT / 'configs'
    configs.mkdir()
    runs = []
    for lam in (0., .1, .5, 1.):
        for seed in (13, 42, 2026):
            name = f'B_seed{seed}' if lam == 0 else f'C_lambda{lam:g}_seed{seed}'
            cfg = dict(base, seed=seed, status='authorized_provisional_training',
                       data_mode='provisional_generated', acknowledge_unreviewed_data=True,
                       budget_status='frozen_1000_updates_per_run',
                       release_manifest='experiments/provisional_v1/data/manifest.json',
                       output_dir=f'experiments/provisional_v1/runs/{name}',
                       revision_provenance='Exact pinned pretrained revision cached and preflight checked')
            path = configs / f'{name}.json'
            write_json(path, cfg)
            runs.append(dict(name=name, config=str(path.relative_to(ROOT)), config_sha256=sha256(path), lambda_value=lam))
    code = ['models/model_b/train.py', 'models/model_b/model.py', 'models/model_b/data.py',
            'models/model_b/provisional_data.py', 'models/model_c/objective.py']
    plan = dict(status='frozen_before_training', official_experiment=False, runs=runs,
                code_sha256={p: sha256(ROOT / p) for p in code},
                manifest_sha256=sha256(dataset / 'manifest.json'), review_counts=review_counts,
                checkpoint_selection='minimum Val MAE, earliest exact tie',
                lambda_selection='mean best-checkpoint Val MAE across three seeds; ties mean Val pair squared distance then smaller lambda',
                caveat='Unreviewed paraphrases; no Test-Human; validation diagnostics are provisional',
                created_at_utc=datetime.now(timezone.utc).isoformat())
    write_json(EXPERIMENT / 'plan.json', plan)
    print(json.dumps(dict(prepared=True, runs=len(runs), files=manifest['files'], review_counts=review_counts)), flush=True)


def run_sweep():
    plan = json.loads((EXPERIMENT / 'plan.json').read_text())
    for path, expected in plan['code_sha256'].items():
        if sha256(ROOT / path) != expected:
            raise ValueError(f'Frozen training code changed: {path}')
    if sha256(EXPERIMENT / 'data/manifest.json') != plan['manifest_sha256']:
        raise ValueError('Frozen manifest changed')
    summaries = []
    for spec in plan['runs']:
        path = ROOT / spec['config']
        if sha256(path) != spec['config_sha256']:
            raise ValueError('Frozen config changed')
        cfg = json.loads(path.read_text())
        output = ROOT / cfg['output_dir']
        if output.exists():
            status = json.loads((output / 'run.json').read_text())
            if status.get('status') != 'complete' or status['config'] != cfg:
                raise ValueError('Existing incomplete/mismatched run needs explicit recovery: ' + str(output))
            if sha256(output / 'best/weights.pt') != status['checkpoint_sha256']:
                raise ValueError('Checkpoint hash mismatch')
        else:
            print('START ' + spec['name'], flush=True)
            subprocess.run([sys.executable, '-u', '-m', 'models.model_b.train', '--config', str(path),
                            '--lambda-value', str(spec['lambda_value'])], cwd=ROOT, check=True)
            status = json.loads((output / 'run.json').read_text())
        metadata = json.loads((output / 'best/metadata.json').read_text())
        summaries.append(dict(name=spec['name'], seed=cfg['seed'], lambda_value=spec['lambda_value'],
                              validation_mae=status['best_validation_mae'], selected_step=status['best_step'],
                              validation_pair_squared_distance=metadata['validation_pair_squared_distance'],
                              validation_micro_f1=metadata['validation_micro_f1'],
                              elapsed_seconds=status['elapsed_seconds'], checkpoint_sha256=status['checkpoint_sha256'],
                              checkpoint=str((output/'best').relative_to(ROOT))))
        write_json(EXPERIMENT / 'progress.json', dict(completed_runs=summaries, total_runs=len(plan['runs'])))
    groups = []
    for lam in (0., .1, .5, 1.):
        runs = [r for r in summaries if r['lambda_value'] == lam]
        maes = [r['validation_mae'] for r in runs]
        groups.append(dict(lambda_value=lam, mean_validation_mae=statistics.mean(maes),
                           std_validation_mae=statistics.stdev(maes),
                           mean_validation_pair_squared_distance=statistics.mean(r['validation_pair_squared_distance'] for r in runs)))
    selected = min(groups[1:], key=lambda g: (g['mean_validation_mae'], g['mean_validation_pair_squared_distance'], g['lambda_value']))
    write_json(EXPERIMENT / 'summary.json', dict(status='complete', official_experiment=False,
               selected_c_lambda=selected['lambda_value'], groups=groups, runs=summaries,
               test_data_used_for_selection=False, human_review_complete=False, test_human_collected=False))
    print('COMPLETE: ' + str(EXPERIMENT / 'summary.json'), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.run:
        run_sweep()
