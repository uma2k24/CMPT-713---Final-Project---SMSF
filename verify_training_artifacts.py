"""Reload fitted B/selected-C checkpoints, reproduce Val scores, and report evidence."""
import gc
import json
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer
from models.model_b.data import sha256
from models.model_b.model import load_checkpoint
from models.model_b.provisional_data import load_provisional
from models.model_b.train import seed_everything, validation_outputs, select_threshold

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT/'experiments/provisional_v1'


def save(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def main():
    summary = json.loads((EXPERIMENT/'summary.json').read_text())
    if summary['status'] != 'complete' or len(summary['runs']) != 12:
        raise ValueError('Complete 12-run sweep required')
    _, splits, _ = load_provisional(EXPERIMENT/'data/manifest.json', require_pairs=True)
    destination = EXPERIMENT/'validation_predictions'
    destination.mkdir(exist_ok=True)
    checks = []
    plan = json.loads((EXPERIMENT/'plan.json').read_text())
    for path, digest in plan['code_sha256'].items():
        if sha256(ROOT/path) != digest:
            raise ValueError('Frozen training source changed: '+path)
    for run in summary['runs']:
        checkpoint = ROOT/run['checkpoint']
        status = json.loads((checkpoint.parent/'run.json').read_text())
        if status['status'] != 'complete' or status.get('supervised_updates') != 1000 or not status.get('training_completed'):
            raise ValueError('Run did not complete the frozen update budget')
        events = [json.loads(line) for line in (checkpoint.parent/'steps.jsonl').read_text().splitlines()]
        if [e['step'] for e in events] != list(range(1, 1001)):
            raise ValueError('Incomplete or unordered optimizer-step log')
        if not all(math.isfinite(e[k]) for e in events for k in ('supervised_loss', 'consistency_loss', 'total_loss')):
            raise ValueError('Nonfinite training losses')
        if sha256(checkpoint/'weights.pt') != run['checkpoint_sha256']:
            raise ValueError('Saved checkpoint hash mismatch')
        if run['lambda_value'] not in (0., summary['selected_c_lambda']):
            continue
        seed_everything(run['seed'])
        model, metadata = load_checkpoint(checkpoint, device='cuda')
        tokenizer = AutoTokenizer.from_pretrained(checkpoint/'tokenizer', local_files_only=True)
        mae, predicted, probabilities = validation_outputs(model, tokenizer, splits['Val'], metadata['config'], 'cuda')
        difference = abs(mae-run['validation_mae'])
        if difference > 1e-6:
            raise ValueError(f'Checkpoint Val MAE does not reproduce: {run["name"]}: {difference}')
        threshold, f1 = select_threshold(probabilities, torch.tensor([r['modified_mask'] for r in splits['Val']]))
        if threshold != metadata['classification_threshold'] or abs(f1-metadata['validation_micro_f1']) > 1e-6:
            raise ValueError('Checkpoint Val classification threshold/F1 does not reproduce')
        rows = [dict(id=r['id'], split='Val', y=y, probabilities=p,
                     modified_mask=[int(x >= metadata['classification_threshold']) for x in p],
                     model_version=metadata['model_version'], official_experiment=False)
                for r, y, p in zip(splits['Val'], predicted.tolist(), probabilities.tolist())]
        path = destination/(run['name']+'.jsonl')
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        checks.append(dict(run=run['name'], validation_mae=mae, absolute_difference=difference,
                           predictions_sha256=sha256(path)))
        print('RELOADED '+run['name']+' Val MAE='+str(mae), flush=True)
        del model, tokenizer, predicted, probabilities
        gc.collect()
        torch.cuda.empty_cache()
    save(EXPERIMENT/'checkpoint_verification.json', dict(status='pass', all_12_checkpoint_hashes_verified=True,
         all_12_training_logs_verified=True, total_optimizer_updates=12000, frozen_training_code_verified=True,
         validation_reproduction=checks, official_experiment=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, ax = plt.subplots(figsize=(9, 5.2), layout='constrained')
    for lam, color in zip((0., .1, .5, 1.), ('#404040', '#0072B2', '#D55E00', '#009E73')):
        traces = []
        for run in summary['runs']:
            if run['lambda_value'] != lam:
                continue
            events = [json.loads(line) for line in (ROOT/run['checkpoint']).parent.joinpath('steps.jsonl').read_text().splitlines()]
            validations = [e for e in events if 'validation_mae' in e]
            steps = [e['step'] for e in validations]
            traces.append([e['validation_mae'] for e in validations])
        values = np.array(traces)
        label = 'B: supervised' if lam == 0 else f'C: lambda = {lam:g}'
        ax.plot(steps, values.mean(axis=0), color=color, label=label, linewidth=2)
        ax.fill_between(steps, values.min(axis=0), values.max(axis=0), color=color, alpha=.12)
    ax.set(xlabel='Supervised optimizer updates', ylabel='Validation MAE (lower is better)',
           title='Provisional T5 training on generated instructions\nMean of three seeds; shading shows seed range')
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.savefig(EXPERIMENT/'validation_learning_curves.png', dpi=180)
    fig.savefig(EXPERIMENT/'validation_learning_curves.svg')
    plt.close(fig)
    lines = ['# Provisional training completed', '',
        'All 12 planned T5-base encoder runs completed: Model B and Model C at lambda 0.1, 0.5, and 1.0, each with seeds 13, 42, and 2026. Each run performed 1,000 AdamW updates on the RTX 4070 using float32.', '',
        'The frozen snapshot contains 1,600 Train rows (800 AST bases plus 800 generated paraphrases), 200 Val rows, 800 Train pairs and 100 Val pairs. All 1,100 source paraphrases remain generated and pending human review. No Test-Human data exists. These are provisional experiments; G2 remains incomplete.', '',
        'B and C use identical supervised data, architecture, group weights, batch size, update budget and supervised row order for each seed. C adds separately sampled pairs and therefore uses more compute. Checkpoints minimize Val MAE at 100-update intervals; ties retain the earlier checkpoint. C lambda minimizes mean best-checkpoint Val MAE across three seeds, with pair distance then smaller lambda as exact tie breaks. No test labels were used for fitting or selection.', '',
        '| Model | Lambda | Val MAE mean | Sample std | Val pair squared distance mean |',
        '|---|---:|---:|---:|---:|']
    for group in summary['groups']:
        lines.append(f'| {"B" if group["lambda_value"] == 0 else "C"} | {group["lambda_value"]:g} | {group["mean_validation_mae"]:.6f} | {group["std_validation_mae"]:.6f} | {group["mean_validation_pair_squared_distance"]:.6f} |')
    lines += ['', f'Selected provisional Model C lambda: **{summary["selected_c_lambda"]:g}**. Validation numbers are selection diagnostics, not held-out performance estimates. Pair distances are computed on generated pairs and are not the official human-verified paraphrase-consistency metric.', '',
              '![Validation training curves](../experiments/provisional_v1/validation_learning_curves.png)', '',
              '| Run | Best update | Val MAE | Val attribute micro F1 | Seconds |', '|---|---:|---:|---:|---:|']
    for run in summary['runs']:
        lines.append(f'| {run["name"]} | {run["selected_step"]} | {run["validation_mae"]:.6f} | {run["validation_micro_f1"]:.6f} | {run["elapsed_seconds"]:.1f} |')
    lines += ['', '## Verification and artifacts', '',
        'Seventeen targeted tests passed before training: 14 existing model/loss tests and 3 new provisional-loader tests. They cover checkpoint round trips, gradients, strict official release loading, data integrity, and pair mismatches. All 12 fitted checkpoint hashes were checked. The six B/selected-C checkpoints were reloaded from disk and reproduced their stored Val MAE within 1e-6.', '',
        '- Frozen plan and code/config hashes: `experiments/provisional_v1/plan.json`',
        '- Aggregate and per-run results: `experiments/provisional_v1/summary.json`',
        '- Checkpoints: `experiments/provisional_v1/runs/<run>/best/` (weights, encoder config, tokenizer, threshold and metadata)',
        '- Every run retains `run.json` and every-update loss history in `steps.jsonl`.',
        '- Checkpoint reload evidence: `experiments/provisional_v1/checkpoint_verification.json`',
        '- Reproduced validation predictions: `experiments/provisional_v1/validation_predictions/`', '',
        '## Remaining research work', '',
        'The fixed budget gives each run 4,000 supervised row presentations (2.5 passes through 1,600 rows); convergence is not established. Complete real paraphrase review and independently authored/dual-annotated Test-Human, then audit and release the reviewed dataset and rerun official experiments. The existing six AI-QC intensity-scope flags remain for human review. Test-ID/Test-Comp performance has not been measured for these fitted checkpoints. FLAME2019 is available for geometry, but the semantic-to-geometry mapping remains only partially demonstrated; geometry assets are not T5 training labels.', '',
        'The earlier milestone reports and execution summaries describe the state before this training run. This report supersedes their statements that B/C have never been fitted; their human-review and official-release limitations still apply.', '']
    (ROOT/'reports/PROVISIONAL_TRAINING_2026-11-23.md').write_text('\n'.join(lines), encoding='utf-8')
    status_path = ROOT/'coordination/STATUS.json'
    status = json.loads(status_path.read_text(encoding='utf-8-sig'))
    status.update(project_execution_status='provisional_12_run_neural_sweep_complete_G2_pending',
        trained_models=['B (provisional)', 'C (provisional)'],
        completed_experiment_runs=[r['name'] for r in summary['runs']],
        official_experiments_completed=False, provisional_training='experiments/provisional_v1/summary.json',
        training_verification='experiments/provisional_v1/checkpoint_verification.json',
        training_report='reports/PROVISIONAL_TRAINING_2026-11-23.md',
        selected_provisional_c_lambda=summary['selected_c_lambda'],
        deliverable_note='12 provisional B/C training runs completed and all checkpoint hashes verified; selected B/C Val scores reproduced after reload. Human review, Test-Human, official experiments and full geometry calibration remain pending.',
        training_targeted_checks_passed=17)
    for role in status['roles']:
        if role['name'] in ('Agent B', 'Agent C'):
            role.update(execution_status='provisional_neural_training_complete', next='Complete reviewed G2 release before official rerun')
    status['blockers'] = [b.replace('required before B/C fitting', 'required before official B/C fitting') for b in status['blockers']]
    for gate in status['gates']:
        if gate['id'] in ('G3', 'G4'):
            gate['status'] = 'implemented_verified_provisionally_fitted_official_G2_pending'
        if gate['id'] == 'G6':
            gate['status'] = 'provisional_three_seed_sweep_complete_official_sweep_and_error_analysis_pending'
    save(status_path, status)
    print('REPORT: reports/PROVISIONAL_TRAINING_2026-11-23.md', flush=True)


if __name__ == '__main__':
    main()
