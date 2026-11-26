"""Shared B/C trainer. Invoke as a module from the workflow directory."""
import argparse
import json
import platform
import os
import random
import re
from pathlib import Path
import time
import numpy as np
import torch
import transformers
from .data import group_weights, load_release, sha256
from .model import GroundingModel, save_checkpoint, supervised_loss, tokenize_checked

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed):
    # Must be set before the first CUDA BLAS operation, including model loading.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def environment():
    return {'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'numpy': np.__version__,
            'cuda_available': torch.cuda.is_available(), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else None}


@torch.no_grad()
def validation_outputs(model, tokenizer, rows, config, device):
    model.eval()
    ys, probabilities = [], []
    for start in range(0, len(rows), config['batch_size']):
        batch = rows[start:start + config['batch_size']]
        out = model(**tokenize_checked(tokenizer, [r['text'] for r in batch], config['max_length'], device))
        ys.append(out['y'].cpu())
        probabilities.append(out['probabilities'].cpu())
    predicted = torch.cat(ys)
    gold = torch.tensor([r['y'] for r in rows], dtype=torch.float32)
    return float((predicted - gold).abs().mean()), predicted, torch.cat(probabilities)


def select_threshold(probabilities, targets):
    """Val-only global micro-F1; ties: closest to .5, then smaller threshold."""
    if probabilities.shape != targets.shape or probabilities.numel() == 0:
        raise ValueError('Matching nonempty probability/target tensors required')
    candidates = [i / 100 for i in range(5, 96, 5)]
    choices = []
    for threshold in candidates:
        mask = probabilities >= threshold
        truth = targets.bool()
        tp = int((mask & truth).sum())
        fp = int((mask & ~truth).sum())
        fn = int((~mask & truth).sum())
        denominator = 2 * tp + fp + fn
        if denominator:
            choices.append((2 * tp / denominator, -abs(threshold - .5), -threshold, threshold))
    if not choices:
        return .5, None
    chosen = max(choices)
    return chosen[3], chosen[0]


def _configuration(config, consistency_lambda):
    if consistency_lambda not in (0., .1, .5, 1.):
        raise ValueError('Use the frozen lambda grid: 0, .1, .5, 1')
    if config.get('seed') not in (13, 42, 2026):
        raise ValueError('Use a preregistered seed')
    if config.get('pretrained_model_id') != 'google-t5/t5-base':
        raise ValueError('Official runs require google-t5/t5-base')
    if not re.fullmatch(r'[0-9a-f]{40}', config.get('pretrained_revision') or ''):
        raise ValueError('Freeze an exact 40-character pretrained commit before training')
    for key in ('batch_size', 'pair_batch_size', 'max_length', 'max_updates', 'validation_every'):
        if not isinstance(config.get(key), int) or isinstance(config[key], bool) or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    if config.get('precision') != 'float32' or config.get('scheduler') != 'constant':
        raise ValueError('Frozen implementation uses float32 and constant learning rate')
    if config.get('learning_rate', 0) <= 0 or config.get('gradient_clip_norm', 0) <= 0:
        raise ValueError('Positive learning rate/gradient norm required')


def run(config, consistency_lambda=0.):
    """Fits only after a released G2 manifest and immutable model revision exist."""
    manifest_path = (PROJECT_ROOT / config['release_manifest']).resolve()
    # Gate before imports/downloads/creating run directories. Missing manifests fail closed.
    mode = config.get('data_mode', 'official')
    if mode == 'provisional_generated' and config.get('acknowledge_unreviewed_data') is True:
        from .provisional_data import load_provisional
        manifest, splits, pairs = load_provisional(manifest_path, require_pairs=consistency_lambda > 0)
    elif mode == 'official':
        manifest, splits, pairs = load_release(manifest_path, require_pairs=consistency_lambda > 0)
    else:
        raise ValueError('Unknown data mode or missing unreviewed-data acknowledgment')
    _configuration(config, consistency_lambda)
    output = (PROJECT_ROOT / config['output_dir']).resolve()
    if not output.is_relative_to(PROJECT_ROOT) or output.exists():
        raise ValueError('Use a new output directory within this workflow; never overwrite a run')
    output.mkdir(parents=True)
    start_time = time.monotonic()
    run_info = {'config': config, 'consistency_lambda': consistency_lambda,
                'data_mode': mode, 'official_experiment': mode == 'official',
                'release_manifest_sha256': sha256(manifest_path),
                'release_manifest': manifest, 'environment': environment(),
                'status': 'initializing', 'training_completed': False}
    status_path = output / 'run.json'
    def save_status():
        status_path.write_text(json.dumps(run_info, indent=2), encoding='utf-8')
    save_status()
    try:
        seed_everything(config['seed'])
        device = config['device']
        if device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA unavailable; no silent device substitution')
        cache_dir = str(PROJECT_ROOT / config['cache_dir']) if config.get('cache_dir') else None
        tokenizer = transformers.AutoTokenizer.from_pretrained(config['pretrained_model_id'],
                             revision=config['pretrained_revision'], local_files_only=config.get('local_files_only', True), cache_dir=cache_dir)
        model = GroundingModel.from_pretrained(config['pretrained_model_id'], config['pretrained_revision'],
                             local_files_only=config.get('local_files_only', True), cache_dir=cache_dir).to(device)
        train, val = splits['Train'], splits['Val']
        # Audit all fit/tune text lengths before optimizer work; no test labels are loaded.
        for rows in (train, val):
            for pos in range(0, len(rows), config['batch_size']):
                tokenize_checked(tokenizer, [r['text'] for r in rows[pos:pos + config['batch_size']]], config['max_length'])
        weights = group_weights(train)
        by_id = {r['id']: r for r in train}
        train_pairs = [p for p in pairs if p['split'] == 'Train']
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        row_rng, pair_rng = random.Random(config['seed']), random.Random(config['seed'] + 100000)
        order, position = [], 0
        best_mae, best_step = float('inf'), None
        run_info['status'] = 'training'
        save_status()
        with (output / 'steps.jsonl').open('w', encoding='utf-8') as log:
            for step in range(1, config['max_updates'] + 1):
                # Fixed-size batches sampled across shuffled epochs. Independent pair RNG
                # ensures B/C consume the same supervised row order.
                indexes = []
                while len(indexes) < config['batch_size']:
                    if position == len(order):
                        order = list(range(len(train)))
                        row_rng.shuffle(order)
                        position = 0
                    amount = min(config['batch_size'] - len(indexes), len(order) - position)
                    indexes.extend(order[position:position + amount])
                    position += amount
                rows = [train[i] for i in indexes]
                model.train()
                optimizer.zero_grad(set_to_none=True)
                out = model(**tokenize_checked(tokenizer, [r['text'] for r in rows], config['max_length'], device))
                sup = supervised_loss(out['logits'], out['y'],
                       torch.tensor([r['y'] for r in rows], dtype=torch.float32, device=device),
                       torch.tensor([r['modified_mask'] for r in rows], dtype=torch.float32, device=device),
                       torch.tensor([weights[i] for i in indexes], dtype=torch.float32, device=device))
                consistency = sup.new_zeros(())
                if consistency_lambda:
                    from models.model_c.objective import consistency_loss
                    batch_pairs = pair_rng.choices(train_pairs, k=config['pair_batch_size'])
                    left = model(**tokenize_checked(tokenizer, [by_id[p['left_id']]['text'] for p in batch_pairs], config['max_length'], device))['y']
                    right = model(**tokenize_checked(tokenizer, [by_id[p['right_id']]['text'] for p in batch_pairs], config['max_length'], device))['y']
                    consistency = consistency_loss(left, right)
                loss = sup + consistency_lambda * consistency
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training objective')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'], error_if_nonfinite=True)
                optimizer.step()
                event = {'step': step, 'supervised_loss': float(sup.detach()),
                         'consistency_loss': float(consistency.detach()), 'total_loss': float(loss.detach())}
                if step % config['validation_every'] == 0 or step == config['max_updates']:
                    mae, predictions, probs = validation_outputs(model, tokenizer, val, config, device)
                    event['validation_mae'] = mae
                    val_indexes = {r['id']: i for i, r in enumerate(val)}
                    pair_errors = [(predictions[val_indexes[p['left_id']]] - predictions[val_indexes[p['right_id']]]).square().sum()
                                   for p in pairs if p['split'] == 'Val']
                    pair_mse = float(torch.stack(pair_errors).mean()) if pair_errors else None
                    event['validation_pair_squared_distance'] = pair_mse
                    if mae < best_mae:
                        best_mae, best_step = mae, step
                        threshold, f1 = select_threshold(probs, torch.tensor([r['modified_mask'] for r in val]))
                        metadata = dict(run_info, model_version=f"{'B' if consistency_lambda == 0 else 'C'}-seed{config['seed']}-step{step}",
                                        classification_threshold=threshold, validation_micro_f1=f1,
                                        validation_mae=mae, selected_step=step,
                                        validation_pair_squared_distance=pair_mse,
                                        checkpoint_selection='minimum_Val_MAE_earliest_exact_tie',
                                        threshold_selection='Val_micro_F1_grid_0.05_to_0.95_step_0.05_tie_closest_0.5_then_lower')
                        save_checkpoint(output / 'best', model, metadata, tokenizer)
                    run_info.update(last_step=step, best_step=best_step, best_validation_mae=best_mae,
                                    elapsed_seconds=time.monotonic() - start_time)
                    save_status()
                    print(json.dumps(dict(event, run=config['output_dir'])), flush=True)
                log.write(json.dumps(event) + '\n')
                log.flush()
        run_info.update(status='complete', training_completed=True, best_step=best_step,
                        best_validation_mae=best_mae, elapsed_seconds=time.monotonic() - start_time,
                        supervised_updates=config['max_updates'], additional_pair_forwards_per_update=2 if consistency_lambda else 0,
                        checkpoint_sha256=sha256(output / 'best' / 'weights.pt'))
        save_status()
        return run_info
    except Exception as error:
        run_info.update(status='failed', error=f'{type(error).__name__}: {error}',
                        elapsed_seconds=time.monotonic() - start_time)
        save_status()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--lambda-value', type=float, default=0.)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
    print(json.dumps(run(config, args.lambda_value), indent=2))


if __name__ == '__main__':
    main()
