"""Download pinned resources and test one backward pass; never fit/save weights."""
import json
import os
from pathlib import Path
import time
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
from models.model_b.model import GroundingModel, supervised_loss, tokenize_checked
from models.model_b.train import environment, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def main():
    config = json.loads((ROOT/'models/model_b/config.json').read_text(encoding='utf-8'))
    destination = ROOT/'models/model_b/preflight'
    destination.mkdir(parents=True, exist_ok=True)
    report = {'status': 'started', 'pretrained_model_id': config['pretrained_model_id'],
              'pretrained_revision': config['pretrained_revision'], 'training_performed': False,
              'optimizer_steps': 0, 'research_checkpoint_created': False}
    start = time.monotonic()
    try:
        print('Downloading/reusing exact pinned model and tokenizer files...', flush=True)
        snapshot = snapshot_download(config['pretrained_model_id'], revision=config['pretrained_revision'],
            cache_dir=ROOT/'external/huggingface_cache',
            allow_patterns=['config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                            'special_tokens_map.json', 'spiece.model'])
        report['snapshot_path'] = snapshot
        print('Loading encoder and checking a single Train-only backward pass...', flush=True)
        seed_everything(13)
        device = 'cuda'
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable for requested resource preflight')
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        model = GroundingModel.from_pretrained(snapshot, config['pretrained_revision'], local_files_only=True).to(device)
        rows = [json.loads(line) for line in (ROOT/'data/candidates/v1/train.jsonl').read_text().splitlines()][:config['batch_size']]
        tokens = tokenize_checked(tokenizer, [r['text'] for r in rows], config['max_length'], device)
        # Exercise the configured sequence budget even if these test instructions are short.
        padding = config['max_length'] - tokens['input_ids'].shape[1]
        if padding > 0:
            tokens['input_ids'] = torch.nn.functional.pad(tokens['input_ids'], (0, padding), value=tokenizer.pad_token_id)
            tokens['attention_mask'] = torch.nn.functional.pad(tokens['attention_mask'], (0, padding), value=0)
        torch.cuda.reset_peak_memory_stats()
        output = model(**tokens)
        loss = supervised_loss(output['logits'], output['y'],
              torch.tensor([r['y'] for r in rows], device=device),
              torch.tensor([r['modified_mask'] for r in rows], dtype=torch.float32, device=device))
        loss.backward()
        torch.cuda.synchronize()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('Nonfinite preflight loss or gradient')
        report.update(status='pass', environment=environment(), batch_size=len(rows),
                      sequence_length=tokens['input_ids'].shape[1], gradients_finite=True,
                      encoder_has_gradients=any(p.grad is not None for p in model.encoder.parameters()),
                      peak_cuda_memory_allocated_bytes=torch.cuda.max_memory_allocated(),
                      peak_cuda_memory_reserved_bytes=torch.cuda.max_memory_reserved(),
                      parameter_count=sum(p.numel() for p in model.parameters()),
                      note='No optimizer constructed or stepped. Candidate Train rows exercised software only; loss is not a research metric.')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['elapsed_seconds'] = time.monotonic() - start
        (destination/'report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
