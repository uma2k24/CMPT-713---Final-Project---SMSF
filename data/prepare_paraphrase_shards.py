"""Prepare exact source instructions for assistant-authored paraphrase batches."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    destination = ROOT / 'paraphrases' / 'shards'
    destination.mkdir(parents=True, exist_ok=True)
    source = []
    for split in ['train', 'val', 'test_id', 'test_comp']:
        source += [json.loads(line) for line in (ROOT / 'candidates/v1' / f'{split}.jsonl').read_text().splitlines()]
    for name, batch in [('train_a', source[:400]), ('train_b', source[400:800]), ('heldout', source[800:])]:
        path = destination / f'{name}_source.jsonl'
        payload = ''.join(json.dumps({'base_id': r['id'], 'text': r['text'], 'split': r['split']}) + '\n' for r in batch)
        if path.exists() and path.read_text(encoding='utf-8') != payload:
            raise ValueError('Existing source shard differs; refusing to overwrite')
        path.write_text(payload, encoding='utf-8')
    print(json.dumps({'total': len(source), 'shards': {'train_a': 400, 'train_b': 400, 'heldout': 300}}))


if __name__ == '__main__':
    main()
