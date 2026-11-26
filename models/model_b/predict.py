"""Export frozen checkpoint predictions; does not inspect gold labels."""
import argparse
import json
from pathlib import Path
from .data import read_jsonl
from .model import load_checkpoint, Predictor


def main():
    from transformers import AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    model, metadata = load_checkpoint(args.checkpoint, args.device)
    tokenizer = AutoTokenizer.from_pretrained(Path(args.checkpoint) / 'tokenizer', local_files_only=True)
    predictor = Predictor(model, tokenizer, metadata, args.device)
    rows = read_jsonl(args.input)
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate input IDs')
    with Path(args.output).open('x', encoding='utf-8') as stream:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            predictions = predictor.predict([r['text'] for r in batch])
            for row, prediction in zip(batch, predictions):
                stream.write(json.dumps(dict(prediction, id=row['id'], split=row['split'])) + '\n')


if __name__ == '__main__':
    main()
