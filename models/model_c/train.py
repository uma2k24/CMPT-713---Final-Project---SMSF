"""Model C uses B's exact architecture, loader and training implementation."""
import argparse
import json
from models.model_b.train import run


def main():
    parser = argparse.ArgumentParser(description="Train consistency-aware Model C after G2 release")
    parser.add_argument("--config", required=True)
    parser.add_argument("--lambda-value", type=float, choices=(0.1, 0.5, 1.0), required=True)
    args = parser.parse_args()
    with open(args.config, encoding="utf-8") as stream:
        config = json.load(stream)
    run(config, consistency_lambda=args.lambda_value)


if __name__ == "__main__":
    main()
