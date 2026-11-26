# Provisional neural training

The user explicitly requested training on November 23, 2026. The frozen plan uses existing generated candidates without claiming human review. Official dataset loading remains strict and separate.

Run commands from the workflow directory:

```powershell
py -3 run_provisional_training.py --run
py -3 verify_training_artifacts.py
```

Preparation has already run; do not use `--prepare` on this existing experiment. The runner skips completed runs after verifying config and checkpoint hashes, and refuses to overwrite incomplete runs. It does not resume partial optimizer state. A failure must retain its evidence and use a documented new attempt directory/config.

The sweep contains B with lambda 0, and C with lambda 0.1, 0.5, and 1.0, each for seeds 13, 42 and 2026. All runs perform 1,000 optimizer updates. C adds four paraphrase pairs per update. No test labels enter fitting, checkpoint selection, threshold tuning, or lambda selection.

`plan.json` freezes code/config and dataset hashes. `progress.json` lists completed runs; each running directory has its own `run.json` and every-update `steps.jsonl`. `summary.json` is written only after all 12 runs finish. `checkpoint_verification.json` and the report are written by the verification command after training.

Saved models are in `runs/<name>/best/`. Both models use the shared prediction command:

```powershell
py -3 -m models.model_b.predict --checkpoint experiments/provisional_v1/runs/B_seed13/best --input my_instructions.jsonl --output my_predictions.jsonl --device cuda --batch-size 4
```

Each input JSONL row needs a unique `id`, `text`, and `split` (use `Demo` for new examples). Output paths must not already exist. This command requires no gold labels. Predictions are six semantic controls, not calibrated FLAME coefficients. The checkpoint metadata records `official_experiment: false`.

Human paraphrase reviews and Test-Human are still pending. The Val pair-distance diagnostic uses generated pairs and must not be reported as official human-verified paraphrase consistency.
