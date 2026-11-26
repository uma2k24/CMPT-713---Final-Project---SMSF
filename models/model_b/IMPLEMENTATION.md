# Model B implementation handoff

## Current provisional training

The latest user request authorizes provisional B/C fitting on generated data. The shared trainer now accepts an explicit `provisional_generated` mode with acknowledgment of unreviewed data, using a separate integrity-checked manifest and preserving review status. Official loading still requires genuine release evidence. See `experiments/provisional_v1/README.md`, its frozen `plan.json`, and `progress.json`; the final report is `reports/PROVISIONAL_TRAINING_2026-11-23.md` after verification. Earlier no-fitting statements below describe the previous milestone.

## Current resource preflight

The exact pinned weights and tokenizer have now been downloaded into `external/huggingface_cache`, and `config.json` points the shared trainer at this project-local cache. `models/model_b/preflight/report.json` records a successful CUDA forward/backward check at batch size4 and sequence length128 using the pretrained encoder. It used no optimizer, performed zero parameter updates, and saved no fitted research checkpoint. Peak allocated CUDA memory was about1.14GB; optimizer state and Model C's extra pair passes were not measured by this check. The historical no-download statement below is superseded; official fitting remains gated on G2.

Implemented 2026-11-23. No pretrained weights were downloaded and no official model was fitted. Coordinator ran the first nine offline unit checks successfully; two further regression tests now cover linked review evidence and malformed numeric labels and require the coordinator's final combined run. Tiny random T5 fixture checks are engineering checks, not project experiments.

`model.py` implements an encoder-only T5, padding-aware mean (including valid EOS tokens), independent linear classification/regression heads, sigmoid probabilities and tanh semantic controls. Classification never gates regression. Loss is attribute-mean stable BCE plus attribute-sum squared regression, averaged over examples. For equal semantic-group weighting, each row weight is N/(number_of_groups * group_size); uniform minibatches use a fixed example denominator. This avoids the biased random ratio produced by normalizing weights within each batch.

`train.py` is the shared B/C trainer. `run(config, consistency_lambda=0)` uses the same unique supervised rows, global group weights, row shuffle RNG, optimizer and update budget for both models. C uses a separate pair RNG and two additional endpoint forward passes. Dropout RNG consumption and compute differ in C; equal supervised budgets do not imply identical compute or identical dropout masks. C's pair gradients are supplied by `models.model_c.objective.consistency_loss`.

Train fits; Val alone chooses the minimum-MAE checkpoint with earliest ties. At that checkpoint, one global threshold is selected on validation micro-F1 using 0.05..0.95 in 0.05 increments; ties choose the threshold closest to 0.5, then smaller. No test label files are opened by the trainer. A full run attempts 1,000 updates, batch size 4, constant AdamW learning rate 1e-4, weight decay .01, float32, norm clip 1.0, validation each 100 updates; resource preflight is still needed. All fit/tune text is checked against the frozen 128-token maximum, with no silent truncation. Deterministic algorithms are enabled and CUBLAS_WORKSPACE_CONFIG=:4096:8 is set before CUDA work. Failures are recorded, never silently substituted with CPU or a different precision.

The pretrained model/tokenizer is pinned to `google-t5/t5-base` revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, verified by the coordinator through the Hugging Face model metadata endpoint on 2026-11-23. `local_files_only=true` prevents incidental downloads. Missing local weights must be resolved after G2, and the preregistered resource settings must be checked before official training.

## Release gate

`data.py` rejects candidate manifests before model loading or run-directory creation. A future release manifest must have release_status=released, g2_complete=true, human_review_complete=true, test_human_collected=true, and files.Train/files.Val entries of `{path,sha256}` relative to the release directory. Files cannot escape that directory. `review_evidence` is a list of hash-verified file entries with the same shape; each referenced file contains a JSON array or JSONL records. `paraphrase_pairs`, when supplied, is a hash-verified Train/Val-only JSONL file. C requires at least one eligible Train pair.

Human/LLM rows require a human_review_id resolving to a record with review_id, endpoint_ids=[row_id], verification_status matching the row review_status, reviewer_id, reviewed_at and outcome=approved. Pair evidence additionally requires pair_id and both endpoint_ids exactly matching. Pairs must have equal semantic group, vectors and masks, within a single split. Missing or unrelated evidence fails. These checks require an auditable provenance assertion; software cannot establish that a claimed reviewer is genuinely human. Test-Human collection is checked as release metadata; its labels remain outside the training loader.

## Commands from workflow directory

```
py -3 -m unittest models.model_b.test_model_b -v
py -3 -m models.model_b.train --config models/model_b/config.json
py -3 -m models.model_b.predict --checkpoint models/model_b/runs/B_seed13/best --input PATH --output NEW_PATH
```

Training currently fails closed because no official released G2 manifest exists. Repeat the frozen configuration with seeds 13, 42 and 2026 and unique output directories only after release. Existing run directories are never overwritten. Checkpoints contain state, encoder configuration, tokenizer, attribute order/schema, data hashes, revision, configuration, software/compute inventory and validation selection evidence. Checkpoint loads use weights_only=True. Prediction export writes a new file and includes input IDs, split, continuous y, modified masks, probabilities and version fields.

API source checked: [Hugging Face T5EncoderModel documentation](https://huggingface.co/docs/transformers/model_doc/t5). Actual installed package versions and successful test logs are recorded by the coordinator in the execution evidence.
