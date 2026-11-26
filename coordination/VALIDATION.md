# Setup validation record

## Latest continuation validation

`execution_next/run_summary.json` records the successful combined run: 40 core implementation tests, 5 independent evaluation tests and 15 geometry/paraphrase-review tests (60 total). It also records candidate integrity, review-import validation and actual FLAME2019 GPU demo execution. `execution_next/review_progress.json` shows 1,100 pending / 0 approved reviews. T5 resource preflight passed with zero optimizer steps; no official fitting or RingNet inference occurred. Geometry previews were visually inspected; semantic calibration is explicitly incomplete.

`FILES_SHA256_NEXT_STEPS.csv` is the current workflow snapshot. It excludes itself, transient Python caches and the `external/` dependency/model-cache directory. External source revision, package version, pretrained revision and user model hashes are documented separately in geometry notes, the T5 resource report and `research/FLAME_PREFLIGHT.json`. Earlier checksum lists remain historical snapshots.

## Execution validation - November 23, 2026

The current integrated run is `execution/run_summary.json`: 40 implementation unit tests, 5 independent evaluation checks, candidate regeneration and byte comparison, and provisional diagnostic inference/scoring all passed. Initial integration exposed a test-module import error; it was fixed before this successful run. Neural review validation was also strengthened to resolve actual evidence records, with passing negative tests.

Data: 800/100/100/100 synthetic bases; 1,650 within-split AST-verified intensity pairs; all 48 signed attribute/intensity primitives present in Train; holdout containment/group/text checks pass. No human paraphrases or Test-Human examples are represented as completed.

Models: A rules and B/C architecture/loss/loader/checkpoint code are implemented. Tiny random encoder fixtures validate software only. PyTorch CUDA is available on RTX 4070, but no pretrained model was downloaded or fitted. Independent metric review is `evaluation/METRIC_REVIEW_EXECUTED.md`; official version/checkpoint validation remains a requirement of the future release evaluator.

Actual provisional A predictions and zero diagnostics are saved under `experiments/provisional/`. A's frozen hash is `MODEL_A_FREEZE.json`. A scores perfectly on this controlled grammar; no test-derived modifications were made, and no human-language generalization claim follows.

Current manifest: `FILES_SHA256_EXECUTION.csv` inventories workflow artifacts excluding itself, Python caches and transient directories. `FILES_SHA256.csv` remains the original setup inventory, not a current integrity manifest. External downloaded assets remain outside the workflow package and have a separate bounded metadata audit in `research/LOCAL_ASSET_INVENTORY.json`.

2026-11-23, coordinator inspection.

- Seven reusable role charters and seven role kickoff documents are present.
- All four setup JSON files parsed successfully; two illustrative JSONL records passed six-dimension, range, binary-mask and preservation/modified checks.
- Data and metric pair manifests share canonical field names and distinguish real human paraphrase review from exact AST-derived intensity evidence.
- Eval independently inspected the metric specification. Software implementation and numerical metric tests remain pending.
- All seven role setup tasks returned completion messages. No training, participant collection or manual verification is claimed.
- The archive packaging step compares every file entry byte-for-byte by SHA-256 against its staged original, and checks entry counts. See the coordinator's delivery result for the outcome.

FILES_SHA256.csv lists all files present before that manifest itself was written; it intentionally excludes itself. The archive has no original proposal files, checkpoints, third-party datasets or generated claims of model performance.
