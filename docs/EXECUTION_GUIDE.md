# Running the delivered workflow

## Latest continuation

Run `py -3 run_next_steps.py` from the workflow root to verify the original workflow plus the new paraphrase/review and geometry work. The final integrated run passed 60 tests (40 core, 5 independent evaluation, 15 new geometry/review checks). Logs are under `coordination/execution_next/`.

The 1,100 paraphrases are now authored, structurally audited and ready for real review at `data/paraphrases/v1/README.md`. T5-base weights/tokenizer are cached and passed a no-optimizer CUDA backward check. FLAME2019 is now the selected geometry demo; see `geometry/README.md` and `geometry/output_2019/`. Source-model hashes protect against cross-version coefficient use, including RingNet's FLAME2019 requirement. No RingNet image inference or official neural fitting is claimed. All setup-era missing-resource statements below are superseded by this paragraph and `reports/NEXT_STEPS_EXECUTED_2026-11-23.md`.

The saved workflow has advanced from plans to local implementation. Read `coordination/STATUS.json` for current stage evidence and `coordination/execution/run_summary.json` for the latest actual integrated run. The prior conversation itself was not retrieved; the delivered November 23 workflow, role charters, handoff and proposal supplied the execution context.

## Reproduce the available stages

From PowerShell:

```powershell
Set-Location 'C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project\Facial_Grounding_Multi_Agent_Workflow_2026-11-23'
py -3 run_workflow.py
```

This runs model/data/metric unit checks, regenerates synthetic candidates in a temporary directory and compares bytes, runs independent evaluation checks, and produces explicitly provisional synthetic diagnostics. It saves subprocess output under `coordination/execution/`. A successful exit means those available stages passed; it does not mean G2, training or the research study is complete. Re-running refreshes the diagnostic logs/results, not human reviews or an official dataset release. The runner does not download weights or fit a pretrained model.

## Artifacts and interpretation

- `data/specs/FROZEN_V1.json`: operational schema, intensity scale, seed, holdouts and split counts.
- `data/candidates/v1/`: 800 Train, 100 Val, 100 Test-ID and 100 Test-Comp synthetic base instructions, AST-derived gold labels, intensity pairs, provenance and leakage audit. These are candidates, not a reviewed release.
- `data/review/`: material for genuine manual review and collection. No completed human reviews are implied.
- `models/model_a/`: scoped lexicon baseline. Unsupported instructions yield a flagged zero fallback and remain in evaluation denominators.
- `models/model_b/` and `models/model_c/`: shared T5 architecture/trainer and consistency objective. Local tiny random-model tests establish software behavior, not pretrained-model performance.
- `evaluation/`: metric implementation and independent review; `experiments/` and `reports/` hold provisional outputs and their limitations.

The three Test-Comp heldouts are eyebrow elevation + mouth width, eye openness + jaw opening, and smile intensity + cheek raise. Train, Val and Test-ID exclude any modified set containing a heldout pair. Template families intentionally overlap; the candidates cannot support a claim about unfamiliar human wording.

## Human work that unlocks the next gate

The user confirmed on November 23 that human instructions and completed reviews do not yet exist. Follow `data/specs/HUMAN_COLLECTION_AND_REVIEW.md` and the frozen annotation conventions:

1. Produce LLM paraphrase candidates for the bases, preserving source IDs and split/group assignments. Full paraphrase generation is not included in the current synthetic base release.
2. Have real people verify meaning, record reviewer identity/date/outcome and retain corrections or rejections. AI-written checks are not human reviews.
3. Collect 150–200 independently authored instructions from participants unfamiliar with the templates. Both project authors annotate separately; retain both annotations and adjudicate disagreements.
4. Validate review evidence and split leakage, freeze the reviewed dataset, and create the official release manifest matching the neural loader contract. Merely changing status flags does not complete review.
5. Fetch the pinned T5-base revision after the data release, then run the shared B/C Train/Val pipeline. Select thresholds, checkpoints and lambda using validation only; Eval handles frozen test scoring.

The November 25 milestone must identify neural/human results as pending until these prerequisites are met. The final three-seed comparison and December 4 report/presentation cannot claim results that have not been run. Optional FLAME rendering remains deferred.

## Downloaded reference assets

`research/LOCAL_ASSETS.md` and `research/LOCAL_ASSET_INVENTORY.json` document the user-added CelebA-Dialog HQ and FLAME texture data. Regenerate that read-only inventory with `py -3 research/inventory_assets.py`. Match dataset records by image filename: `5380.jpg` lacks all three text annotations. The 6,909 smiling request IDs are indexed only as supplementary language references. Their original degree labels are not our six signed semantic controls. The inspected FLAME asset is texture space; geometry/model parameters needed for deformation were not found. These assets do not complete human review or replace Test-Human.
