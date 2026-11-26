# Same Meaning, Same Face

**Compositional and Paraphrase-Consistent Natural Language Grounding for 3D Facial Geometry**  
CMPT 713 - Natural Language Processing, Simon Fraser University  
Jill Ference and Utsav Anantbhat

This project maps facial editing instructions to six signed semantic controls and asks whether paraphrase-consistency training improves accuracy on unfamiliar wording and unseen combinations of edits. It compares a rule-based baseline with supervised and consistency-regularized T5 encoders. A local FLAME2019 viewer renders the predicted controls through a separate geometric mapping.

**Status as of November 25, 2026:** the implementations, 12 provisional neural training runs, synthetic evaluation, six-control geometric calibration, and milestone report are complete. Human paraphrase verification, Test-Human collection, and perceptual validation remain pending. Current results are provisional and do not establish performance on independent human instructions.

- [Milestone report (five-page PDF)](reports/milestone_2026-11-24/milestone.pdf)
- [Editable LaTeX and BibTeX build instructions](reports/milestone_2026-11-24/README.txt)
- [Training report](reports/PROVISIONAL_TRAINING_2026-11-23.md)
- [Synthetic evaluation and error analysis](reports/AI_VALIDATION_2026-11-23.md)
- [FLAME calibration report](reports/FLAME_CALIBRATION_2026-11-24.md)

## Quick start: existing local workspace

Run the commands below from **`Final Project - SMSF`**, the directory containing this README. They use the Windows Python launcher and the existing Python environment with the project's dependencies installed.

```powershell
Set-Location 'C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project\Final Project - SMSF'

# Check implementation tests, data integrity, and saved experiment hashes.
py -3 run_flame_live.py --check

# Start the viewer in the foreground; use Ctrl+C to stop it.
py -3 run_flame_live.py --serve
```

Open **http://127.0.0.1:8765**. The viewer offers:

| Mode | What it does |
|---|---|
| Text edits | Runs frozen Model A, B, or C and renders all six regression values. |
| Six controls | Applies direct signed controls, independently of language-model errors. |
| Raw basis | Explores native expression PCs and jaw rotation without semantic labels. |

Try `Raise the eyebrows much.`, `Increase the smile much.`, or `Raise the eyebrows slightly and open the jaw moderately.` Model B/C predictions can include unintended edits; inspect the displayed values. Model A flags unsupported wording and uses its zero-vector fallback.

The viewer loads the existing seed-13 B checkpoint and seed-13 C checkpoint at validation-selected `lambda=0.5`. It uses CUDA when available and otherwise CPU. Both neural checkpoints are loaded at startup, even when using direct sliders or Model A. It binds to localhost only.

For a detached Windows server, use `py -3 run_flame_live.py --launch`; its log and process record are written under `geometry/live_output/`. If port 8765 is occupied, use `--serve --port 8766` and open the corresponding URL. The full `--check` command requires the existing datasets, FLAME assets, all 12 checkpoints, and saved evaluation files; it is not an asset-free installation test.

## Task and models

The vector order is fixed:

| Index | Control | Positive direction |
|---:|---|---|
| 0 | `eyebrow_elevation` | Raise eyebrows |
| 1 | `eye_openness` | Increase eyelid opening |
| 2 | `mouth_width` | Widen mouth |
| 3 | `smile_intensity` | Increase smile intensity |
| 4 | `jaw_opening` | Open jaw |
| 5 | `cheek_raise` | Raise cheeks |

Targets lie in `[-1, 1]`. Zero means no change relative to the reference. The annotation scale uses magnitudes `0.25`, `0.50`, `0.75`, and `1.00`; an omitted intensity defaults to `0.50`. These values are semantic labels, not direct FLAME coefficients or human-calibrated physical units.

For example, the intended annotation for `Raise the eyebrows slightly and open the jaw moderately.` is:

```json
{
  "y": [0.25, 0, 0, 0, 0.5, 0],
  "modified_mask": [1, 0, 0, 0, 1, 0],
  "preserved_mask": [0, 0, 0, 0, 0, 0]
}
```

`preserved_mask` records explicit requests to keep an attribute unchanged. Unmentioned attributes have zero targets but are not automatically preservation examples. See the [frozen schema](data/specs/FROZEN_V1.json) and [annotation guidelines](data/specs/ANNOTATION_DRAFT.md).

| Model | Implementation | Training objective |
|---|---|---|
| A | Attribute lexicon, intensity mappings, and scoped rules | No neural fitting |
| B | T5-base encoder, masked mean pooling, classification and bounded regression heads | Supervised loss |
| C | Same architecture and supervised examples as B | Supervised loss plus paraphrase agreement |

The supervised loss is mean binary cross-entropy over six modified-attribute decisions plus the sum of six squared regression errors, averaged over examples with semantic-group weighting. Consistency is the mean summed squared difference between predictions for equivalent instructions, with gradients through both endpoints. Classification thresholds are selected on Validation; regression outputs are not gated by those decisions.

## Environment and assets

### Recorded execution environment

| Component | Recorded version or hardware |
|---|---|
| Python | 3.12.2 |
| PyTorch | 2.9.1+cu130 |
| Transformers | 5.3.0 |
| NumPy | 2.2.6 |
| SMPL-X package | 0.1.28, installed under `external/python_packages/` |
| GPU | NVIDIA GeForce RTX 4070, 12 GB |

Additional runtime dependencies include SciPy, Matplotlib, Pillow, and the Hugging Face model/tokenizer dependencies. Asset inventory scripts also use `pypdf`. Report rendering uses PyMuPDF and Poppler; report compilation uses LaTeX/BibTeX or the local Tectonic executable.

This is an existing research workspace, not an installable Python package. There is currently no complete dependency lockfile. For a separate machine, create a Python 3.12 environment and match the recorded versions; a clean installation has not been independently validated. The upstream `external/FLAME_PyTorch/requirements.txt` is for that repository's original environment and is not this project's dependency specification.

### Expected folder layout

```text
Project/
|-- Final Project - SMSF/                 # This README and project code
|   |-- external/huggingface_cache/       # Pinned pretrained T5 resources
|   |-- external/python_packages/        # Local SMPL-X and PDF dependencies
|   |-- geometry/calibration/assets/
|   |   `-- flame_static_embedding.pkl
|   `-- experiments/provisional_v1/runs/  # Trained checkpoints
|-- FLAME/
|   `-- FLAME2019/FLAME2019/generic_model.pkl
`-- CelebA-Dialog (HQ)/                   # Optional supplementary resource
```

The supplied FLAME model and CelebA-Dialog files remain separate from the code. Acquire missing assets through their upstream distribution channels and follow their terms. This README does not grant redistribution rights to datasets, model files, or third-party software.

The geometric mapping requires the exact FLAME2019 model with SHA-256:

```text
e7f4f6ac7aa97045b3989893386aa94dc5245ded7105bbbf44116bfdf801a050
```

The static landmark embedding is also hash-pinned. FLAME2020/2023 coefficients are not interchangeable with this calibration. The project does not run a RingNet image predictor or reconstruct the CelebA photographs.

Training uses `google-t5/t5-base` at revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`. To download/reuse that pinned resource and run the CUDA preflight:

```powershell
py -3 -m models.model_b.preflight
```

This command performs one backward-pass check, writes a preflight report, and performs **no optimizer updates**. It requires CUDA and may access Hugging Face. Checkpoint inference instead loads the tokenizer and weights from the saved checkpoint directory.

## Data and provenance

| Split | Synthetic bases | Generated paraphrases | Total instructions |
|---|---:|---:|---:|
| Train | 800 | 800 | 1,600 |
| Validation | 100 | 100 | 200 |
| Test-ID | 100 | 100 | 200 |
| Test-Comp | 100 | 100 | 200 |
| Test-AI | Separate AI-authored challenge | 75 intended-equivalent pairs | 150 |
| Test-Human | Planned independent collection | Not collected | 0; target 150-200 |

Base labels are derived from the generation grammar. All 1,100 paraphrases retain generated status and proposed inherited labels. Structural validation covers IDs, labels, masks, hashes, and split/group integrity; it does not prove meaning preservation. An AI semantic review of 150 pairs flagged six cases. It is not human review, and historical AI review samples should not be added together without deduplication.

Test-Comp holds out the modified-attribute combinations **eyebrows + mouth width**, **eyes + jaw**, and **smile + cheeks**. Every individual primitive occurs in Train, and no Train/Validation/Test-ID modified set contains one of these pairs. Template families overlap across splits, so this is not a test of wholly unseen template families.

Test-AI contains 150 assistant-authored instructions, with labels supplied by the same authoring process. Fifty rows share a target/preservation signature with Train/Validation. It is a lexical/scope diagnostic, not Test-Human or a claim of semantic disjointness.

### CelebA-Dialog's current role

The local HQ inventory contains 30,000 images and 29,999 records per text annotation JSON; `5380.jpg` lacks text annotations. A separate export contains **6,909 smiling requests**, retaining the original degree/target fields. These are supplementary language references and were **not used in the completed training, evaluation, or FLAME calibration**.

Source attribute scores cannot be directly converted into the project's six signed controls. A separately annotated Test-CelebA study and, optionally, a controlled augmentation experiment are proposed future work. The dataset's text must not be relabeled as the independent human collection required by the proposal. See the [reference manifest](research/celeba_references/manifest.json).

### Human review workflow

Use the [paraphrase review pack](data/paraphrases/v1/README.md) and [human collection protocol](data/specs/HUMAN_COLLECTION_AND_REVIEW.md). The review sheets are `data/paraphrases/v1/review_*.csv`; collection, dual-author annotation, and adjudication templates are under `data/review/`.

After actual reviews are recorded, validate them into a new directory:

```powershell
py -3 -m data.import_paraphrase_reviews --reviews data/paraphrases/v1/review_train.csv data/paraphrases/v1/review_val.csv data/paraphrases/v1/review_test_id.csv data/paraphrases/v1/review_test_comp.csv --output data/review/imports/review_001
```

The importer validates recorded evidence but does not declare the official dataset release complete. Genuine review, independent Test-Human collection, dual annotation/adjudication, and the release audit remain required. Never fill human-review fields with AI-generated reviewer actions.

## Training and checkpoint inference

The completed provisional sweep contains B (`lambda=0`) and C (`lambda=0.1, 0.5, 1.0`) for seeds **13, 42, and 2026**. Each run uses 1,000 AdamW updates, a supervised batch of four, learning rate `1e-4`, weight decay `0.01`, gradient clipping `1.0`, maximum sequence length 128, and float32 precision. C samples four additional paraphrase pairs per update. Supervised exposure and update counts match; compute costs do not.

Checkpoints minimize Validation MAE, checked every 100 updates. C's weight is selected by mean Validation MAE over the three seeds, yielding **`lambda=0.5`**. Test labels are not used for fitting or selection. Recorded run time averages approximately 68 seconds for B and 127 seconds for C, including initialization, validation, and checkpoint writes. The short budget does not establish convergence.

```powershell
# Verify and skip completed runs; execute only runs that have no output directory.
py -3 run_provisional_training.py --run

# Reload saved checkpoints and verify recorded validation results.
py -3 verify_training_artifacts.py
```

`--run` is not a fresh retraining command: it verifies frozen code/config/checkpoint hashes, skips completed runs, and refuses incomplete or mismatched existing runs. It does not resume optimizer state. `--prepare` refuses the already-existing experiment directory. New experiments require a documented new version and output paths; preserve the current evidence. The default official model config still targets an unreleased reviewed dataset; use the frozen provisional configs for these completed runs.

### Predict on your own instructions

Save UTF-8 JSONL as `my_instructions.jsonl`, one object per line:

```json
{"id":"demo-001","text":"Raise the eyebrows slightly.","split":"Demo"}
{"id":"demo-002","text":"Increase the smile much.","split":"Demo"}
```

Then run the shared B/C prediction interface:

```powershell
py -3 -m models.model_b.predict --checkpoint experiments/provisional_v1/runs/C_lambda0.5_seed13/best --input my_instructions.jsonl --output my_predictions.jsonl --device cuda --batch-size 4
```

Use `--device cpu` if needed. For B, replace the checkpoint path with `experiments/provisional_v1/runs/B_seed13/best`. Input IDs must be unique, and the output file must not already exist. No gold labels are required. The export contains semantic predictions and model metadata; use the viewer to render them through FLAME.

## Evaluation and preliminary findings

The six evaluation metrics are attribute micro-F1, mean absolute error, verified paraphrase consistency, strict Test-Comp accuracy, explicit-preservation accuracy, and weak/strong intensity-ordering accuracy. Full-vector and preservation tolerances are strictly below `0.1`. Metrics with no eligible examples are unavailable, not zero. Generated-pair distances are separate diagnostics, because no paraphrase pairs currently have genuine human verification.

The table below summarizes existing frozen-model results. B/C are means over three seeds; C uses validation-selected `lambda=0.5`. A is deterministic. Full sample standard deviations and denominators are in [results.json](experiments/ai_validation_v1/results.json).

| Split | A F1 / MAE | B F1 / MAE | C F1 / MAE |
|---|---|---|---|
| Test-ID (200) | 0.7073 / 0.1381 | 0.9828 / 0.1283 | 0.9892 / 0.1216 |
| Test-Comp (200) | 0.6928 / 0.0950 | 0.4541 / 0.1524 | 0.4812 / 0.1423 |
| Test-AI (150) | 0.2128 / 0.1425 | 0.6174 / 0.1764 | 0.6384 / 0.1581 |

C improves mean error over B on these sets, but strict Test-Comp accuracy remains **2.50%**, versus **0.67%** for B and **53.0%** for A. A benefits from known templates and rejects 137 of 150 Test-AI instructions. Its zero fallback can make global MAE and preservation accuracy look favorable without understanding the instruction. Read the [error analysis](reports/AI_VALIDATION_2026-11-23.md) before interpreting rankings.

`run_ai_validation.py --prepare` freezes the AI review/challenge inputs, and `--evaluate` evaluates frozen models. Both stages have already completed in `experiments/ai_validation_v1/`; the runner deliberately refuses to overwrite those inputs or results. Use the saved report and JSON for the current study. A new evaluation requires an explicitly versioned destination, not deletion of the existing results.

## FLAME calibration and exported visualizations

```powershell
# Recompute measurement checks and regenerate PNG/GIF/OBJ calibration outputs.
py -3 calibrate_flame.py --animate

# Inspect landmark and calibration definitions without the full sweep.
py -3 calibrate_flame.py --inspect
```

Calibration fits actual FLAME expression coefficients to nine target measurements and three inner-lip separation constraints. The geometric reference uses a jaw angle of `0.175` radians; signed jaw controls span absolute angles `0` through `0.35` radians. Other controls use documented millimeter-scale changes. Smile means lip-corner lift; cheek raise means elevation of fixed surface patches.

The recorded checks passed all **54 single-control cases** and accepted **96 of 100 mixed cases**. Four mixed requests exceeded the `+/-3` coefficient envelope and were explicitly rejected. No requested controls were silently clipped. These are geometric checks, not human perceptual validation or full mesh-collision guarantees.

- [Static control comparison](geometry/calibration/control_sweeps.png)
- [Animated control sweeps](geometry/calibration/control_sweeps.gif)
- [Calibration specification](geometry/calibration/specification.json)
- [Numerical validation](geometry/calibration/validation.json)

Browser animations interpolate endpoint vertices for display; exported sweep frames are separately decoded by FLAME. The historical `run_flame_live.py --export` output remains a jaw-only demonstration. Use `calibrate_flame.py --animate` for the current six-control exports.

## Checks and project layout

The last recorded full verification passed **74 tests**, checked all **12 trained checkpoint hashes**, and verified **21 evaluation prediction files**. It did not repeat training. Logs are in `geometry/live_output/verification.log` and `verification.json`. To run focused geometry checks:

```powershell
py -3 -m unittest geometry.test_flame geometry.test_semantic_calibration geometry.test_live_engine -v
```

| Directory | Contents |
|---|---|
| `data/` | Schema, generator, base instructions, paraphrases, and human-review templates |
| `models/model_a/` | Rule baseline |
| `models/model_b/` | Shared neural architecture, training, checkpoint loading, and prediction |
| `models/model_c/` | Consistency objective |
| `evaluation/` | Metric implementations and evaluation checks |
| `experiments/` | Frozen plans, configs, training histories, checkpoints, and predictions |
| `geometry/` | FLAME decoder, calibration, viewer, and generated meshes/previews |
| `research/` | Asset audits, provenance, related work, and CelebA reference export |
| `reports/` | Experiment reports and milestone LaTeX/PDF sources |
| `docs/` | Execution guidance and project history |
| `external/` | Third-party resources, local packages, cached weights, and report tools |

### Rebuild the milestone

The source package uses the supplied ACL template and a BibTeX bibliography. In the existing workspace:

```powershell
py -3 reports/milestone_2026-11-24/build_milestone.py --compile --render
```

Inspect the rendered pages under `tmp/pdfs/milestone_2026-11-24/`, then run the same script with `--package` to create the PDF and editable source ZIP under `output/pdf/`. `--prepare` regenerates the included tables and figure from saved experiment records and exported meshes. The [report README](reports/milestone_2026-11-24/README.txt) also explains standalone LaTeX and Overleaf builds.

## Troubleshooting and remaining work

- **Moved workspace:** current code generally resolves paths relative to its own location, but historical inventories and the recorded T5 preflight contain the previous absolute folder name. The older `run_next_steps.py` checks that saved snapshot path; rerun the T5 preflight if you need that older check after relocation. Historical documents can describe earlier project stages. Prefer current result JSON, calibration evidence, and the milestone report for status.
- **Existing server after a move:** `--launch` reuses a server already on the requested port. Stop the old instance before restarting from this directory, or use a free port, then refresh the browser.
- **Missing checkpoints or tokenizer:** the viewer needs both saved B/C checkpoint directories. Inspect `experiments/provisional_v1/summary.json` for the exact paths. The pretrained T5 cache alone is not a fitted project checkpoint.
- **CUDA unavailable:** checkpoint inference and the viewer support CPU; the recorded training configs and T5 resource preflight request CUDA explicitly.
- **Rejected geometric combination:** reduce the direct slider intensities. The viewer retains the last accepted mesh and displays an error; coefficient rejection does not mean the language prediction was altered.
- **Missing module:** use the environment containing the recorded dependencies. `smplx` is loaded from `external/python_packages/`; that directory is not a replacement for the full Python environment.

Next research steps are genuine paraphrase review; collection and independent dual annotation of 150-200 Test-Human instructions; an audited reviewed-data release; final A/B/C evaluation and error analysis; optional separately annotated CelebA transfer experiments; and human assessment of the geometric proxies. Do not tune against the existing test results or replace missing human evidence with synthetic records.
## References and project documentation

- [T5: Raffel et al. (2020)](https://jmlr.org/papers/v21/20-074.html)
- [Compositional generalization: Keysers et al. (2020)](https://arxiv.org/abs/1912.09713)
- [Consistency regularization: Yin et al. (2023)](https://aclanthology.org/2023.acl-long.72/)
- [Talk-to-Edit and CelebA-Dialog: Jiang et al. (2021)](https://arxiv.org/abs/2109.04425)
- [FLAME: Li et al. (2017)](https://flame.is.tue.mpg.de/)
- [Project BibTeX references](reports/milestone_2026-11-24/references.bib)
- [Workflow](WORKFLOW.md), [execution authorization](coordination/EXECUTION_AUTHORIZATION.md), and [machine-readable status](coordination/STATUS.json)

The original proposal remains in the parent project directory. Preserve original datasets, model assets, and frozen experiment files when extending the project.
