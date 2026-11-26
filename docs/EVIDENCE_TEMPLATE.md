# Evidence record template

Copy a relevant section for each artifact/run. Use pending or not applicable with a reason; never invent values. Keep provenance files in their owner's directory and link them here.

## Artifact or decision

- Record ID / date / owner:
- Stage and status (configured, ready, in_progress, blocked, complete, validated):
- Description / reason:
- Relative artifact paths and version or hash:
- Source reference / exact URL / access date where relevant:
- Checks performed and exact evidence:
- Limitations / unresolved issue:
- Next action and next owner:

## Dataset release

- Dataset and schema version / frozen annotation-policy version:
- Generator/grammar/config versions and generation provenance:
- Canonical attribute order / intensity and ambiguity policies:
- Base instructions, paraphrases and total rows per split (separate counts):
- Semantic/paraphrase group and composition manifests:
- Duplicate/group/composition/template overlap audit and findings:
- Human-authorship provenance; independent annotations and adjudication records:
- Paraphrase review states and genuine manual reviewer evidence:
- Paraphrase/intensity pair manifests and preservation-target counts:
- Freeze approval/decision reference / remaining exclusions:

## Training and checkpoint

- Model A/B/C / run ID / seed / data version:
- Source/config identity; pretrained model/tokenizer exact revision:
- Architecture, loss equations/reductions and output-mask policy:
- Supervised rows/group weights and pair endpoint policy; B/C budget comparison:
- Optimizer/scheduler, batches/updates, precision and deterministic settings:
- Python/dependency versions; actual hardware/compute and nondeterminism:
- Exact command / execution status / timestamps / logs:
- Validation checkpoint and threshold selection rule/evidence:
- Lambda candidate and selection evidence (validation only):
- Checkpoint path/hash and round-trip check:

## Evaluation or analysis

- Frozen protocol/metric version / split/data version:
- Model/checkpoint/config / prediction paths/hashes:
- Per-seed six metric values and eligible counts; null reasons:
- MAE next to paraphrase consistency; strict epsilon and intensity conventions:
- Mean/std with aggregation convention and number of actual seeds:
- Comparability limitations, missing runs or absent Test-Human evidence:
- Error examples by negation/intensity/unseen combination; supported conclusion:
- Optional FLAME access/license, control calibration and visualization provenance:
- Eval review reference / report location:

## Report or presentation claim

- Claim and scope:
- Supporting source/run/data records:
- Observation versus inference:
- Limitations and uncertainty:
- Reviewer and final artifact/version:
