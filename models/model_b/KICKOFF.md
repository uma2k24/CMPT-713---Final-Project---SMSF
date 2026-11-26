# Model B kickoff

Status: configured; implementation/training pending. Date: 2026-11-23.

## Architecture and tensor contract

Use the pretrained T5-base encoder only. The canonical order is eyebrow_elevation, eye_openness, mouth_width, smile_intensity, jaw_opening, cheek_raise.

| Tensor | Shape | Meaning |
|---|---|---|
| input_ids | [batch, tokens] | Tokenized instructions |
| attention_mask | [batch, tokens] | 1 for actual tokens, 0 for padding |
| encoder states H | [batch, tokens, hidden] | Last encoder-layer hidden states |
| pooled h | [batch, hidden] | Masked token mean |
| classification logits z | [batch, 6] | Independent modified-attribute decisions |
| probabilities p | [batch, 6] | sigmoid(z) |
| regression r | [batch, 6] | Unbounded linear-head outputs |
| y_hat | [batch, 6] | tanh(r), continuous semantic changes |
| gold y / modified mask m | [batch, 6] | Frozen gold semantic vector / binary modified labels |

Read hidden width from encoder config rather than hard-coding it. Define h_i = sum_t(mask_it * H_it) / max(sum_t(mask_it),1). Reject empty instructions or tokenization that produces no valid tokens; the denominator clamp is a numerical safeguard. Specify whether EOS contributes to the mean (proposed: yes, since it is a non-padding encoder token). Use separate linear classification/regression heads; an extra hidden MLP or extra dropout is an experimental change, requiring an explicit shared B/C decision.

The classifier predicts modification rather than mention. For 'keep the jaw unchanged', jaw is preserved and its modified target is zero. Regression predicts all six controls directly. Proposed: no classifier-based gating or post-hoc zeroing; the threshold affects only modified_mask and Attribute F1.

## Exact supervised-loss proposal

For example i and dimension j, BCE is computed from logits using a numerically stable binary cross-entropy formulation.

bce_i = (1/6) * sum_j BCEWithLogits(z_ij,m_ij)
reg_i = sum_j (y_hat_ij-y_ij)^2
L_sup = sum_i w_i*(bce_i+reg_i) / sum_i w_i

The regression term is a sum across six dimensions, matching the proposal's squared L2 norm; it is not mean squared error across dimensions. BCE uses mean across attributes. With unit example weights, the outer reduction is the batch mean. These reductions determine the effective consistency strength, so Agent C must reuse them exactly. No class weights, active-only regression mask or separate regression coefficient is proposed initially.

These are proposed implementation conventions, not a frozen coordinator decision. If per-group weighting is adopted, its reduction and sampling must be recorded and identical in B/C; never combine inverse-group weights with a group-balanced sampler without defining the resulting objective.

## Comparable supervised data for B/C

Proposed policy: both models see each approved original and verified paraphrase as supervised rows with the same gold vector. Each semantic group receives equal total supervised weight regardless of paraphrase count. Deduplicate row IDs before constructing supervised rows; a row must not acquire extra supervised weight merely because it appears in multiple consistency pairs. Pair losses are a separate C-only stream. Alternative pair-endpoint policies are valid only if frozen before runs and applied to both models.

Use identical supervised row manifests, group weighting, tokenizer/model revision, initialization seed, batch sequence, optimizer/scheduler, number of supervised updates and validation cadence. Agent C may require extra paired forward passes; report their compute cost separately. Equal supervised budgets do not imply equal wall-clock compute. B uses lambda=0; the final comparison uses the same three recorded seeds. Do not let C gain extra supervised updates by expanding pairs.

## Checkpointing and classification threshold

Proposed initial checkpoint rule: minimize all-dimension MAE on Val; break ties with the earlier checkpoint. Freeze this rule with Eval/Agent C before training. Select classification threshold on Val only (proposed: one global threshold, micro-F1 objective, deterministic tie-breaking). Do not select thresholds using test examples or use an unfrozen threshold to gate regression. Record whether thresholds are per-run or common; apply the same selection procedure to B/C.

Val remains separate from Train, including paraphrase/semantic groups. Test-ID, Test-Comp and Test-Human are only evaluated after decisions are frozen. A milestone table must report missing/unfinished runs as pending, not infer their scores.

## Reproducibility and compute discovery

Before any execution: inventory available Python/PyTorch/Transformers versions, accelerator model/VRAM, CUDA availability, CPU memory, local pretrained-cache state and checkpoint disk space. Record exact pretrained model/tokenizer identifiers and revisions. Do not claim available GPU capacity from this plan. Confirm package requirements with current official documentation when implementation begins.

Freeze max token length based on Train/Val inspection and record truncation counts; prefer rejecting/handling long inputs explicitly over unnoticed instruction truncation. Freeze optimizer, learning rate, epoch/update limit, batch/accumulation, scheduler, weight decay, gradient clipping, precision mode and early-stopping rule. The draft config intentionally leaves these unset until compute and data are known. Record random seeds for Python/NumPy/PyTorch and data loaders, deterministic settings and remaining nondeterminism. Keep the run ledger and capture actual logs/checkpoint metadata.

## Deliverable and verification plan

- Implement encoder/heads, loading and predict(texts) with exact shared output schema.
- Implement loss separately from the trainer so B/C can share it after coordinator agreement.
- Check masking/padding behavior, dimension order, boundary/range behavior and zero-change supervision.
- Verify exact loss reductions using small hand-calculated tensors and verify gradient reaches encoder and both heads.
- Check group/split integrity before loading; refuse a manifest with cross-split paraphrase groups.
- Run a tiny developer-only fixture to catch wiring errors; do not report it as held-out performance.
- Verify checkpoint round-trip prediction agreement in evaluation mode.
- Train only after freeze/compute discovery, export real validation evidence, then hand the checkpoint/config/data hashes to Eval.

## Blockers and next action

1. Data schema, full intensity scale, negative-control meaning and ambiguity/contradiction policies are not frozen.
2. Verified paraphrases and the real independently authored/annotated Test-Human set are pending.
3. Shared BCE reduction, group weighting, pair endpoint policy, threshold/checkpoint rules need coordinator freeze with Agent C/Eval.
4. Compute/dependency inventory and exact training settings are not yet established.
5. User destination C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project is write-denied in this session. These files are staged in the writable project workspace.

Next action: obtain the frozen data/schema and protocol decisions, discover compute, then implement the smallest end-to-end Model B training/prediction path. No model has been downloaded or trained, and no checkpoint or metric result exists from this kickoff.
