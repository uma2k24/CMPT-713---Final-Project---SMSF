# Metric specification — proposal Appendix A

Status: draft specification inspected by Eval; protocol freeze and code pending. Canonical dimension count D=6, order and masks from `coordination/CONTRACT.md`. The formulas follow the proposal; aggregation/empty-set conventions below are explicit implementation choices to freeze before experiments. `epsilon = 0.1` with strict inequalities.

## Input and pair contracts
Ground truth is keyed by unique `(split, id)` and includes finite gold `y[6]`, binary `modified_mask[6]`, explicit `preserved_mask[6]` and review/provenance fields. Predictions must match every evaluated id exactly once and contain finite `y_hat[6]` in [-1,1] and binary modified decisions. Reject missing/duplicate/extra ids, wrong shapes, nonfinite/out-of-range vectors or invalid masks; never silently drop examples or clip outputs after seeing results. Resolve any abstention policy in advance.

Neural modified decisions come from a threshold frozen on validation; do not derive them from gold or retune by test split. Rule baseline emits its own modified decisions under the same mask semantics.

Paraphrase manifest fields: `pair_id, split, left_id, right_id, semantic_group_id, verification_status, human_review_id`. Pairs must be unique unordered pairs, refer to existing distinct instructions within that split, and have explicit `verification_status=human-verified` or `verification_status=human-adjudicated` for semantic equivalence before primary evaluation. `human_review_id` must link to the corresponding actual human review record; an identifier alone is not evidence of completed verification. Do not compare every possible sentence pair or derive gold equivalence from similar predictions. Report number of excluded/unverified pairs separately; no automatic promotion to verified.

Intensity manifest fields: `pair_id, split, weak_id, strong_id, attribute_index, verification_status, other_controls_equal, review_id`. Endpoints must differ only in weaker/stronger intensity for the stated attribute, keeping other edits/direction fixed. `other_controls_equal` must explicitly be true and validated against gold controls. Exact template-generated intensity pairs may use `verification_status=ast-verified` when `review_id` links to concrete AST/generator evidence that the pair differs only in target intensity. Human-written or paraphrased intensity pairs require real semantic review; their `review_id` must link to an actual review record with provenance matching `verification_status`. AST checks alone cannot verify the meaning of human or paraphrased text. The gold magnitudes must establish the claimed strict ordering. Both endpoints stay within one split. Human split pairs must actually exist and be verified; do not generate replacements and call them Test-Human.

Preservation targets are unique `(instruction_id, attribute_index)` pairs drawn from explicit `preserved_mask=1`, not from every zero-valued or unmentioned dimension. Record negation surface forms separately for error analysis; preservation cues such as 'keep' are included in the proposal's negation metric.

## 1. Modified-attribute F1
Micro aggregate binary decisions over every example and all six dimensions:

`TP = sum 1[m=1 and m_hat=1]`, `FP = sum 1[m=0 and m_hat=1]`, `FN = sum 1[m=1 and m_hat=0]`.

`precision = TP/(TP+FP)`, `recall = TP/(TP+FN)`, `F1 = 2TP/(2TP+FP+FN)`.

Proposed undefined-case convention: precision or recall with zero denominator is null with reason; F1 is null only when `2TP+FP+FN=0` (no gold or predicted positives). When TP=0 and FP+FN>0, F1=0. Report TP/FP/FN, N instructions and 6N decisions. Use the direct F1 formula so undefined precision does not suppress a defined zero F1. Macro/per-attribute scores are optional supplements, not replacements for micro-F1.

## 2. Parameter MAE
`MAE = sum_i sum_j abs(y_ij - y_hat_ij) / (N*6)`.

All six dimensions count, including unmodified/preserved values. Report N and 6N denominator. Modified-only MAE may be a clearly named supplement. Empty split yields null/count=0/reason.

## 3. Paraphrase consistency
Let P be the frozen eligible within-split paraphrase-pair set:

`consistency = sum_(a,b in P) sum_j (y_hat_aj - y_hat_bj)^2 / |P|`.

This is mean **summed** squared vector distance, not dimension-mean distance, square root, cosine distance or label agreement. Lower is better. Report |P| and contributing unique-instruction count. With no eligible pairs, return null/pair_count=0/reason.

Always show MAE alongside consistency. A constant predictor has zero consistency and can still be inaccurate; add an explicitly labelled constant-predictor diagnostic to catch this failure of interpretation.

## 4. Compositional accuracy
For Test-Comp only, with frozen held-out-combination set D_comp:

`Acc_comp = sum_(i in D_comp) 1[max_j abs(y_ij - y_hat_ij) < 0.1] / |D_comp|`.

Every dimension must pass. Exactly 0.1 error fails. Report correct count, instruction count and held-out-combination coverage. On Test-ID and Test-Human, report this formal metric as null/reason='defined for Test-Comp'; optionally report the same formula under the separate name `all_dimension_tolerance_accuracy`.

## 5. Negation/preservation accuracy
For explicit preservation targets S:

`Acc_neg = sum_(i,j in S) 1[abs(y_hat_ij) < 0.1] / |S|`.

Exactly abs(prediction)=0.1 fails. Report correct target count, |S| and number of contributing instructions. No explicit targets -> null/target_count=0/reason. Unmentioned zeros never inflate this denominator. Different preserved attributes in one instruction are separate targets, as in Appendix A.

## 6. Intensity ordering accuracy
For verified weak/strong pairs Q with target attribute j:

`Acc_int = sum_(a,b,j in Q) 1[abs(y_hat_j(a)) < abs(y_hat_j(b))] / |Q|`.

Ties fail; a direction error may still pass this magnitude metric, so interpret it beside MAE and direction/error breakdowns. Report correct pair count, |Q|, target attribute coverage and unique endpoint count. No valid pairs -> null/pair_count=0/reason. Do not compare unrelated edits or pool predictions across seeds to form pairs.

## Precision, aggregation and reporting
Compute in float64 from stored model predictions where practical. Compare with literal strict boundaries; do not add an epsilon fudge factor that changes `<` to `<=`. Boundary tests should use explicitly represented values and document floating-point behavior. Keep full numerical outputs in machine-readable records; round only presentation.

Produce one result record per model/run/split with dataset version/hash, prediction artifact, schema/config/checkpoint identifiers, seed, metric values, numerators/denominators, eligibility/exclusion counts and null reasons. Report all three test splits even if some metrics are inapplicable. Never combine splits for a headline score without labelling weighting.

Final neural summaries: retain each of three seed-level values; report arithmetic mean and sample standard deviation (`ddof=1`) for the same fixed configuration. Fewer than two runs -> std=null with available count, not invented stability. A is deterministic unless its implementation declares randomness; do not fabricate three independent A seeds. Eval owns final aggregation and protocol approval.

## Required meaningful verification cases
- Perfect predictions yield F1=1, MAE=0 and eligible accuracy=1; equivalent pair outputs give consistency=0.
- One wrong dimension prevents compositional success; errors immediately below, at and above 0.1 enforce strict boundary behavior.
- Intensity ties fail; ordering is by absolute magnitude and only the named attribute.
- Preserved zero and unmentioned zero contribute differently to the preservation denominator.
- All-negative classification can have undefined F1; false positives or false negatives yield a defined zero F1.
- Empty split/pair/target sets produce null with counts/reasons; malformed ids/pairs fail validation.
- A constant all-zero predictor gives zero consistency but nonzero MAE on nonzero gold; this must not be presented as model success.


