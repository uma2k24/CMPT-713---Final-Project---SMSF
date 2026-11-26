# Evaluation protocol — proposed until coordinator freeze

Execution update, November 23: `coordination/EXECUTION_AUTHORIZATION.md` freezes the operational decisions, and `experiments/SWEEP_PLAN.json` records the planned seeds/selection rules. See `METRIC_REVIEW_EXECUTED.md` for independent implementation review and `reports/MILESTONE_STATUS_2026-11-23.md` for actual results/limits. Only provisional synthetic Model A/zero diagnostics have run; no official comparison, neural fitting or human verification is complete. The proposed/setup wording below is historical where superseded by these records.

Date: 2026-11-23. Authority: detailed proposal and coordination/CONTRACT.md. Metric implementation owner: Agent C. Evaluation acceptance and final results owner: Eval. No runs have been executed in workflow setup.

## Evaluation objects

Evaluate A, B and C on Test-ID, Test-Comp and Test-Human. Counts in the proposal refer to base instructions; report base count and expanded/paraphrase row count separately. Use immutable schema/data/manifest versions and file hashes, alongside model version, configuration, checkpoint hash, tokenizer/pretrained revision, software and hardware.

Test sets must be frozen before evaluation. Keep semantic/paraphrase groups together and record held-out composition keys. Data Builder audits exact/normalized duplicates, group overlap, composition overlap, primitive coverage and template-family overlap. Test-Human requires genuine independent authorship, two independent annotations and adjudication; uncollected or unverified data remain unavailable, not an AI substitute.

Models see Train labels only for fitting; thresholds, stopping, lambda and architecture choices use Val. Eval uses test labels solely for frozen-model scoring and post-selection analysis. Do not send test-derived fixes back into a tuned model while describing the same test evaluation as untouched. Any later exploratory revision must be clearly labeled and requires new confirmatory data for fresh claims.

## Model comparison controls

B and C share pretrained revision, architecture, pooling, heads, output bounds, tokenization, supervised rows, supervision of pair endpoints, batch weighting, optimizer, scheduler, stopping limits and seed schedule. Document additional consistency computation and wall time; do not claim identical compute if pair passes add work. Use the same policy for any regression gating. Classifier masks use a validation-frozen threshold; A returns an explicit rule decision rather than deriving mentions from zero values.

Supervised and consistency reduction conventions must be frozen with Agents B/C. The proposal's regression/consistency terms sum over six dimensions. Any mean-over-dimension change rescales effective lambda and must be recorded. Every model returns bounded raw y values and modified masks. Do not silently clip invalid predictions during scoring; fail validation and report the issue.

## Proposed selection and sweep

Seeds: 13, 42, 2026 (proposed, not yet a reported experiment). Train B at lambda=0 and C at each lambda in {0.1,0.5,1.0} for each seed: 12 neural runs total, plus A's deterministic evaluation. Pair the same seeds and data order where feasible; record remaining nondeterminism.

For each neural run, choose the checkpoint with the lowest validation MAE; break exact ties by earliest checkpoint. Choose the final C lambda using mean validation MAE across the three predeclared seeds. Break exact ties by lower mean validation consistency, then smaller lambda. Freeze all choices before test scoring. Use each seed's checkpoint at that shared lambda for the primary C summary; never choose a seed using test results. Report the other lambdas in a sweep table as secondary analyses after selection. Three seeds are repeated runs, not three independent datasets.

Milestone evaluation may use fewer completed runs if clearly marked preliminary, with explicit seed/run counts and unavailable cells. It must not be described as the final three-seed sweep. Final comparisons use all scheduled successful runs; failed or missing runs are listed and prevent an unqualified claim of completion.

## Metrics and eligibility

Metric formulas and strict boundaries are defined in Agent C's evaluation/METRICS_SPEC.md, to be independently reviewed. Required fields for each reported value: model, split, seed, metric_name, value, eligible_count, numerator if meaningful, exclusion_count/reason, data_version, checkpoint/config reference and metric_version.

| Metric | Eligible population | Interpretation |
| --- | --- | --- |
| Attribute micro-F1 | Six modified/unmodified decisions per test instruction | Modified and mentioned attributes differ; explicit preservation has gold modified=0 |
| MAE | All six outputs for all valid instructions | Primary absolute grounding error; lower is better |
| Paraphrase consistency | Verified within-split equivalent-input pairs | Mean sum-squared vector difference; lower is better; report alongside MAE |
| Compositional accuracy | Test-Comp instructions | All six errors strictly below 0.1 |
| Negation accuracy | Explicit preservation instruction/attribute targets | Absolute predicted value strictly below 0.1; unmentioned zeros do not qualify |
| Intensity accuracy | Predefined within-split weaker/stronger pairs and specified attribute | Strict magnitude ordering; ties fail |

Formal compositional accuracy is only defined for Test-Comp. It is null/not_applicable on Test-ID/Human. If useful, report a separately named all_dimension_tolerance_accuracy on other splits. Missing pairs or preservation targets yield null, eligible_count=0 and a reason. Never construct post-hoc favorable pairs from predictions. Empty whole test splits are unavailable and produce no ordinary score.

Proposed micro-F1 zero convention: calculate 2TP/(2TP+FP+FN); if this denominator is zero, return null with reason=no_positive_gold_or_predictions. If Agent C adopts another explicit convention, resolve it before freeze; do not silently mix conventions. Record TP/FP/FN and the full N*6 decision count separately from the F1 denominator.

Use original float outputs and strict '<0.1'; rounding display values must not affect pass/fail. Test exactly equal to epsilon, ties in intensity, empty pair lists, modified versus preserved masks and pair membership. Agent C owns tests; Eval records independent review evidence.

## Aggregation and baselines

Report every seed-level result. For B/C, aggregate the same split/metric over completed seeds with arithmetic mean and sample standard deviation (ddof=1), alongside n_runs. Standard deviation is null for fewer than two values. Missing values are excluded with explicit reasons, never replaced with zero. A is deterministic: evaluate once, seed=null and std=null; do not fabricate three stochastic A runs.

A constant-zero predictor is an optional diagnostic, not one of the three proposed models. It illustrates why consistency/negation can look good without correct edits. Also report active-attribute MAE and per-attribute errors as supplementary diagnostics so unmodified dimensions do not conceal failures. Keep these separate from the six preregistered metrics. Seeds alone do not justify statistical significance claims; report paired deltas and limitations without overstating confidence.

## Error-analysis record

Freeze category rules before counting. Record example id, provenance, composition key, attribute, gold/predicted values and masks, model/checkpoint/seed, error magnitude and flags for preservation scope, negation, intensity, ambiguous wording and unseen combinations. Flags may overlap; report overlapping counts honestly. Include both representative successes and failures and pair examples where wording changes predictions. Report category eligibility and denominators rather than only cherry-picked text.

Compare B/C on the same examples and seed; distinguish sign errors, magnitude errors, spurious edits, missed edits and preservation violations. For intensity, incorrect sign can coexist with correct magnitude ordering, so inspect grounding MAE/sign alongside the proposal's ordering metric. For unseen combinations, verify all component primitives occurred in Train and analyze by edit count/composition key.

## Optional FLAME and release

Researcher verifies model/data access terms. A calibrated semantic-control mapping and fixed neutral identity/camera/render settings are prerequisites. Render the same input/gold/predictions across models and label all visualizations supplementary; semantic vectors are not native FLAME coefficients. Rendering does not replace numeric evaluation or resolve anatomy ambiguity automatically.

Before release, ensure tables trace to versioned runs, human reviews are genuine, unavailable work is marked, and Doc can reconstruct the milestone/final narrative. Preserve author privacy in released data and retain pseudonymous provenance only where necessary.

## Research claim boundaries

The proposed loss adapts consistency regularization to continuous semantic vectors; do not describe it as a literal reproduction of Yin et al.'s original objective. Researcher identified a later Talk-to-Edit extension with 3D-aware editing, so avoid claiming that all prior Talk-to-Edit work is only 2D or that language-driven 3D face editing itself is new. Ground originality claims in the actual task, controlled comparisons and evidence. All shared B/C supervision/reduction choices remain proposed until the coordinator records their freeze.

