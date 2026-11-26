# Model C kickoff

Status: configured; implementation blocked on frozen data and shared Model B contract. Scope completed here: architecture/loss/batching/evaluation plan. No training or performance results exist.

## Architecture and loss
Reuse Model B's pretrained T5-base **encoder**, attention-mask-aware mean pooling, six sigmoid classification outputs and six bounded regression outputs in the canonical attribute order. The consistency term is on continuous regression predictions only, as specified by the proposal; adding classifier consistency would be a separate experiment. Do not silently gate regression by the classifier.

Proposed shared reduction (aligned with Agent B, pending coordinator freeze):

- Per instruction: `BCE = mean_j BCE(p_j, modified_mask_j)` and `regression = sum_j (prediction_j - gold_j)^2`.
- Supervised loss is the weighted mean of per-instruction `BCE + regression`.
- For each verified pair `(x, x_prime)`, `pair_loss = sum_j (prediction_j(x) - prediction_j(x_prime))^2`.
- Consistency loss is the weighted mean of pair losses. Total objective: `L_sup + lambda * L_cons`.
- Never use a six-dimension mean for consistency while calling it the proposal's summed loss: that changes lambda's effective strength by a factor of six.

## Pair batching and fair supervision
Build a deterministic supervised schedule shared by B and C for each seed. Include both verified paraphrase endpoints in supervised training, with equal total weight per semantic group and deduplicated instruction rows. This proposal must be frozen with Agent B/Data Builder before training. Multiple paraphrases must not cause a group to receive extra supervised influence simply because more variants were generated.

A paired minibatch contains a fixed set of semantic groups, their unique selected endpoint texts, labels/masks and within-group pair indices. Forward unique texts once where practical; look up both endpoint predictions for pair loss. Backpropagate through both predictions. Pair labels must agree in values and relevant masks; differing labels or failed meaning verification invalidate a pair. Distinct text strings with different preservation scope are not automatically equivalent.

Record the exact group sampler, number of endpoint exposures, pair weighting, update count and token budgets. B receives the same supervised batches and weights as C. C's extra pair comparisons must not expand its labeled training set. If a second dropout realization or extra forward is used, record the additional computation rather than claiming identical compute. Document whether each group's pair losses are averaged before averaging across groups, so groups with many paraphrases cannot dominate unintentionally. Empty-pair minibatches contribute zero consistency loss during training and their frequency is logged; evaluation with no pairs returns null, not zero.

## Lambda/checkpoint selection
Candidates: `{0.1, 0.5, 1.0}`. Model B supplies the lambda=0 control. Proposed primary selection criterion: validation MAE, with a preregistered tie rule (lower validation consistency, then smaller lambda); coordinator/Eval must freeze this before runs. Do not inspect Test-Comp or Test-Human results to choose lambda, thresholds, epochs or seeds.

Milestone runs may use one recorded seed and must be labelled preliminary. Final workflow belongs to Eval: run the grid over three preregistered seeds, retain every seed result, and select a common lambda by a frozen validation aggregation rule rather than selecting the best seed. Recommended aggregation is mean validation MAE over all three seeds, using the same tie rule. Report all three same-lambda test results with mean and standard deviation. Record any unavailable runs explicitly.

## Metric implementation and initial A evaluation
Implement the shared contract in `evaluation/METRICS_SPEC.md` after Eval review. Validate against hand-calculated examples and a constant predictor; cover strict `<0.1` boundaries, intensity ties, empty target sets, classification zero denominators, missing predictions, pair integrity and split leakage. Eval reviews these cases independently.

Once the data manifest and Model A are frozen, export A's predictions for Test-ID, Test-Comp and Test-Human. Compute all applicable metrics with counts, split/checkpoint/schema/config identifiers and reasons for inapplicable metrics. Never invent human data or substitute generated instructions. Hand raw predictions, metric summaries and known limitations to Eval; final A/B/C comparison remains Eval's responsibility.

## Next actions and blockers
1. Data Builder freezes intensity/negation conventions, schema, pair manifests, holdouts and verified examples.
2. Agent B and coordinator freeze group weighting, pair-endpoint supervision, BCE reduction, inference mask threshold policy and no-gating convention.
3. Eval approves metric definitions and edge-case expectations, lambda-selection rule and three seeds.
4. Discover software versions, dependency availability, available compute and model access before implementation/training. No installation, download or compute expenditure has occurred during setup.
5. Implement Model C and metric code, validate, then perform initial A evaluation only when actual data and predictions exist.
