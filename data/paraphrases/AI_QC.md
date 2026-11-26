# Bounded AI semantic quality check - November 23, 2026

An independent assistant reviewer inspected 161 source/candidate pairs: 80 Train (40 from each authoring shard), 27 Val, 27 Test-ID and 27 Test-Comp. The exact source indices, IDs, original/candidate text and findings are retained in `ai_qc_findings.json`.

Sampling was fixed before inspection: Train shards use zero-based index `10*i + (i mod 4)` for i=0..39, distributing choices over the four intensity levels. Each 100-row heldout split uses `floor(i*100/27)` for i=0..26. The reviewer compared attribute identity, direction, magnitude and explicit preservation scope. No model predictions were inspected; no rule baseline or learned model was changed.

No definite meaning mismatch was identified in the 161 inspected pairs. Six Train candidates deserve particular human attention because one intensity adverb is shared across coordinated verbs. Their intended scope is plausible, but a human should confirm it or request explicit repetition of the modifier:

- `base-a32d79bc1f33ddf6d4d9f405`
- `base-a5bd60292dc9029e9a49ac81`
- `base-4e37b0fca18feb88f7596459`
- `base-9464175e8f094bd2e765d827`
- `base-a98a62e80777dd90958ae447`
- `base-68ca959511e2186974d6cec7`

These are ambiguity-risk flags, not established wrong labels. Other inspected transformations included reordered clauses, explicit spelling-out of the 0.50 default, attribute noun phrases and preservation synonyms. This finding does not establish equivalence of the unsampled 939 rows, naturalness for every reader, or physical feasibility of signed edits from neutral.

All 1,100 candidates still require genuine manual semantic review. This AI check creates no human-review evidence, changes no `review_status`, and does not authorize official training or evaluation. Human reviewers should use the frozen schema and retain corrections/rejections; a correction must be reviewed in its final wording.
