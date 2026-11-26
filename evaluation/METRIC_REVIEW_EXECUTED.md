# Independent implementation review - November 23, 2026

Reviewer: Eval worker, separate from metric author Agent C. Reviewed `metrics.py`, `run_provisional.py`, the delivered candidate audit/manifest, `models/model_b/data.py`, the original proposal Appendix A extracted in `coordination/PROPOSAL_TEXT.txt`, and saved provisional results. This is a software/protocol review, not human data verification.

## Accepted for provisional synthetic diagnostics

All six formulas match the proposal and frozen contract: micro-F1 over modified decisions; MAE over N*6; mean pairwise sum squared difference; Test-Comp all-dimension error strictly below 0.1; explicit preserved-target absolute value strictly below 0.1; strict absolute-magnitude ordering for eligible intensity pairs. Zero F1 denominator and missing populations return null with reasons. Unmentioned zeros do not enter preservation accuracy. Raw floating-point values determine boundaries; display rounding does not change outcomes.

The implementation requires exact prediction ID coverage and same-split endpoints, rejects duplicate pairs and malformed/nonfinite vectors, checks equivalence/other-control equality before scoring, and distinguishes real linked human-review evidence from exact template AST evidence. Pair counts and exclusion reasons are reported. The training loader now resolves hashed review records and validates row/pair endpoint linkage rather than accepting a review-ID string alone. A program cannot establish that a person truly performed a review; the retained evidence must still be genuine.

The provisional runner checks candidate hashes, the audit result and Model A's frozen source hash, includes every test example, saves predictions and explicitly labels Test-Human, B/C and paraphrase consistency unavailable. Perfect A scores on this grammar do not support a claim of robustness to human wording. The zero diagnostic's perfect preservation but nonzero MAE appropriately illustrates the limitation of preservation alone.

## Evidence and additional independent check

Coordinator reported passing Agent C's 13 metric tests and 3 objective tests, plus four independently hand-calculated coordinator tests. Eval inspected those fixtures and added `test_delivered_candidates_independent_holdout_and_hash_audit` to `test_independent_review.py`. It independently checks delivered file hashes, 800/100/100/100 counts, unique semantic signatures, heldout-set containment, all 48 signed intensity primitives in Train, 1,650 within-split ordered pairs, and the empty paraphrase manifest. Eval did not execute a subprocess; the coordinator's final integrated run is the execution evidence for this added test.

## Limits before official release

Candidate data remain unreleased. Full LLM paraphrases are not generated, manual reviews are absent, and 150–200 human instructions with two independent author annotations/adjudication are absent. No B/C fitted checkpoint, official A/B/C comparison or three-seed study exists.

`metrics.evaluate` validates numeric/identity semantics but does not itself enforce `schema_version` or `model_version`. Current diagnostics use a fixed generator/manifest and frozen A implementation; before official evaluation, the release evaluator must explicitly validate schema, attribute order and model/checkpoint versions. This is an integration requirement, not a numerical-metric defect. Similarly, full Test-Human provenance and all-split release approval belong to the data release process; the neural trainer intentionally reads Train/Val only.

Template families overlap across every split pair. Intensity examples form four-level related blocks and the 150 test intensity pairs are not 150 independent instructions. Report the existing example/pair denominators without independence or significance claims. No post-test rule tuning was made during this review.
