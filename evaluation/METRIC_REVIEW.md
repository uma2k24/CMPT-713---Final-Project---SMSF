# Independent metric specification review

Reviewer: Eval. Date: 2026-11-23. Artifact: evaluation/METRICS_SPEC.md, draft written by Agent C. Review type: specification inspection only; no metric implementation exists or has been executed in this setup.

Outcome: the six formulas and populations align with coordination/CONTRACT.md and the proposal Appendix A. The specification correctly preserves summed squared vector distance, strict <0.1 bounds, strict intensity magnitude ordering, modified-versus-mentioned masks and explicit preservation targets. It rejects malformed prediction/pair records and records missing eligibility as null with counts. Its proposed F1 zero-denominator convention agrees with Eval's proposed protocol.

The specification also correctly distinguishes formal Test-Comp compositional accuracy from the supplementary all_dimension_tolerance_accuracy elsewhere, retains seed-level outputs with sample standard deviation, and warns that constant outputs can be perfectly consistent but inaccurate. Eval has aligned the supplementary metric name in PROTOCOL.md.

Before acceptance for experimental use:

- Coordinator must freeze undefined-case and aggregation conventions, seeds and validation selection rules with the protocol.
- Data Builder must provide frozen verified manifests and provenance; test paraphrase pairs and intensity pairs cannot be assumed to exist, particularly in Test-Human.
- Agent C must implement the metrics and execute the listed meaningful verification cases, including boundary, empty-set, duplicate/malformed ID and constant-predictor checks.
- Eval must inspect those actual results and validate prediction coverage/config provenance before accepting a scored release.

This review accepts the draft specification as a suitable implementation target. It does not validate software, predictions, datasets or model results.

Integration review: Agent C subsequently added human_review_id to paraphrase records and explicit human-verified/human-adjudicated eligibility, plus other_controls_equal and review_id for intensity records. These additions make provenance/equality requirements explicit and are consistent with this review; execution remains pending.
