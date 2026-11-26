# Eval kickoff

Status: configured; evaluation execution blocked by prerequisites. Date: 2026-11-23.

## Completed in setup

- Wrote reusable Eval charter, proposed evaluation protocol, three-seed sweep manifest and milestone report template.
- Separated Agent C's metric implementation/initial A scoring from Eval's independent acceptance and final all-model scoring.
- Specified validation-only choices, matched B/C supervision and budgets, per-split eligibility, seed aggregation and error-analysis evidence.

## Not completed

No model training, checkpoint collection, test prediction, metric execution, human-data collection or experimental comparison has occurred. No numeric result is available. Proposed seeds and selection tie-breaks require coordinator acceptance before runs.

## Readiness gates

1. G1: Data Builder freezes anatomy/intensity/negation policy, schema and composition holdout; B/C freeze supervision, reductions and selection rules.
2. G2: Data Builder supplies group-disjoint split and verified pair manifests, leakage audit and genuine Test-Human provenance/adjudication.
3. G3/G4: A/B/C supply working versioned interfaces/checkpoints; Agent C supplies metrics and meaningful boundary/corner-case checks.
4. Eval independently reviews metrics and validates coverage/IDs/ranges against manifests, then freezes the evaluation release.
5. G5: Score available frozen models for a clearly preliminary November 25 milestone; list missing evidence.
6. G6/G7: Complete the planned sweep, analyze the required error categories, write final evidence and optionally render FLAME. Final report/code/presentation target: December 4.

## Blockers and next action

Current destination C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project is write-denied; files are staged in the coordinator's workspace. Dataset and models have not been built in this setup. Real human generation/verification remains human work. Immediate next action: obtain coordinator agreement on proposed evaluation choices and review Agent C's metric specification when available; no execution should be represented as completed before these gates pass.
