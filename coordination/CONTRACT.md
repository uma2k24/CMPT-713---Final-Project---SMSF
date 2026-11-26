# Shared project contract v0.1

## Current execution addendum (2026-11-23)

Read EXECUTION_AUTHORIZATION.md for the current authorized execution scope and frozen operational decisions. The setup-only and destination-blocked statements below describe the prior session. The delivered folder is writable, and implementation/candidate generation/local verification are now authorized. The user confirmed that human-authored instructions and manual paraphrase reviews have not yet been collected. G2 and official neural experiments remain pending that work. Schema 1.0 and frozen holdouts are documented under data/specs/; no synthetic diagnostic is an official evaluation release.

Project: Compositional and Paraphrase-Consistent Natural Language Grounding for 3D Facial Geometry.
Setup date: 2026-11-23. Read the detailed Project Proposal.pdf as experimental authority; Project Abstract + Proposal.docx is supporting background with some earlier brainstorming. Both originals exist at C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project. Preserve them. This session cannot write to that destination; stage work here and report the blocker honestly.

## Scope and roles
Current task is multi-agent workflow setup with actionable role charters, initial plans and research. Do not train models, download large assets, install packages, buy generation, or fabricate experiments during setup. No participant outreach is authorized.
Each role writes agents/<role>.md (reusable prompt, all assigned responsibilities, owned files, dependencies, acceptance criteria) plus <owned directory>/KICKOFF.md (initial work, blockers, next actions). Report exact outputs to coordinator. Agents read one another but edit only their owned paths; shared changes go through coordinator.
Researcher owns research/. Data Builder owns data/. Agent A owns models/model_a/. Agent B owns models/model_b/. Agent C owns models/model_c/ and metric implementation in evaluation/. Eval owns evaluation protocol/review (separate files from Agent C), experiments/ and reports/. Doc owns docs/. Agent C implements metrics and evaluates A after frozen data; Eval independently reviews metrics and performs final A/B/C evaluation. Coordinator owns coordination/, WORKFLOW.md, AGENTS.md and packaging.

## Canonical vector and records
Fixed order: eyebrow_elevation, eye_openness, mouth_width, smile_intensity, jaw_opening, cheek_raise. y has six values in [-1,1], changes relative to neutral, 0=no change. These are semantic controls, not direct FLAME coefficients.
modified_mask is six binary MODIFIED decisions; preserved attributes have mask 0. Store explicit preserved_mask separately; mentioned and modified are different. Unmentioned zero values are not negation examples.
JSONL fields: id, text, y, modified_mask, preserved_mask, semantic_group_id, paraphrase_group_id, template_family_id (nullable for human), composition_key, source, split, schema_version, review_status. Pair manifests link paraphrases and weak/strong intensity examples within a split. Track provenance, annotation disagreements/adjudication and counts.
Canonical paraphrase manifest: pair_id, split, left_id, right_id, semantic_group_id, verification_status, human_review_id. Primary paraphrase evaluation requires human-verified or human-adjudicated status with real reviewer evidence. Canonical intensity manifest: pair_id, split, weak_id, strong_id, attribute_index, verification_status, other_controls_equal, review_id. Exact grammar/AST-derived intensity pairs may be marked ast-verified with generator/AST evidence in review_id; naturally authored pairs require recorded semantic review. Never label ast-verified or AI-reviewed records human-verified.
Model interface: predict(texts) -> per-example y[6], modified_mask[6], optional probabilities[6], model_version, schema_version. Prediction exports include example id and split. Thresholds from validation only; no implicit regression gating unless a documented shared B/C choice.
Proposal specifies slightly=0.25, much=0.75. Data Builder must propose remaining levels/defaults, direction conventions, ambiguous instructions, contradictions, negation scope and feasibility of negative values from neutral. Label them proposed until freeze.

## Data and honesty
Targets: Train 800, Val 100, Test-ID 100, plus Test-Comp 100 synthetic BASE instructions; Test-Human 150–200. This totals 1,100 synthetic bases despite approximate 1,000 prose; record this planning interpretation and distinguish paraphrase rows.
Hold out Test-Comp combinations before generation, retain their primitives in Train. Every semantic/paraphrase group stays in one split; audit duplicate text, group/composition overlap and template-family overlap. Do not claim template families are disjoint without an audit.
LLM paraphrases require real manual meaning verification. Review states distinguish generated, AI-reviewed, human-verified and human-adjudicated. Agents cannot claim human verification. Test-Human must be authored by real people unfamiliar with templates and independently annotated by both project authors, then adjudicated. Prepare forms and mark collection pending.
Train only fits; Val tunes lambda/thresholds/checkpoints; test labels are held out from tuning. Eval owns test evaluation. Protocol changes recorded before runs, never selected from test results.

## Models
A: lexicon for attribute aliases, directions, intensities and preservation, combined with scoped rules.
B: pretrained T5-base ENCODER with attention-mask-aware mean pooling, six-label classification head p and bounded six-value regression head. L_sup = BCE(p,m) + sum_j(y_hat_j-y_j)^2, mean over examples. Record BCE dimension reduction.
C: same architecture, supervised examples and training budget; add lambda*L_cons where L_cons=sum_j(y_hat_j(x)-y_hat_j(x_prime))^2, mean over pairs. Lambda={0.1,0.5,1.0}; B is lambda=0. Freeze pair-endpoint supervision identically in B/C. Three recorded seeds for final sweep. Record software/compute/config/checkpoints and nondeterminism; no invented results.

## Six metrics from Appendix A
Attribute F1: micro TP/FP/FN over modified decisions, document zero denominator convention; per-attribute supplementary breakdown encouraged.
MAE: mean absolute error over N*6 values.
Paraphrase consistency: mean SUM squared vector difference over verified test pairs; lower better; always report MAE beside it (constant outputs are trivially consistent).
Compositional accuracy: Test-Comp fraction with max_j abs(error_j) STRICTLY <0.1.
Negation accuracy: explicit preservation targets where abs(predicted value) STRICTLY <0.1.
Intensity accuracy: eligible same-edit weak/strong pairs where abs(weak)<abs(strong), strict inequality, specified attribute.
Report all applicable metrics on all three test splits with denominators. No targets/pairs -> null/count=0/reason, never invented pairs or zero. Formal compositional accuracy applies to Test-Comp; all-vector tolerance accuracy elsewhere is a separately named supplement. Report seed values and mean/std with convention.

## Stage gates and dates
G0 workflow -> G1 frozen schema/guidelines/split policy -> G2 reviewed data + leakage audit -> G3 A/B implementations -> G4 C + validated metrics -> G5 preliminary milestone comparison -> G6 three-seed sweep/error analysis -> G7 final report/code/presentation and optional FLAME.
Milestone Nov 25; final deliverables Dec 4. Earlier planned dates are not completed work. Rebaseline from current evidence. FLAME visualization is optional and requires access/license checks plus calibrated semantic mapping.
