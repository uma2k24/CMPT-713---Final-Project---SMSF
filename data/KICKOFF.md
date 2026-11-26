# Data Builder kickoff

Status: configured; annotation, grammar and split specifications drafted. Dataset generation, LLM paraphrasing, manual verification and Test-Human collection have NOT occurred.

## Outputs
- agents/data_builder.md: reusable role charter and staged acceptance criteria.
- data/specs/ANNOTATION_DRAFT.md: six-control ontology, proposed scale, masks and ambiguity/conflict policy.
- data/specs/GRAMMAR_AND_SPLITS.md: AST-first grammar, record/pair contract, count interpretation, split design and release audit.
- data/specs/HUMAN_COLLECTION_AND_REVIEW.md: proposed independent-writing prompt and annotation/paraphrase-review forms.
- data/specs/EXAMPLE_RECORDS.jsonl: two synthetic illustrations explicitly excluded from releases.

## Key decisions requiring freeze
1. The proposal fixes slightly=.25/much=.75; .50 default and endpoint policy are proposed.
2. Signed edits from neutral may not be physically realizable for all dimensions. Keep semantic labels separate from FLAME calibration; decide whether a non-neutral reference is needed.
3. Do not raise is neither reverse direction nor unambiguous full preservation. Smile wider is ambiguous. Proposed policy is adjudication/exclusion rather than silent labels.
4. Exact synthetic base targets total 1,100, with paraphrases counted separately.
5. Proposed composition keys use modified attribute sets; freeze held-out keys and whether containment in larger training sets is forbidden.
6. Human authorship, two independent annotations/adjudication and human paraphrase verification remain real human dependencies.

## Next actions and handoffs
Coordinator approves/records G1 decisions with project authors; Agent A/B/C/Eval review schema and semantics. Then implement deterministic generator and allocation/audit code, freeze manifests, and arrange authorized human work. Eval receives held-out labels after data freeze; model agents receive only authorized fitting/tuning inputs.

## Storage blocker
Requested destination C:\SFU Grad\Fall 2026\CMPT 713 - NLP\Project was write-denied to the coordinator. All outputs are staged under facial_grounding in the writable workspace; no destination write retries were made by this agent.

## Research handoff incorporated
Researcher identifies the custom corpus as the primary supervision source. CelebA-Dialog supplies 2D editing requests for five attributes that do not match this six-control continuous-vector task; it cannot be treated as already labeled 3D supervision. Any use is supplementary phrasing research with explicit provenance and a fresh annotation decision. Semantic controls remain separate from direct FLAME coefficients. See research/KICKOFF.md for verified source/access details.
