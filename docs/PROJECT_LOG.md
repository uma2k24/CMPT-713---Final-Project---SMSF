# Project log

## Latest continuation - actual next-step results

User authorized proceeding and added FLAME2019 after identifying RingNet's model-version constraint. Three Data Builder authoring agents completed 1,100 actual LLM paraphrase strings. Structural assembly passed; an independent AI review sampled 161 pairs and flagged six possible shared-modifier scope issues, without claiming human verification. Review CSVs and a provenance-validating importer are ready, with 1,100 pending and zero approved reviews.

Pinned T5-base weights/tokenizer were downloaded locally. A real CUDA forward/backward check at batch 4 / sequence 128 passed with no optimizer updates or fitted checkpoint. A separate local SMPL-X dependency supports the FLAME geometry-only adapter. FLAME2019 is selected by default, with separate earlier2023 artifacts and a strict parameter-bundle model hash/RingNet2019 check. The GPU geometry demo and an independent comparison with upstream mesh equations passed. Semantic calibration is explicitly incomplete.

Final integrated validation passed 60 tests and the review-import/data-integrity checks; canonical logs are `coordination/execution_next/`. The local CelebA reference export retains 6,909 smiling requests with original labels, never project gold labels. Human collection, manual review and official fitting remain pending; no originals were modified.

## Latest execution handoff - November 23, 2026

User requested running the saved workflow; coordinator resumed execution in the delivered, now-writable project folder. Data Builder, B and C implemented their owned components; coordinator implemented A, integration and documentation; independent Eval reviewed metrics/data and Researcher reviewed the newly downloaded assets. Earlier setup snapshots below remain historical.

The final integrated run passed 40 unit tests plus 5 independent checks, deterministic byte reproduction and actual provisional A/zero scoring. See `coordination/execution/run_summary.json` and saved logs. Implementations, 1,100 synthetic bases and 1,650 exact intensity pairs exist. A is perfect on the controlled synthetic grammar; B/C have not been pretrained-fitted. No manual review, human-authored test corpus or official comparison exists. The 1,100-row paraphrase generation queue and blank review/annotation forms are ready.

The user confirmed human collection/review is pending and subsequently added CelebA-Dialog HQ and FLAME texture assets. Read-only inspection found 30,000 images, 29,999 annotations per text JSON (missing 5380.jpg), 6,909 smiling requests, and texture data without a located FLAME geometry model. The inventory and research notes document fit to the project; no external records became custom gold or Test-Human.

Next: generate paraphrases and complete genuine review/collection/annotation, release G2 data, obtain the pinned weights, run B/C Train/Val fitting and frozen official evaluation. The November 25 milestone status and missing results are explicit in `reports/MILESTONE_STATUS_2026-11-23.md`.

Dates use America/Vancouver. This log records evidence-backed events; plans and hypotheses remain labeled.

| Date | Owner | Event and status | Evidence | Next action |
|---|---|---|---|---|
| 2026-11-23 | Coordinator | All seven role sessions created; workflow integration in progress; detailed PDF selected as protocol | `coordination/CONTRACT.md`, `WORKFLOW.md` | Integrate seven role outputs and final setup status |
| 2026-11-23 | Coordinator | Requested destination write denied; staged workspace used | `coordination/DECISIONS.md`; coordinator-reported Access denied | Deliver staged package; destination save remains blocked |
| 2026-11-23 | Agent A | Rule-based design and draft vocabulary prepared; implementation pending | `models/model_a/KICKOFF.md`, `lexicon_draft.json` | Freeze annotation policies, then implement |
| 2026-11-23 | Agent B | Encoder/head/loss/comparability design prepared; training pending | `models/model_b/KICKOFF.md`, `config.draft.json` | Freeze data/protocol and discover compute |
| 2026-11-23 | Doc | Charter, kickoff, log and evidence template prepared | `agents/doc.md`, `docs/` | Index remaining finalized outputs after integration |

**Historical concurrent snapshot (superseded by final integration below):** Agent A and Agent B completed kickoff; Researcher, Data Builder, Agent C and Eval were still working at this snapshot. Doc has prepared its own setup files. The staged package is intended to contain root contents for extraction directly into the requested folder; extraction has not occurred.

## Evidence boundary at the historical snapshot

No completed human collection/manual review, frozen dataset, executable model, training run, checkpoint or evaluation result has been inspected by Doc. Other roles are writing concurrently; this snapshot must not replace the coordinator's final stage status. Earlier dates in the proposal describe planned work, not proven progress.

## Next log entry

Record date, owner, concrete artifact/version, stage-specific status, check evidence, remaining limitation and next owner. When a decision changes labels, splits or training, preserve the prior decision and its downstream impact. Do not retrospectively alter a protocol to fit held-out results.



## Final workflow integration — November 23, 2026

Coordinator confirms all seven role kickoff outputs are complete. `coordination/STATUS.json` records G0 workflow setup complete, with G1–G7 pending and project execution not started. The final integration supersedes the earlier concurrent-work snapshot above.

Coordinator validation covered seven charters, seven kickoff files, four JSON drafts and two illustrative JSONL records. These checks establish package structure and draft record validity, not dataset completion or model correctness. Eval reviewed the metric specification; metric implementation/code validation remains pending. No experiments, training, human collection or manual paraphrase verification are complete.

Pair evidence has distinct conventions: exact template-generated intensity pairs may be `ast-verified` using retained canonical AST/compositional gold-label provenance. Primary paraphrase consistency pairs require actual `human-verified` or `human-adjudicated` semantic review evidence. Human-authored/paraphrased intensity comparisons require corresponding semantic review. See `data/specs/GRAMMAR_AND_SPLITS.md` and the evaluation protocol; none of these statuses are implied merely by preparing a manifest field.

The requested destination remains write-denied. Coordinator archive packaging follows this integration entry; the archive has not been verified by Doc and destination saving has not succeeded. Next: package and check the staged deliverable, then begin G1 specification freeze with the relevant owners.
