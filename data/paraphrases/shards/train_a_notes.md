# Train A paraphrase authoring

Date: 2026-11-23. Role: session assistant worker `/root/paraphrase_train_a` acting as Data Builder. Exact backend model revision was not exposed and is not asserted.

Authored 400 individual output strings directly from the 400 source instructions in `train_a_source.jsonl`, in four batches of 100. No generic synonym-replacement program, template expansion, external generation API, or paid generation service produced these strings. Source IDs/order are retained. Wording varies clause ordering, noun/verb phrasing, and explicit modifier placement; intended magnitude and preservation scope follow the frozen operational conventions.

PowerShell JSON parsing/identity checks passed: 400 source and 400 generated rows, zero missing/extra IDs, zero ordering differences, zero unchanged source texts, and zero duplicate output texts. Two awkward phrases were subsequently reworded without changing IDs or intended meaning.

These are generated LLM candidates. This authoring pass is not independent review, human verification, or proof of semantic equivalence. No reviewer identity/date, human-review claim, verified pair status, or released gold annotation was added. Every candidate still needs genuine meaning review before official training or scoring. No model predictions or held-out labels were used to tune wording; this shard contains Train sources only.
