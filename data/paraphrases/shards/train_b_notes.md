# Train B paraphrase authoring

Authored 400 candidate utterances directly in this assistant session on 2026-11-23, role `/root/paraphrase_train_b`. The source was `train_b_source.jsonl`; each output has the original base ID and a new text string. No synonym-replacement script, template-expansion code, paid generation service or external text-generation API produced these strings. The exact served model revision is not available to this role and must not be invented in downstream provenance.

Lexical, clause-order and sentence-structure variations preserve the intended operations and scoped intensities. These are LLM-generated candidates, not human-authored data or human-verified paraphrases. Human reviewers must check meaning, especially shared-modifier scopes such as "maximally raise ... and open ..." and coordinated nominal phrases. No source labels or model code were changed and no model predictions were inspected.

Structural check through PowerShell JSON parsing: 400 source rows; 400 output rows; 0 ordered ID mismatches; 400 distinct IDs; 400 distinct text strings; 0 strings identical to their corresponding source. This check establishes structural completeness, not semantic verification. Coordinator owns full cross-shard validation and the review queue.
