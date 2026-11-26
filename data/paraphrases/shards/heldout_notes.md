# Heldout paraphrase authoring notes

Date: 2026-11-23. Authoring role: `/root/paraphrase_heldout`, a Codex assistant worker in the current session. Exact underlying deployment/model revision is not supplied and is not asserted. No external text-generation service was called.

Authored 300 individual candidate texts directly from `heldout_source.jsonl`: 100 Val, 100 Test-ID and 100 Test-Comp. Text strings were individually composed in six batches of 50 and written directly as JSONL patches. No generic synonym replacement or deterministic paraphrase-generation code was used. Repeated phrasing across related intensity instructions is intentional; these candidates are not claimed to be independent human language.

Preserved source IDs and order. Used explicit intensity language for omitted default modifiers (moderate, 0.50), maintained independent smile strength versus mouth width, and preserved only the attributes explicitly preserved in the source. No model predictions, experimental results or tuning decisions were consulted.

Structural check executed with PowerShell: 300 source records, 300 output records, 300 unique IDs, zero ID/order mismatches, zero texts identical to the corresponding source, 300 unique output texts. The one wording refinement after this check removed a redundant intensity phrase without changing its ID or intended meaning.

Review status is **generated, awaiting real human semantic review**. These are LLM-authored paraphrase candidates, not human-authored Test-Human examples and not verified pairs. Structural checks and authoring-time reasoning do not establish human verification. The coordinator owns canonical record assembly, cross-shard audit and the human review queue.
