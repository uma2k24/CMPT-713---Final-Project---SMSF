# Model A kickoff

Status: configured; implementation not started. This is a baseline design, not an evaluated model.

## Proposed implementation

Use a small dependency-free Python package: `lexicon.json` for editable lexical rules, `model.py` for inference, and `predict.py` for JSONL exports. The candidate lexicon in this directory is a draft, not a frozen annotation policy.

Inference stages:

1. Normalize Unicode, casing and token boundaries while retaining offsets for traces. Match multiword lexical entries before shorter overlaps.
2. Split coordinated/contrastive clauses and identify attribute spans. Resolve shared directions only through documented coordination patterns; never broadcast every modifier across the sentence.
3. Bind attribute-specific direction, local intensity and preservation/negation cues. Use attribute-specific direction entries because verbs such as “raise” or “open” have different targets.
4. Produce per-clause operations: modify(attribute, sign, magnitude), preserve(attribute), or unresolved(reason). “Do not raise the brows” must not become a negative brow edit; “not slightly” must not become strong intensity without a guideline.
5. Merge operations using the frozen conflict policy. Do not invent “last clause wins” or sum conflicting edits without an agreed annotation rule.
6. Emit vector, modified mask and versions. Keep an optional diagnostic trace recording spans, matched entries and unresolved clauses, separate from evaluation-facing output.

`predict(texts: list[str]) -> list[dict]` returns `y`, `modified_mask`, `model_version` and `schema_version`. No artificial probability scores are necessary for the deterministic baseline. The export wrapper adds example ID, split and configuration identity.

## Lexicon examples and boundaries

| Control | Candidate aliases | Positive direction | Negative direction |
| --- | --- | --- | --- |
| eyebrow_elevation | eyebrow, eyebrows, brow, brows | raise, lift | lower |
| eye_openness | eye, eyes, eyelid, eyelids | open, widen | close, narrow |
| mouth_width | mouth width, width of the mouth | widen, broaden | narrow |
| smile_intensity | smile intensity, smile strength | increase, strengthen | decrease, reduce |
| jaw_opening | jaw opening, jaw | open | close |
| cheek_raise | cheek, cheeks | raise, lift | lower |

These aliases are candidates. Bare “smile” and “widen the smile” require Data Builder's fixed interpretation before inclusion; a width request must not silently modify both mouth width and smile intensity. “Widen the eyes” is a different control from “widen the mouth.” Negative semantic directions require an agreed interpretation relative to neutral and need not correspond to physically feasible negative FLAME coefficients.

Proposal-backed magnitudes are “slightly” = 0.25 and “much” = 0.75. Synonyms such as “a little,” a default for unmodified verbs, intermediate levels and “very” remain proposed until annotation freeze. Preservation patterns may include “keep … unchanged,” “leave … as is,” and “do not change …”; exact scope and aliases require the same freeze.

## Development cases to resolve before implementation acceptance

| Instruction | Required behavior |
| --- | --- |
| “Raise the eyebrows slightly.” | First dimension +0.25; only first modified mask entry is 1. |
| “Keep the jaw unchanged.” | All values 0; modified mask all 0; explicit preservation trace targets jaw. |
| “Raise the eyebrows slightly, but keep the jaw unchanged.” | Combine brow edit and jaw preservation without intensity leakage. |
| “Do not raise the eyebrows.” | Follow frozen negation policy; never assume lower eyebrows. |
| “Do not change the jaw; raise the cheeks slightly.” | Negation stays on jaw; cheek edit remains positive. |
| “Raise the brows and lower the cheeks slightly.” | Follow frozen modifier scope; do not assume the modifier targets both. |
| “Widen the smile.” | Follow frozen width-versus-intensity interpretation. |
| “Raise the eyebrows and lower the eyebrows.” | Follow shared contradiction/ambiguity policy with diagnostic evidence. |
| “Make the face happier.” | Follow unsupported/underspecified policy; do not silently invent six edits. |

These are planned behavioral checks, not executed test results or held-out evaluation examples.

## Handoff and acceptance

Wait for G1 specification freeze before implementation labels/rules become authoritative. Once built, provide Agent C and Eval the immutable lexicon/configuration version, runnable inference command and prediction schema. Agent C performs initial Model A evaluation with the common six metrics once G2 data is ready. Eval reviews the metrics and runs the final A/B/C comparison. Report missing evaluation pairs as inapplicable with counts; do not fabricate results.

Expected limitations: vocabulary dependence, long-distance scope, implicit edits, ellipsis, rare paraphrases, ambiguous intensity and novel compositions. Record these as hypotheses until supported by real errors. Do not improve the baseline using held-out failure examples.

## Next action and blockers

Next: agree the full intensity/default scale and negation/conflict/ambiguity policies with Data Builder, then freeze lexicon version 1 and implement inference on Train/Val only. Blocked on that specification, frozen splits and metric readiness. No model code, training, predictions or evaluation results exist yet.
