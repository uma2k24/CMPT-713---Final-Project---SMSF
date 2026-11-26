# Annotation guidelines — DRAFT v0.1

Status: proposed operational conventions; not frozen. The proposal fixes the six attributes, order, [-1,1] range, zero meaning no change, slightly=0.25 and much=0.75. Remaining choices below need coordinator/project-author agreement.

## Attribute order and meaning
| Index | Key | Positive change | Negative change | Avoid conflation |
|---|---|---|---|---|
| 0 | eyebrow_elevation | raise brows | lower brows | Brow shape/thickness are unsupported. |
| 1 | eye_openness | open eyes more | close/narrow eyelids | Eye size and spacing are unsupported. |
| 2 | mouth_width | widen mouth horizontally | narrow mouth horizontally | Width is separate from smiling/expression. |
| 3 | smile_intensity | strengthen smile | reduce smile | A frown is not automatically a negative smile. |
| 4 | jaw_opening | open jaw more | close jaw | Jaw width and chin shape are unsupported. |
| 5 | cheek_raise | raise cheeks | lower/relax cheeks | Cheek fullness is unsupported. |

These dimensions encode language-level signed edits, not guaranteed physically reachable endpoints from a neutral FLAME mesh. Negative jaw opening/eye openness/smile/cheek raise may encounter a neutral-state boundary. Proposed policy: preserve the signed semantic label for NLP experiments, mark rendering feasibility separately, and never claim a calibrated geometric effect until geometry mapping is validated. The alternative is an explicit non-neutral reference state; adopting it would change the proposal and needs a recorded decision.

## Proposed intensity vocabulary
| Magnitude | Terms | Authority |
|---|---|---|
| 0.25 | slightly, a little, a bit | 0.25 is proposal-backed; lexical aliases proposed. |
| 0.50 | moderately, somewhat; omitted intensity | Proposed middle level and default. |
| 0.75 | much, a lot, considerably | 0.75 is proposal-backed; lexical aliases proposed. |
| 1.00 | maximally, as much as possible | Proposed endpoint; omit from generation until freeze. |

Use sign * magnitude once per target. Do not assign physical units or equate adjective intensity with a numerical percentage without a rule. Unsupported modifiers (barely, very, more or less, twice as much) require adjudication until mapped. Bare comparatives such as wider use the proposed default only after that rule is frozen. Zero is preservation/no change, not a weak edit.

## Scope and masks
- y[j]=0 for an unmentioned attribute; modified_mask[j]=0; preserved_mask[j]=0.
- Explicit preservation (keep the jaw unchanged, do not change the jaw, leave the jaw alone): y[j]=0, modified_mask[j]=0, preserved_mask[j]=1.
- Ordinary directed edit: modified_mask[j]=1, preserved_mask[j]=0, signed nonzero y[j]. Attribute extraction evaluates modification, not every mention.
- A negated direction is not its opposite: do not raise the brows does NOT mean lower the brows. It also does not unambiguously mean preserve all brow movement. Proposed policy: mark needs_adjudication and exclude from released gold until intent is resolved. Do not narrow the mouth similarly does not imply widen it.
- Not too wide, less unhappy, no more than slightly, and double negation are unsupported until an explicit semantic rule is frozen. They cannot be silently converted into preservation cases.
- Explicit preservation affects only its scoped target. Keep the jaw unchanged while widening the mouth modifies mouth width only.
- All other features unchanged preserves the complement of the explicitly modified set only if the complement scope is unambiguous; record expanded preserved_mask and cue provenance.

## Ambiguity and composition
Make the smile wider can refer to mouth width or stronger smiling. Proposed grammar avoids this phrase and uses widen the mouth versus strengthen the smile. Naturally authored ambiguous examples retain raw text and both annotations; clarification/adjudication must establish one defensible label or exclude from scored gold with a reason. Never copy a convenient template label onto ambiguous paraphrases.

Conjunction joins independent edits. Clause order and paraphrases do not change labels. Do not sum repeated instructions, apply last-clause-wins, or saturate/clamp conflicting labels without a frozen rule. Redundant identical clauses collapse to one semantic edit; conflicting directions, different magnitudes on one target, or edit+preserve conflicts are marked needs_adjudication. Sequential/stateful requests and asymmetric left/right edits are outside the current six-global-control scope.

## Annotation record and review
Each annotator independently records y, modified_mask, preserved_mask, unsupported/ambiguous spans, confidence and rationale using the same frozen guidelines. Preserve both raw annotations. Compute exact vector/mask agreement and per-attribute disagreements before discussion; report the denominator and excluded cases. Adjudication records final label, both annotator IDs, adjudicator/date, reason and guideline version. A human review is complete only when a real human performs and records it.

## Pending freeze decisions
Default/modifier scale; endpoint inclusion; signed neutral interpretation; ambiguous smile-width policy; negated-direction treatment; conflict/unsupported exclusion rules; composition-key definition; actual annotator/collection logistics. Agent-generated examples and drafts are not evidence these decisions were approved.
