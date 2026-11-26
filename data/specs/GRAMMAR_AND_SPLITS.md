# Template grammar, record contract and split design — DRAFT v0.1

## Grammar and exact labels
Generate a semantic abstract syntax tree (AST) first, render language second. A base is one AST with a unique semantic label bundle; alternative strings are variants, not additional independent bases.

```ebnf
instruction = clause, { conjunction, clause } ;
clause = edit | preserve ;
edit = [ polite_prefix ], directed_edit, [ intensity ] ;
preserve = "keep", attribute_phrase, "unchanged"
         | "do not change", attribute_phrase ;
conjunction = " and " | ", then " | ", but " | " while " ;
polite_prefix = "please " ;
```

Directed-edit productions must be attribute-specific, for example raise/lower the eyebrows; open/close the eyes; widen/narrow the mouth; strengthen/reduce the smile; open/close the jaw; raise/lower the cheeks. Intensity placement must be grammatical (raise the eyebrows slightly; slightly raise the eyebrows); do not attach a modifier to the wrong clause. The then conjunction expresses order only for distinct independent attributes; stateful sequences on one attribute are excluded.

Canonical AST nodes: {attribute, operation: edit|preserve, direction: -1|1|null, magnitude: number|null}. Reject duplicate conflicting targets before rendering. Initialize three six-dimensional arrays to zero; each edit fills signed magnitude and modified_mask=1; each preservation fills preserved_mask=1. No parser or learned model generates synthetic gold labels.

The semantic signature includes all edits AND explicit preservation targets, not merely y. Two instructions with identical y but different preservation cues are not the same evaluation-label bundle. Stable semantic_group_id derives from the canonical signature. paraphrase_group_id links variants; meaning-equivalent groups must not cross splits even if generated through different templates.

## JSONL record
Required fields: id, text, y[6], modified_mask[6], preserved_mask[6], semantic_group_id, paraphrase_group_id, template_family_id (null for human), composition_key, source, split, schema_version, review_status. Keep provenance (generator/prompt/model/version/seed, reviewer/date, template version) and an optional canonical AST. Examples are illustrations in EXAMPLE_RECORDS.jsonl and are excluded from any training/test release.

Proposed source values: template, llm_paraphrase, human. Review states: generated, AI-reviewed, human-verified, human-adjudicated. Needs-adjudication/rejected are held in review manifests rather than released as gold. The original generated state cannot be overwritten without retaining review history.

Pair manifests:
- Paraphrase: pair_id, split, left_id, right_id, semantic_group_id, verification_status, human_review_id.
- Intensity: pair_id, split, weak_id, strong_id, attribute_index, verification_status, other_controls_equal, review_id.
For primary paraphrase evaluation, verification_status must be human-verified or human-adjudicated, with real reviewer evidence linked by human_review_id. AI review alone is ineligible. For exact template-generated intensity comparisons, verification may use the canonical AST and compositional gold-label evidence; record verification_status=ast-verified and link review_id to that provenance/audit record rather than requiring an unnecessary human review of exact gold. Human-authored or paraphrased intensity comparisons need the corresponding semantic review evidence. These eligibility rules are part of the metric input contract and must match evaluation specifications.

Eligible intensity pairs share target direction and all other edits/preservation; only the specified attribute magnitude differs. Construct such pairs deliberately; do not infer pairs between unrelated test rows. Pair endpoints share one split. Test-Human has no paraphrase/intensity pairs by default; additions require separately collected/verified evidence.

## Split policy and counts
| Split | Synthetic base target | Extra text variants | Purpose |
|---|---:|---|---|
| Train | 800 | Separately counted verified paraphrases | Fitting |
| Val | 100 | Separately counted verified paraphrases | Tuning only |
| Test-ID | 100 | Separately counted verified paraphrases | Familiar combination evaluation |
| Test-Comp | 100 | Separately counted verified paraphrases | Held-out attribute combinations |
| Test-Human | 0 | 150–200 actual independently authored instructions | Human language evaluation |

The total synthetic target is 1,100 bases, not exactly 1,000; this follows the proposal table. Final manifest must state base_count, variant_count, pair_count and final_row_count per split. A default one paraphrase per base is a planning option, not completed generation.

Proposed composition_key: sorted SET of modified attribute indices; explicit preserved attributes form a separate preservation_key. Hold out a chosen list of multi-attribute modified sets from both Train and Val; all individual attributes, directions and levels must occur in Train. To prevent trivial exposure through larger sets, proposed stronger constraint prohibits a training/tuning modified set containing any held-out set as a subset. Freeze actual held-out keys and this containment policy before generation. Audit preservation combinations separately so the generalization claim matches what was withheld.

Within allowed Train/Val/Test-ID compositions, allocate whole semantic groups reproducibly and stratify attribute/direction/intensity/preservation/clause count coverage. Related weak/strong intensity-pair groups must be assigned as an allocation unit if pairs are to be scored; this supplements semantic-group allocation. Do not hold out an entire primitive by accident. Template families may overlap for Test-ID; report actual overlap, and use separate optional family-held-out analysis if desired.

## Release audits and handoff
- Exact ordered dimensions, finite ranges, binary masks, modified and preserved masks mutually exclusive; preserved and unmentioned values are zero.
- No duplicate/near-duplicate text across splits and no semantic/paraphrase-group overlap. Do not equate distinct vectors with distinct semantics without checking AST/provenance.
- Frozen held-out composition and containment rules pass; primitive coverage is recorded.
- Pair endpoint existence/same split/label equivalence and explicit verification_status/provenance; intensity ordering differs only in target magnitude. Primary paraphrase metrics admit only human-verified or human-adjudicated pairs; exact template intensity pairs may be ast-verified against retained canonical gold evidence.
- Paraphrase human-review evidence complete; human authorship and two blind annotations/adjudication evidence complete.
- Record file hashes, schema/guideline versions, seed, rejection/exclusion counts, coverage and denominator tables.
- Fit/tune data go to Agent A/B/C as appropriate; Eval controls final held-out evaluation labels and test-pair manifests. Full labels are not used to tune lexicons or hyperparameters after freeze.

Status: design draft only. No generator has run and no data splits are built.

