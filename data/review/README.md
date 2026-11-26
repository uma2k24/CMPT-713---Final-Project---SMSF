# Real review and collection work remaining

## Current handoff

All 1,100 paraphrases have now been authored and packaged at `data/paraphrases/v1/`. Start with that directory's README and populated review CSVs. The old blank generation queue below is retained as the initial setup artifact. `data/import_paraphrase_reviews.py` now validates completed sheets and exports linked evidence without declaring a full G2 release. Human collection, independent author annotation and adjudication forms in this directory remain blank and ready to use.

These CSVs contain headers only because no human instructions or reviews exist. They are input templates, not evidence of completed work. Do not fill reviewer names or dates unless the named human actually performed the review. User confirmation: no human data or reviews have been collected.

1. `paraphrase_generation_queue.csv` lists each synthetic base and leaves paraphrase text and generator provenance blank. Produce an actual LLM paraphrase for each base, preserve its base ID/split, and record the real model/version/date. Do not label another deterministic template as an LLM paraphrase.
2. Copy candidate pairs into `paraphrase_reviews.csv`. A human compares both meanings against the frozen annotation rules and records reviewer ID, ISO date/time, outcome (`approved`, `rejected`, or `needs_adjudication`), meaning preservation and notes. A correction needs a new review of the corrected text. Empty fields never imply approval. Preserve rejected records and review history.
3. Collect 150-200 independently authored instructions using the participant prompt below. Store pseudonymous provenance and actual consent in `human_collection.csv`. Do not show participants the generator, existing examples or review queue.
4. Give the same collected text independently to the two project authors. Each fills their own `author_a_annotations.csv` or `author_b_annotations.csv` without seeing the other's labels. JSON vector cells must have six values in canonical order. Record ambiguity/unsupported spans rather than guessing.
5. Compute raw annotation agreement before discussion; record disagreements and actual resolution in `adjudications.csv`. Do not discard the original independent annotations. Only approved, adjudicated records enter Test-Human.
6. A future importer must validate evidence, IDs, masks, semantic equivalence, splits and hashes before issuing a release. No automated import/release utility is claimed in this version. Coordinator/Eval must review the completed evidence before G2 release.

## Participant prompt (no template examples)

Write natural-language instructions for changing a digital face. Available global controls are eyebrow height, eye openness, mouth width, smile strength, jaw opening and cheek raising. You may request one or several changes, specify how strong the change should be, and ask for some features to remain unchanged. Use your own words. Do not consult example instructions or generated templates. Please avoid names or other identifying personal information. Submit only if you consent to your pseudonymous text being used in this course project; public release requires separately recorded consent.

## Official release interface for the model loader

Candidate data deliberately fails the neural loader's G2 gate. A future official manifest requires `release_status: released`, `g2_complete: true`, `schema_version: 1.0` (string), `human_review_complete: true`, `test_human_collected: true`, `review_evidence: [{path, sha256}]`, `files: {Train: {path, sha256}, Val: {path, sha256}}`, and `paraphrase_pairs: {path, sha256}`. Evidence paths are relative to the release manifest. The training pair file contains Train/Val only, with real evidence IDs resolving to completed reviews. Test data/pairs remain Eval-owned. Do not turn these flags on merely to bypass the gate.
