# Human collection and paraphrase review — prepared procedure

Status: collection and manual verification PENDING. No people have been contacted. This file is a form/procedure, not collected data. Coordinator/project authors arrange collection and any course-required consent/ethics process; agents do not initiate outreach.

## Test-Human collection prompt (proposed)
We are studying how people describe facial editing requests. In your own words, write a short instruction describing a change you would like made to a digital face, optionally including features that should stay unchanged. Use only these supported controls: eyebrow elevation, eye openness, mouth width, smile strength, jaw opening and cheek raising. You may describe one or several controls and how small or large the change should be. Do not look at system-generated examples or use an LLM to write the instruction. Do not include personal information. If you cannot express an intended change clearly, explain the uncertainty in a separate note.

Show no templates or worked examples. Providing the supported controls is not template exposure; record exactly which prompt/guidance each author saw. Avoid directing everyone to reproduce template intensities or exact combinations. Record author self-report of template/LLM exposure and exclude contaminated examples from the primary Test-Human claim, retaining provenance.

Collection fields: collection_id, pseudonymous_author_id, raw_text, author_note, prompt_version, collection_date, prior_template_exposure, LLM_assisted (self-report), consent_or_course_permission_reference, inclusion_status/reason. Minimize identifying information. Preserve original wording independently of later normalization.

## Independent annotation form
instruction_id; raw_text; annotator_id; guideline_version; annotation_timestamp; y[6]; modified_mask[6]; preserved_mask[6]; ambiguous_spans; unsupported_spans; confidence; rationale; recommended_inclusion.

Each project author fills this independently without viewing the other's labels or model predictions. Compare annotations only after both are submitted. Keep both originals; adjudication records final labels, disagreement categories, rationale, participants and date. Do not relabel based on model output.

## Paraphrase-generation request (for later execution)
Rewrite the supplied editing instruction naturally while preserving every target attribute, edit direction, degree, preservation constraint and scope. Add no edits. Return only the rewritten instruction. If the original is ambiguous, flag it instead of choosing an interpretation.

Supply original text and canonical label/AST in generation provenance. Preserve raw responses, model identifier, prompt version and generation settings. Do not release an instruction just because an LLM states it is equivalent.

## Human paraphrase-verification form
review_id; original_id; paraphrase_id; original_text; candidate_text; reviewer_id; review_date; guideline_version; attribute_match; direction_match; intensity_match; preservation_match; scope_match; fluency; verdict (accept|edit_and_recheck|reject|ambiguous); corrected_text; reason; recheck_record.

A real reviewer compares both texts against the AST/labels. Any semantic mismatch is rejected or edited and checked again. Accepted records carry human-verified status with review evidence. AI checks may flag issues beforehand but retain AI-reviewed status. Do not count an edited candidate as accepted before recheck.

## Completion report
Report number of authors and independent instructions; exclusions; number independently annotated twice; pre-adjudication agreement and disagreement types; adjudicated count; final human-test count; paraphrases reviewed/accepted/edited/rejected; outstanding work. Missing counts remain pending, never zero-completion claims disguised as a finished dataset.
