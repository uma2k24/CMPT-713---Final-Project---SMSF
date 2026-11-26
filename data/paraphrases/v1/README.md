# Human review pack: 1,100 actual LLM paraphrase candidates

One candidate was directly authored by an assistant for every synthetic base: 800 Train, 100 Val, 100 Test-ID and 100 Test-Comp. Raw authored strings, source IDs and authoring notes are retained in `../shards/`. A generic synonym/template program was not used to author these strings. The exact underlying model revision is not exposed by this session and is recorded as unknown, not invented.

`candidates.jsonl` carries proposed inherited labels and generated status. `paraphrase_pairs.jsonl` preserves original split/semantic-group assignments. Structural audits verify coverage, unique IDs, changed text and no normalized cross-meaning/split text collision. They do not prove meaning preservation. No candidate is human-verified or eligible for official fitting yet.

Open the CSV corresponding to the split:

- `review_train.csv`: 800 pairs.
- `review_val.csv`: 100 pairs.
- `review_test_id.csv`: 100 pairs.
- `review_test_comp.csv`: 100 pairs.

Compare original and candidate meaning against `data/specs/FROZEN_V1.json` and the annotation guidelines. Check attribute, direction, modifier magnitude and exactly which features are explicitly preserved. Fill your actual `reviewer_id`, an ISO timestamp with timezone in `reviewed_at`, `outcome` (`approved`, `rejected`, or `needs_adjudication`), `meaning_preserved` (`true` for approval), and notes. Leave unreviewed rows blank. Do not change IDs, source text or proposed labels in these sheets. Wording corrections should be recorded as needs_adjudication with the proposed correction in notes, then issued as a new candidate version and reviewed in its final wording.

An independent AI spot-check inspected 161 pairs and flagged six shared-modifier scope risks in `../AI_QC.md`. This is a priority list for human attention, not completed manual review or a guarantee about other rows.

From the workflow root, validate supplied reviews and export linked evidence into a NEW directory:

```powershell
py -3 -m data.import_paraphrase_reviews --reviews data/paraphrases/v1/review_train.csv data/paraphrases/v1/review_val.csv data/paraphrases/v1/review_test_id.csv data/paraphrases/v1/review_test_comp.csv --output data/review/imports/review_001
```

The importer verifies frozen source hashes, exact text/ID/label linkage, reviewer/date/outcome and explicit approval, then retains all decisions and emits separate row/pair evidence records matching the neural loader. It cannot prove that the recorded reviewer actually performed the review. It never declares G2 released; independent Test-Human collection/dual annotation/adjudication and the full release audit remain required.

Reassembly refuses to overwrite review sheets if ANY content has changed from their original hashes. Preserve the existing reviewed version before issuing corrections. These files remain in a separate candidate directory; the original synthetic base release was not rewritten.
