# Model A v1.0

Implemented by coordinator acting as Agent A, from the saved lexicon and frozen operational annotation rules, before looking at candidate test labels or scores. No model tuning from test examples is authorized.

`model.py` implements full-clause matching, attribute-specific directions, local prefix/suffix intensity, explicit preservation, unambiguous complement preservation and duplicate/conflict handling. Default magnitude is 0.50. Input case/whitespace are normalized. Multiple independent clauses may use and/but/while/then/commas/semicolons.

Unsupported wording, ambiguous negated direction and contradictions reject the entire instruction. The common numeric interface returns a zero-vector fallback with status=unsupported and a reason; evaluation must include these examples and report unsupported coverage, never silently drop them. This is a deterministic baseline limitation, not semantic certainty. It does not implement an open-language parser or native geometry mapping.

Run from the workflow root:

```powershell
py -3 -m unittest models.model_a.test_model -v
py -3 -m models.model_a.model --input data/candidates/v1/test_id.jsonl --output experiments/provisional/model_a_test_id.jsonl
```

The latter is allowed only as explicitly provisional synthetic diagnostic until G2 is released. Official Test-Human and paraphrase robustness remain unavailable. Each prediction carries input ID/split, versions, code hash and rule trace.
