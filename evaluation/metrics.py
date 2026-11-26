"""Six proposal metrics with fail-closed, identity-preserving input validation.

This module uses only the Python standard library. Review evidence is supplied
explicitly, never inferred from a pair's status string. It validates records but
does not itself grant a dataset release or certify human authorship.
"""
from collections import Counter
import math
from numbers import Real

METRIC_VERSION = "1.0.0"
ATTRIBUTES = ("eyebrow_elevation", "eye_openness", "mouth_width",
              "smile_intensity", "jaw_opening", "cheek_raise")
HUMAN_STATUSES = {"human-verified", "human-adjudicated"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _vector(value, name, binary=False):
    _require(isinstance(value, (list, tuple)) and len(value) == 6,
             f"{name} must have exactly six entries")
    for x in value:
        _require(isinstance(x, Real) and not isinstance(x, bool) and math.isfinite(x),
                 f"{name} must contain finite numbers")
        _require(x in (0, 1) if binary else -1 <= x <= 1,
                 f"{name} has invalid values")
    return list(value)


def _index(rows, split, gold):
    result = {}
    for row in rows:
        _require(isinstance(row, dict), "records must be objects")
        key = row.get("id")
        _require(isinstance(key, str) and bool(key), "nonempty string id required")
        _require(key not in result, f"duplicate id: {key}")
        _require(row.get("split") == split, f"wrong split for {key}")
        y = _vector(row.get("y"), f"{key}.y")
        m = _vector(row.get("modified_mask"), f"{key}.modified_mask", True)
        if gold:
            p = _vector(row.get("preserved_mask"), f"{key}.preserved_mask", True)
            for j in range(6):
                _require(not (m[j] and p[j]), f"conflicting masks for {key}")
                _require(bool(m[j]) == (y[j] != 0), f"gold edit/value mismatch for {key}")
        result[key] = row
    return result


def _score(value, count, numerator=None, reason=None, **extra):
    item = {"value": value, "eligible_count": count, "numerator": numerator,
            "reason": reason, "exclusion_count": 0}
    item.update(extra)
    return item


def _pairs(pairs, gold, split, kind):
    seen_ids, seen_endpoints = set(), set()
    for pair in pairs:
        _require(isinstance(pair, dict), "pairs must be objects")
        pid = pair.get("pair_id")
        _require(isinstance(pid, str) and pid and pid not in seen_ids,
                 "pair_id must be nonempty and unique")
        seen_ids.add(pid)
        _require(pair.get("split") == split, f"wrong split for pair {pid}")
        fields = ("left_id", "right_id") if kind == "paraphrase" else ("weak_id", "strong_id")
        a, b = (pair.get(k) for k in fields)
        _require(isinstance(a, str) and isinstance(b, str) and a != b,
                 f"distinct string endpoints required: {pid}")
        _require(a in gold and b in gold, f"missing endpoint: {pid}")
        signature = tuple(sorted((a, b)))
        if kind == "intensity":
            j = pair.get("attribute_index")
            _require(type(j) is int and 0 <= j < 6, f"bad attribute index: {pid}")
            signature += (j,)
        _require(signature not in seen_endpoints, f"duplicate endpoint pair: {pid}")
        seen_endpoints.add(signature)
        yield pair, a, b


def _evidence_map(evidence):
    if evidence is None:
        return {}
    if isinstance(evidence, dict):
        return evidence
    _require(isinstance(evidence, list), "review_evidence must be a mapping or list")
    result = {}
    for entry in evidence:
        key = entry.get("review_id")
        _require(isinstance(key, str) and key and key not in result, "invalid/duplicate review_id")
        result[key] = entry
    return result


def _human_evidence(pair, evidence, key, a, b):
    record = evidence.get(pair.get(key))
    if not isinstance(record, dict):
        return False
    expected_ids = {a, b}
    actual_ids = set(record.get("endpoint_ids", []))
    if not actual_ids:
        actual_ids = {record.get("left_id", record.get("weak_id")),
                      record.get("right_id", record.get("strong_id"))}
    return (record.get("verification_status") == pair.get("verification_status")
            and bool(record.get("reviewer_id")) and bool(record.get("reviewed_at"))
            and record.get("outcome") == "approved"
            and record.get("pair_id") == pair["pair_id"] and actual_ids == expected_ids)


def _ast_labels(row):
    ast = row.get("ast")
    if not isinstance(ast, list):
        return False
    y, m, p = [0.0] * 6, [0] * 6, [0] * 6
    seen = set()
    for node in ast:
        if not isinstance(node, dict):
            return False
        attr = node.get("attribute")
        j = ATTRIBUTES.index(attr) if attr in ATTRIBUTES else attr
        if type(j) is not int or not 0 <= j < 6 or j in seen:
            return False
        seen.add(j)
        if node.get("operation") == "preserve":
            p[j] = 1
        elif node.get("operation") == "edit":
            direction, magnitude = node.get("direction"), node.get("magnitude")
            if direction not in (-1, 1) or magnitude not in (0.25, 0.5, 0.75, 1.0):
                return False
            y[j], m[j] = direction * magnitude, 1
        else:
            return False
    return y == row["y"] and m == row["modified_mask"] and p == row["preserved_mask"]


def _ast_evidence(pair, evidence, left, right):
    record = evidence.get(pair.get("review_id"))
    return (isinstance(record, dict) and record.get("verification_status") == "ast-verified"
            and record.get("weak_id") == left["id"]
            and record.get("strong_id") == right["id"]
            and record.get("attribute_index") == pair["attribute_index"]
            and bool(record.get("generator_version"))
            and left.get("source") == right.get("source") == "template"
            and _ast_labels(left) and _ast_labels(right))


def evaluate(gold_rows, prediction_rows, *, split, paraphrase_pairs=(),
             intensity_pairs=(), review_evidence=None):
    """Return all six metrics for one split, requiring exact prediction coverage.

    Predictions use canonical ``y`` (not y_hat) plus id/split/modified_mask.
    Structurally invalid pairs raise ValueError even if unverified. Semantically
    valid but unverified/evidence-less pairs are excluded and counted. Supplying
    evidence is an auditable assertion by the caller, not proof of its truth.
    """
    _require(isinstance(split, str) and split, "split required")
    gold, pred = _index(gold_rows, split, True), _index(prediction_rows, split, False)
    _require(set(gold) == set(pred), "prediction ids must match gold ids exactly")
    evidence = _evidence_map(review_evidence)
    n = len(gold)
    tp = fp = fn = 0
    absolute_errors, active_errors, comp = [], [], 0
    neg_correct = neg_n = 0
    neg_examples = set()
    for key, row in gold.items():
        prediction = pred[key]
        errors = [abs(a-b) for a, b in zip(row["y"], prediction["y"])]
        absolute_errors.extend(errors)
        comp += all(error < 0.1 for error in errors)
        for j, (m, phat) in enumerate(zip(row["modified_mask"], prediction["modified_mask"])):
            tp += m == phat == 1
            fp += m == 0 and phat == 1
            fn += m == 1 and phat == 0
            if m:
                active_errors.append(errors[j])
            if row["preserved_mask"][j]:
                neg_n += 1
                neg_correct += abs(prediction["y"][j]) < 0.1
                neg_examples.add(key)
    fden = 2*tp + fp + fn
    metrics = {
        "attribute_f1": _score(2*tp/fden if fden else None, n*6, 2*tp,
            None if fden else "no_positive_gold_or_predictions", tp=tp, fp=fp, fn=fn,
            f1_denominator=fden, precision=tp/(tp+fp) if tp+fp else None,
            recall=tp/(tp+fn) if tp+fn else None),
        "mae": _score(math.fsum(absolute_errors)/(n*6) if n else None, n*6,
            math.fsum(absolute_errors), None if n else "empty_split"),
        "compositional_accuracy": _score(comp/n if n and split == "Test-Comp" else None,
            n if split == "Test-Comp" else 0, comp if split == "Test-Comp" else None,
            "defined_for_Test-Comp" if split != "Test-Comp" else (None if n else "empty_split")),
        "negation_accuracy": _score(neg_correct/neg_n if neg_n else None, neg_n,
            neg_correct, None if neg_n else "no_explicit_preservation_targets",
            contributing_instruction_count=len(neg_examples)),
    }
    para_values, para_excluded, para_endpoints = [], Counter(), set()
    for pair, a, b in _pairs(paraphrase_pairs, gold, split, "paraphrase"):
        left, right = gold[a], gold[b]
        _require(all(left[k] == right[k] for k in ("y", "modified_mask", "preserved_mask")),
                 f"paraphrase gold mismatch: {pair['pair_id']}")
        _require(bool(pair.get("semantic_group_id")) and
                 left.get("semantic_group_id") == right.get("semantic_group_id") == pair["semantic_group_id"],
                 f"paraphrase semantic group mismatch: {pair['pair_id']}")
        if pair.get("verification_status") not in HUMAN_STATUSES:
            para_excluded["not_human_verified"] += 1
            continue
        if not _human_evidence(pair, evidence, "human_review_id", a, b):
            para_excluded["missing_or_invalid_human_evidence"] += 1
            continue
        para_values.append(math.fsum((x-y)**2 for x, y in zip(pred[a]["y"], pred[b]["y"])))
        para_endpoints.update((a, b))
    pn = len(para_values)
    metrics["paraphrase_consistency"] = _score(math.fsum(para_values)/pn if pn else None,
        pn, math.fsum(para_values), None if pn else "no_verified_paraphrase_pairs",
        exclusion_count=sum(para_excluded.values()), exclusions=dict(para_excluded),
        contributing_instruction_count=len(para_endpoints), companion_mae=metrics["mae"]["value"])
    intensity_correct = intensity_n = 0
    intensity_excluded, intensity_endpoints, intensity_attributes = Counter(), set(), set()
    for pair, a, b in _pairs(intensity_pairs, gold, split, "intensity"):
        left, right, j = gold[a], gold[b], pair["attribute_index"]
        _require(pair.get("other_controls_equal") is True, "other_controls_equal must be true")
        _require(left["modified_mask"] == right["modified_mask"] and
                 left["preserved_mask"] == right["preserved_mask"] and
                 all(left["y"][k] == right["y"][k] for k in range(6) if k != j),
                 f"intensity pair changes other controls: {pair['pair_id']}")
        _require(left["y"][j] * right["y"][j] > 0 and
                 abs(left["y"][j]) < abs(right["y"][j]), "invalid gold intensity ordering")
        status = pair.get("verification_status")
        valid = (_ast_evidence(pair, evidence, left, right) if status == "ast-verified"
                 else _human_evidence(pair, evidence, "review_id", a, b) if status in HUMAN_STATUSES
                 else False)
        if not valid:
            intensity_excluded["missing_or_invalid_verification_evidence"] += 1
            continue
        intensity_n += 1
        intensity_correct += abs(pred[a]["y"][j]) < abs(pred[b]["y"][j])
        intensity_endpoints.update((a, b))
        intensity_attributes.add(j)
    metrics["intensity_accuracy"] = _score(intensity_correct/intensity_n if intensity_n else None,
        intensity_n, intensity_correct, None if intensity_n else "no_verified_intensity_pairs",
        exclusion_count=sum(intensity_excluded.values()), exclusions=dict(intensity_excluded),
        contributing_instruction_count=len(intensity_endpoints), attribute_indices=sorted(intensity_attributes))
    return {"metric_version": METRIC_VERSION, "split": split, "instruction_count": n,
        "metrics": metrics, "supplementary": {
            "active_attribute_mae": _score(math.fsum(active_errors)/len(active_errors) if active_errors else None,
                len(active_errors), math.fsum(active_errors), None if active_errors else "no_modified_targets"),
            "all_dimension_tolerance_accuracy": _score(comp/n if n else None, n, comp,
                None if n else "empty_split")}}
