"""AI-only review and synthetic diagnostics. Never creates human evidence."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data/ai_challenge_v1/authored_pairs.json'
DEST = ROOT/'experiments/ai_validation_v1'


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def save_rows(path, values):
    Path(path).write_text(''.join(json.dumps(r)+'\n' for r in values), encoding='utf-8')


def norm(text):
    return ' '.join(''.join(c.lower() if c.isalnum() else ' ' for c in text).split())


def signature(row):
    return tuple(row['y']), tuple(row['preserved_mask'])


def prepare():
    from data.import_paraphrase_reviews import verify_inputs
    from models.model_b.data import validate_rows
    verify_inputs()
    if DEST.exists():
        raise ValueError('Use a new experiment version; do not overwrite frozen inputs')
    base_splits = {split: rows(ROOT/f'data/candidates/v1/{name}.jsonl') for split, name in
                   [('Train','train'), ('Val','val'), ('Test-ID','test_id'), ('Test-Comp','test_comp')]}
    bases = {r['id']: r for group in base_splits.values() for r in group}
    paras = rows(ROOT/'data/paraphrases/v1/candidates.jsonl')
    pairs = rows(ROOT/'data/paraphrases/v1/paraphrase_pairs.jsonl')
    pair_by_right = {p['right_id']: p for p in pairs}
    structural = []
    for split, base in base_splits.items():
        validate_rows(base, split)
        validate_rows([r for r in paras if r['split'] == split], split, allow_generated=True)
    for candidate in paras:
        base = bases[candidate['provenance']['base_id']]
        pair = pair_by_right[candidate['id']]
        keys = ('split','y','modified_mask','preserved_mask','semantic_group_id','paraphrase_group_id')
        if any(base[k] != candidate[k] for k in keys) or pair['left_id'] != base['id']:
            raise ValueError('Original paraphrase structural mismatch')
        if candidate['human_review_id'] is not None or pair['human_review_id'] is not None:
            raise ValueError('Unexpected review evidence; recheck current data')
        structural.append(dict(id=candidate['id'], pair_id=pair['pair_id'], status='structural_checks_pass',
                               meaning_verified=False, human_verified=False))
    # These decisions record the assistant's actual reading of the displayed sample,
    # not classifier output or a synthetic human review.
    flags = {
        'Train/22': 'Shared maximally modifier may attach only to mouth widening rather than also cheek raising.',
        'Train/25': 'Trailing as much as possible may attach only to smile reduction rather than also eye closing.',
        'Val/3': 'Adds then, a temporal connector absent from the original. Final targets agree under static semantics; sequencing is outside the frozen task.',
        'Train-extra/22': 'Shared to the maximum modifier across cheek raising and mouth widening merits scope confirmation.',
        'Train-extra/35': 'Maximally precedes coordinated brow lowering and eye closing; repeat degree for each to remove scope doubt.',
        'Train-extra/39': 'Maximally precedes coordinated brow raising and smile reduction; scope should be confirmed.'}
    selections = []
    for split, count in [('Train',40), ('Val',20), ('Test-ID',30), ('Test-Comp',30)]:
        group = [r for r in paras if r['split'] == split]
        for i in range(count):
            selections.append((f'{split}/{i}', group[i*len(group)//count]))
    train_paras = [r for r in paras if r['split'] == 'Train']
    for i in range(40):
        if i % 4:
            selections.append((f'Train-extra/{i}', train_paras[20*i+i%4]))
    reviews = []
    for key, candidate in selections:
        base = bases[candidate['provenance']['base_id']]
        reviews.append(dict(sample_key=key, base_id=base['id'], paraphrase_id=candidate['id'],
            split=candidate['split'], original_text=base['text'], candidate_text=candidate['text'],
            proposed_y=base['y'], proposed_preserved_mask=base['preserved_mask'],
            review_type='AI_semantic_review', reviewer='Codex assistant', reviewer_model_revision='not_exposed',
            human_review_id=None, human_verified=False,
            assessment='needs_human_attention' if key in flags else 'no_definite_meaning_mismatch_identified_AI_only',
            rationale=flags.get(key, 'Compared attributes, edit direction, degree, explicit preservation and scope; no definite mismatch identified under the frozen static-control conventions.')))
    if len(reviews) != 150 or len({r['paraphrase_id'] for r in reviews}) != 150:
        raise ValueError('Expected 150 distinct inspected candidates')
    authored = load(SOURCE)
    if len(authored) != 75:
        raise ValueError('Expected 75 directly authored instruction pairs')
    challenge, challenge_pairs = [], []
    for i, (category, edits, preserved, texts) in enumerate(authored, 1):
        if len(texts) != 2 or any(not t.strip() for t in texts):
            raise ValueError('Two nonempty direct-authored texts per intent required')
        values = [float(edits.get(str(j), 0)) for j in range(6)]
        if any(abs(v) not in (0,.25,.5,.75,1.) for v in values):
            raise ValueError('Unknown intensity')
        if set(map(int, edits)) & set(preserved):
            raise ValueError('Edit and preservation conflict')
        group = f'ai-intent-{i:03}'
        for variant, text in zip(('a','b'), texts):
            challenge.append(dict(id=f'ai-challenge-{i:03}-{variant}', text=text, y=values,
                modified_mask=[int(v != 0) for v in values], preserved_mask=[int(j in preserved) for j in range(6)],
                semantic_group_id=group, paraphrase_group_id=group, template_family_id=None,
                composition_key=','.join(str(j) for j,v in enumerate(values) if v), split='Test-AI',
                source='llm_synthetic', schema_version='1.0', review_status='AI-authored-unverified',
                human_review_id=None, category=category,
                provenance=dict(author='Codex assistant', labeler='same_author', independent_annotation=False,
                    model_revision='not_exposed', method='directly_authored_literal_text_and_proposed_labels',
                    source_file=str(SOURCE.relative_to(ROOT)), source_pair_index=i,
                    not_human_data=True, purpose='post_training_synthetic_lexical_and_scope_stress_test')))
        challenge_pairs.append(dict(pair_id=f'ai-pair-{i:03}', split='Test-AI',
            left_id=f'ai-challenge-{i:03}-a', right_id=f'ai-challenge-{i:03}-b',
            semantic_group_id=group, verification_status='generated', human_review_id=None))
    if len({norm(r['text']) for r in challenge}) != 150:
        raise ValueError('Duplicate text within challenge')
    old_rows = list(bases.values()) + paras
    old_text = {norm(r['text']):r['id'] for r in old_rows}
    collisions = [dict(id=r['id'], existing=old_text[norm(r['text'])], text=r['text'])
                  for r in challenge if norm(r['text']) in old_text]
    if collisions:
        print(json.dumps({'exact_normalized_collisions':collisions}, indent=2))
        raise ValueError('Rewrite exact collisions before freezing; no inference has run')
    train_val = [r for r in old_rows if r['split'] in ('Train','Val')]
    train_signatures = {signature(r) for r in train_val}
    audit = dict(status='pass', challenge_rows=150, intent_pairs=75,
        exact_normalized_text_collisions=0, internal_text_duplicates=0,
        rows_with_existing_train_val_label_signature=sum(signature(r) in train_signatures for r in challenge),
        semantic_disjointness_claim=False,
        semantic_overlap_note='A lexical/scope diagnostic; identical label signatures may occur in Train/Val. New IDs do not establish new semantic groups.',
        categories=dict(Counter(r['category'] for r in challenge)),
        modified_attribute_counts=[sum(r['modified_mask'][j] for r in challenge) for j in range(6)],
        explicit_preservation_targets=sum(sum(r['preserved_mask']) for r in challenge))
    DEST.mkdir(parents=True)
    source_hashes = {str(SOURCE.relative_to(ROOT)):digest(SOURCE)}
    for path in [ROOT/'data/paraphrases/v1/candidates.jsonl', ROOT/'data/paraphrases/v1/paraphrase_pairs.jsonl',
                 *(ROOT/f'data/candidates/v1/{name}.jsonl' for name in ('train','val','test_id','test_comp')),
                 ROOT/'data/candidates/v1/intensity_pairs.jsonl', ROOT/'data/candidates/v1/ast_verification.jsonl']:
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
    save_rows(DEST/'structural_checks.jsonl', structural)
    save_rows(DEST/'ai_semantic_reviews.jsonl', reviews)
    save_rows(DEST/'challenge.jsonl', challenge)
    save_rows(DEST/'challenge_pairs.jsonl', challenge_pairs)
    save(DEST/'audit.json', audit)
    training = load(ROOT/'experiments/provisional_v1/summary.json')
    selected_runs = [r for r in training['runs'] if r['lambda_value'] in (0., training['selected_c_lambda'])]
    plan = dict(status='frozen_before_inference', official_evaluation=False, human_review_complete=False,
        test_human_collected=False, created_at_utc=datetime.now(timezone.utc).isoformat(),
        selected_c_lambda=training['selected_c_lambda'], checkpoints=selected_runs,
        training_summary_sha256=digest(ROOT/'experiments/provisional_v1/summary.json'),
        source_hashes=source_hashes,
        input_sha256={name:digest(DEST/name) for name in ('challenge.jsonl','challenge_pairs.jsonl','ai_semantic_reviews.jsonl','audit.json')},
        code_sha256={p:digest(ROOT/p) for p in ('run_ai_validation.py','evaluation/metrics.py','models/model_a/model.py','models/model_b/model.py')},
        structural_screen_count=len(structural), ai_semantic_review_count=len(reviews),
        ai_flagged_count=len(flags), uninspected_semantic_count=len(paras)-len(reviews),
        review_sampling='40 Train/20 Val/30 Test-ID/30 Test-Comp at floor(i*n/count), plus 30 Train at 20*i+i%4 where i%4 != 0. Extra sample remedies first-intensity bias of the initial Train stride.',
        evaluation_policy='A plus B and selected C at all three frozen seeds; no fitting, threshold tuning, lambda selection or post-result text/label editing.',
        reporting_policy='Existing Test-ID/Test-Comp (bases plus generated paraphrases) and new Test-AI; AI pair distance separately named, official unverified-pair metrics remain null.',
        limitation='Assistant-created and same-author labeled after exposure to project examples; not independently collected or dual-human-annotated Test-Human. Convenience sample QC cannot certify remaining rows.')
    save(DEST/'plan.json', plan)
    print(json.dumps(dict(prepared=True, structural_rows=len(structural), AI_semantic_reviews=len(reviews),
                         flagged=len(flags), challenge=audit), indent=2))


def evaluate_frozen():
    import gc
    import torch
    from transformers import AutoTokenizer
    from evaluation.metrics import evaluate
    from models.model_a.model import predict as predict_a
    from models.model_b.model import load_checkpoint, Predictor
    from models.model_b.train import seed_everything
    plan = load(DEST/'plan.json')
    if (DEST/'results.json').exists():
        raise ValueError('Results already exist; do not overwrite an evaluation')
    for path, expected in {**plan['source_hashes'], **plan['code_sha256']}.items():
        if digest(ROOT/path) != expected:
            raise ValueError('Frozen source changed: '+path)
    for name, expected in plan['input_sha256'].items():
        if digest(DEST/name) != expected:
            raise ValueError('Frozen input changed: '+name)
    if digest(ROOT/'experiments/provisional_v1/summary.json') != plan['training_summary_sha256']:
        raise ValueError('Training selection changed')
    if digest(ROOT/'models/model_a/model.py') != load(ROOT/'coordination/MODEL_A_FREEZE.json')['code_sha256']:
        raise ValueError('Rule baseline changed after freeze')
    paras = rows(ROOT/'data/paraphrases/v1/candidates.jsonl')
    old_pairs = rows(ROOT/'data/paraphrases/v1/paraphrase_pairs.jsonl')
    sets = {'Test-AI':rows(DEST/'challenge.jsonl')}
    pair_sets = {'Test-AI':rows(DEST/'challenge_pairs.jsonl')}
    for split, name in [('Test-ID','test_id'), ('Test-Comp','test_comp')]:
        sets[split] = rows(ROOT/f'data/candidates/v1/{name}.jsonl')+[r for r in paras if r['split']==split]
        pair_sets[split] = [p for p in old_pairs if p['split']==split]
    intensity = rows(ROOT/'data/candidates/v1/intensity_pairs.jsonl')
    evidence = rows(ROOT/'data/candidates/v1/ast_verification.jsonl')
    outputs_dir = DEST/'predictions'
    outputs_dir.mkdir(exist_ok=False)
    results = []
    models = [dict(name='A',seed=None,checkpoint=None)] + [dict(r, name=r['name']) for r in plan['checkpoints']]
    for spec in models:
        if spec['checkpoint']:
            checkpoint = ROOT/spec['checkpoint']
            if digest(checkpoint/'weights.pt') != spec['checkpoint_sha256']:
                raise ValueError('Fitted checkpoint changed')
            seed_everything(spec['seed'])
            model, metadata = load_checkpoint(checkpoint, device='cuda')
            tokenizer = AutoTokenizer.from_pretrained(checkpoint/'tokenizer', local_files_only=True)
            predictor = Predictor(model, tokenizer, metadata, device='cuda')
            predict = predictor.predict
        else:
            predict = predict_a
        for split, gold in sets.items():
            predictions = []
            for pos in range(0,len(gold),4):
                batch = gold[pos:pos+4]
                predictions.extend(dict(p,id=r['id'],split=split) for r,p in zip(batch,predict([r['text'] for r in batch])))
            path = outputs_dir/f'{spec["name"]}_{split}.jsonl'
            save_rows(path,predictions)
            result = evaluate(gold,predictions,split=split,paraphrase_pairs=pair_sets[split],
                              intensity_pairs=[p for p in intensity if p['split']==split],review_evidence=evidence)
            indexed = {p['id']:p for p in predictions}
            distances = [sum((a-b)**2 for a,b in zip(indexed[p['left_id']]['y'],indexed[p['right_id']]['y'])) for p in pair_sets[split]]
            result.update(model=spec['name'],seed=spec['seed'],family=spec['name'][0],official_evaluation=False,
                prediction_sha256=digest(path),unsupported_count=sum(p.get('status')=='unsupported' for p in predictions),
                AI_generated_pair_squared_distance=dict(value=statistics.mean(distances),count=len(distances),human_verified=False))
            result['source_breakdown'] = {}
            for source in sorted({r['source'] for r in gold}):
                subset = [r for r in gold if r['source']==source]
                selected = [indexed[r['id']] for r in subset]
                subresult = evaluate(subset,selected,split=split)
                result['source_breakdown'][source] = dict(n=len(subset),mae=subresult['metrics']['mae']['value'],attribute_f1=subresult['metrics']['attribute_f1']['value'])
            results.append(result)
            print(f'{spec["name"]} {split}: MAE={result["metrics"]["mae"]["value"]:.6f}',flush=True)
        if spec['checkpoint']:
            del predictor,predict,model,tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    aggregates = []
    for split in sets:
        for family in ('A','B','C'):
            items = [r for r in results if r['split']==split and r['family']==family]
            group = dict(split=split,family=family,seeds=len(items),n=items[0]['instruction_count'],metrics={})
            for key in ('attribute_f1','mae','negation_accuracy','compositional_accuracy','intensity_accuracy'):
                values = [r['metrics'][key]['value'] for r in items]
                eligible = [v for v in values if v is not None]
                group['metrics'][key] = dict(mean=statistics.mean(eligible) if eligible else None,
                    sample_std=statistics.stdev(eligible) if len(eligible)>1 else None,
                    eligible_count_per_run=items[0]['metrics'][key]['eligible_count'])
            for key, values in [('AI_pair_squared_distance',[r['AI_generated_pair_squared_distance']['value'] for r in items]),
                                ('all_dimension_tolerance_accuracy',[r['supplementary']['all_dimension_tolerance_accuracy']['value'] for r in items])]:
                group['metrics'][key] = dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None)
            aggregates.append(group)
    result = dict(status='complete',official_evaluation=False,human_reviews_created=0,human_test_instructions_created=0,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),plan_sha256=digest(DEST/'plan.json'),
        model_run_count=7,split_evaluation_count=21,selected_c_lambda=plan['selected_c_lambda'],
        aggregate=aggregates,per_run=results,training_performed=False,
        interpretation='Synthetic diagnostics only, with proposed AI paraphrase labels and same-author Test-AI labels; not a human generalization estimate.')
    save(DEST/'results.json', result)
    lines = ['# AI review and provisional evaluation', '',
        'No human reviews or human-authored test instructions were generated. The original human-review sheets remain pending. This run produces clearly labeled AI assistance and synthetic diagnostics, with the trained models frozen.', '',
        '## Review evidence', '',
        'All 1,100 paraphrases passed structural checks of inherited labels, masks, split/group membership and pair endpoints. These checks do not establish semantic equivalence. The assistant actually read 150 distinct original/paraphrase pairs: 70 Train, 20 Val, 30 Test-ID, 30 Test-Comp. Six were flagged for modifier scope or a temporal connector; no definite label mismatch was identified in this sample. The other 950 rows received structural screening only in this run. This sample is not independent human review or a statistical estimate of corpus-wide correctness.', '',
        'Full texts, sample keys and AI findings: `experiments/ai_validation_v1/ai_semantic_reviews.jsonl`. Earlier bounded AI QC remains a separate historical artifact; sample counts must not be added without deduplication.', '',
        '## New challenge set', '',
        'The assistant directly authored 150 instructions as 75 intended-equivalent pairs and supplied their proposed labels. Categories include single edits, two/three edits, explicit preservation, complement preservation, and preservation-only requests. All six controls are represented. No people were recruited, and there are no independent dual-human annotations. The writer had access to project examples and rules; this is not a blind or natural human-language sample.', '',
        f'The audit found zero exact normalized text duplicates. {load(DEST/"audit.json")["rows_with_existing_train_val_label_signature"]} of 150 rows share their value/preservation signature with existing Train/Val examples. This is a lexical/scope stress test, not a claim of disjoint semantic groups. The original Train/Val/Test split files were not changed.', '',
        '## Frozen-model results', '',
        'B and C use all three fitted seeds. C lambda=0.5 and each model threshold/checkpoint were selected previously on Val. No new fitting or selection occurred. Test-ID and Test-Comp each contain 100 existing AST bases plus 100 generated paraphrases. Test-AI contains the 150 new instructions. Values below are means across seeds (A is deterministic); JSON includes sample standard deviations, per-seed scores, source breakdowns and denominators.', '',
        '| Split | Model | N | Attribute F1 | MAE | Preservation accuracy | AI pair distance |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in aggregates:
        m=g['metrics']
        fmt=lambda x: 'N/A' if x is None else f'{x:.4f}'
        lines.append(f'| {g["split"]} | {g["family"]} | {g["n"]} | {fmt(m["attribute_f1"]["mean"])} | {fmt(m["mae"]["mean"])} | {fmt(m["negation_accuracy"]["mean"])} | {fmt(m["AI_pair_squared_distance"]["mean"])} |')
    lines += ['', 'AI pair distance is mean summed squared prediction difference over generated pairs. It is a supplementary diagnostic, not the official human-verified consistency metric. Unsupported A requests produce its existing zero-vector fallback, so low pair distance or high preservation accuracy must be read beside MAE, F1 and unsupported counts.', '',
        'The standard evaluator leaves official paraphrase consistency unavailable because no pairs have genuine human verification. Test-AI also has no verified intensity pairs. Formal composition accuracy is defined only for Test-Comp; the JSON additionally reports all-dimension tolerance accuracy separately. These are synthetic diagnostics, not evidence of performance on Test-Human.', '',
        '## Human work still pending', '',
        'Use `data/paraphrases/v1/review_*.csv` for actual reviews. The collection and independent annotation procedure is `data/specs/HUMAN_COLLECTION_AND_REVIEW.md`. Real people must write the 150-200 Test-Human instructions without seeing generated examples or using an LLM; both project authors independently annotate them and adjudicate disagreements. AI-authored rows in this experiment must not be relabeled human or imported as genuine reviewer evidence.', '',
        '## Reproduction', '',
        '`run_ai_validation.py --prepare` freezes a new version before inference, and `--evaluate` runs the frozen models. Both refuse to overwrite existing frozen/result artifacts. See `experiments/ai_validation_v1/plan.json`, `audit.json`, `results.json` and `predictions/` for hashes, provenance, results and raw outputs.', '']
    (ROOT/'reports/AI_VALIDATION_2026-11-23.md').write_text('\n'.join(lines),encoding='utf-8')
    status=load(ROOT/'coordination/STATUS.json')
    status.update(ai_validation_report='reports/AI_VALIDATION_2026-11-23.md',
        ai_validation_results='experiments/ai_validation_v1/results.json',
        ai_semantic_review_sample_count=150,ai_challenge_instruction_count=150,
        human_review_completed=False,test_human_collected=False,official_experiments_completed=False)
    save(ROOT/'coordination/STATUS.json',status)
    print('COMPLETE: reports/AI_VALIDATION_2026-11-23.md',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--evaluate',action='store_true')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    if args.evaluate:
        evaluate_frozen()
