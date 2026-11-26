"""Frozen, deliberately limited lexicon baseline; no dataset labels at inference."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

ATTRIBUTES = ('eyebrow_elevation', 'eye_openness', 'mouth_width', 'smile_intensity', 'jaw_opening', 'cheek_raise')
MODEL_VERSION = 'rules-1.0'
SCHEMA_VERSION = '1.0'
ALIASES = (
    r'(?:eyebrows?|brows?)', r'(?:eyes?|eyelids?|eye openness)',
    r'(?:mouth|mouth width|width of the mouth)', r'(?:smile|smile intensity|smile strength)',
    r'(?:jaw|jaw opening)', r'(?:cheeks?|cheek raise)',
)
DIRECTIONS = (
    (('raise', 'lift'), ('lower',)), (('open', 'widen'), ('close', 'narrow')),
    (('widen', 'broaden'), ('narrow',)), (('strengthen', 'increase'), ('reduce', 'decrease')),
    (('open',), ('close',)), (('raise', 'lift'), ('lower', 'relax')),
)
INTENSITIES = {'slightly': .25, 'a little': .25, 'a bit': .25,
               'moderately': .5, 'somewhat': .5, 'much': .75,
               'a lot': .75, 'considerably': .75,
               'maximally': 1., 'as much as possible': 1.}
MODIFIER = '|'.join(re.escape(s) for s in sorted(INTENSITIES, key=len, reverse=True))


class UnsupportedInstruction(ValueError):
    pass


def _clause(text):
    text = re.sub(r'^please\s+', '', text.strip())
    if re.fullmatch(r'(?:keep|leave) (?:all )?other (?:features|attributes) (?:unchanged|as is|alone)', text):
        return ('complement', None, 0.)
    for j, alias in enumerate(ALIASES):
        target = rf'(?:the )?{alias}'
        preserve = rf'(?:(?:keep|leave) {target} (?:unchanged|as is|alone)|(?:do not|don\x27t) (?:change|modify) {target})'
        if re.fullmatch(preserve, text):
            return ('preserve', j, 0.)
        for sign, verbs in ((1, DIRECTIONS[j][0]), (-1, DIRECTIONS[j][1])):
            pattern = rf'(?:(?P<pre>{MODIFIER}) )?(?:{"|".join(verbs)}) {target}(?: (?P<post>{MODIFIER}))?'
            match = re.fullmatch(pattern, text)
            if match:
                if match['pre'] and match['post']:
                    raise UnsupportedInstruction('Multiple intensity modifiers in one clause')
                magnitude = INTENSITIES.get(match['pre'] or match['post'], .5)
                return ('edit', j, sign * magnitude)
    raise UnsupportedInstruction(f'Unsupported or ambiguous clause: {text}')


def parse(text):
    if not isinstance(text, str) or not text.strip():
        raise UnsupportedInstruction('Instruction must be nonempty text')
    normalized = re.sub(r'\s+', ' ', text.casefold().replace('\u2019', "'")).strip().rstrip('.!')
    clauses = re.split(r'\s*(?:[,;]|\b(?:and|but|while|then)\b)\s*', normalized)
    assignments, complement, trace = {}, False, []
    for clause in filter(None, clauses):
        operation, j, value = _clause(clause)
        trace.append({'clause': clause, 'operation': operation, 'attribute_index': j, 'value': value})
        if operation == 'complement':
            complement = True
            continue
        assignment = (operation, value)
        if j in assignments and assignments[j] != assignment:
            raise UnsupportedInstruction(f'Conflicting instructions for {ATTRIBUTES[j]}')
        assignments[j] = assignment
    if not trace:
        raise UnsupportedInstruction('No supported clauses')
    y, modified, preserved = [0.] * 6, [0] * 6, [0] * 6
    for j, (operation, value) in assignments.items():
        y[j] = value
        modified[j] = int(operation == 'edit')
        preserved[j] = int(operation == 'preserve')
    if complement:
        preserved = [int(not modified[j]) for j in range(6)]
    return y, modified, preserved, trace


class RuleBasedModel:
    def predict(self, texts):
        results = []
        for text in texts:
            try:
                y, modified, preserved, trace = parse(text)
                status, reason = 'supported', None
            except UnsupportedInstruction as exc:
                y, modified, preserved, trace = [0.] * 6, [0] * 6, [0] * 6, []
                status, reason = 'unsupported', str(exc)
            results.append({'y': y, 'modified_mask': modified, 'preserved_mask': preserved,
                            'model_version': MODEL_VERSION, 'schema_version': SCHEMA_VERSION,
                            'status': status, 'reason': reason, 'trace': trace})
        return results


def predict(texts):
    return RuleBasedModel().predict(texts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate example IDs')
    outputs = predict([row['text'] for row in rows])
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for row, output in zip(rows, outputs):
        output.update(id=row['id'], split=row['split'], code_sha256=fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(''.join(json.dumps(o, ensure_ascii=False) + '\n' for o in outputs), encoding='utf-8')
    print(json.dumps({'examples': len(outputs), 'unsupported': sum(o['status'] != 'supported' for o in outputs), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
