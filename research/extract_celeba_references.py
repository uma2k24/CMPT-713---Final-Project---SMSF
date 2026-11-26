"""Index source smiling requests for supplementary language inspection, not gold."""
import csv
import hashlib
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT.parent/'CelebA-Dialog (HQ)/text/request_annotated_hq.json'
    records = json.loads(source.read_text(encoding='utf-8'))
    rows = [{'image_id': key, **value} for key, value in records.items() if value['attribute'] == 'Smiling']
    target = ROOT/'research/celeba_references'
    target.mkdir(exist_ok=True)
    fields = ['image_id', 'text', 'request_mode', 'attribute', 'attr_label',
              'score_change_direction', 'score_change_value', 'target_score']
    with (target/'smiling_requests.csv').open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    result = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'records': len(rows), 'request_modes': dict(Counter(r['request_mode'] for r in rows)),
              'six_control_labels_generated': False, 'included_in_project_train_or_test': False,
              'purpose': 'Supplementary vocabulary and annotation-design reference; original degree/target fields retained verbatim',
              'limits': 'Source smiling degree is not a signed change from our neutral reference; do not rescale into gold y without a separate validated protocol.'}
    (target/'manifest.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
