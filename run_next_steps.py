"""Verify new paraphrase, pretrained-resource and FLAME work without fitting models."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from data.import_paraphrase_reviews import verify_inputs
from data.assemble_paraphrases import validate_candidates, read_rows

ROOT = Path(__file__).resolve().parent


def main():
    evidence = ROOT/'coordination/execution_next'
    evidence.mkdir(parents=True, exist_ok=True)
    checks = []
    def run(name, args):
        process = subprocess.run([sys.executable, *args], cwd=ROOT, text=True, encoding='utf-8',
                                 errors='replace', stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (evidence/f'{name}.log').write_text(process.stdout, encoding='utf-8')
        checks.append({'name': name, 'exit_code': process.returncode, 'log': f'coordination/execution_next/{name}.log'})
        print(f'{name}: {"PASS" if process.returncode == 0 else "FAIL"}', flush=True)
        if process.returncode:
            raise RuntimeError(f'{name} failed; inspect its log')
    try:
        run('core_workflow', ['run_workflow.py'])
        run('new_independent_tests', ['-m', 'unittest', 'geometry.test_flame', 'data.test_paraphrase_pipeline', '-v'])
        verify_inputs()
        bases = [row for name in ('train', 'val', 'test_id', 'test_comp')
                 for row in read_rows(ROOT/f'data/candidates/v1/{name}.jsonl')]
        candidates = read_rows(ROOT/'data/paraphrases/v1/candidates.jsonl')
        validate_candidates(bases, [{'base_id': row['provenance']['base_id'], 'text': row['text']} for row in candidates])
        checks.append({'name': 'paraphrase_structure_and_immutable_hashes', 'exit_code': 0})
        with tempfile.TemporaryDirectory(prefix='review_validation_', dir=evidence) as temporary:
            review_output = Path(temporary)/'validated'
            run('review_import_check', ['-m', 'data.import_paraphrase_reviews', '--reviews',
                 *[str(ROOT/f'data/paraphrases/v1/review_{name}.csv') for name in ('train', 'val', 'test_id', 'test_comp')],
                 '--output', str(review_output)])
            progress = json.loads((review_output/'summary.json').read_text(encoding='utf-8'))
            (evidence/'review_progress.json').write_text(json.dumps(progress, indent=2)+'\n', encoding='utf-8')
        run('flame2019_demo', ['-m', 'geometry.run_demo'])
        preflight = json.loads((ROOT/'models/model_b/preflight/report.json').read_text(encoding='utf-8'))
        if preflight['status'] != 'pass' or preflight['optimizer_steps'] != 0 or not Path(preflight['snapshot_path']).is_dir():
            raise RuntimeError('Pinned pretrained resource preflight is missing/invalid')
        checks.append({'name': 'recorded_T5_resource_preflight_and_local_cache', 'exit_code': 0})
        status, error = 'next_automated_steps_pass_human_gate_pending', None
    except Exception as exc:
        status, error = 'failed', str(exc)
    summary = {'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'status': status, 'error': error,
               'checks': checks, 'official_training_performed': False, 'ringnet_inference_performed': False,
               'semantic_geometry_calibration_complete': False}
    (evidence/'run_summary.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(status, flush=True)
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
