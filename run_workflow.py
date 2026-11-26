"""Run the authorized local workflow through its real human-data gate.

No pretrained download, neural fit, outreach or invented human review occurs.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import tempfile

ROOT = Path(__file__).resolve().parent


def main():
    evidence_dir = ROOT / 'coordination' / 'execution'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    checks = []

    def run(name, arguments):
        command = [sys.executable, *arguments]
        result = subprocess.run(command, cwd=ROOT, text=True, encoding='utf-8', errors='replace',
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (evidence_dir / f'{name}.log').write_text(result.stdout, encoding='utf-8')
        checks.append({'name': name, 'arguments': arguments, 'exit_code': result.returncode,
                       'log': f'coordination/execution/{name}.log'})
        print(f'{name}: {"PASS" if result.returncode == 0 else "FAIL"}', flush=True)
        if result.returncode:
            raise RuntimeError(f'{name} failed; see its saved log')

    try:
        run('unit_tests', ['-m', 'unittest', 'models.model_a.test_model', 'data.test_generate',
                          'models.model_b.test_model_b', 'evaluation.test_metrics',
                          'models.model_c.test_objective', '-v'])
        # Verify regeneration without modifying delivered candidate records or reviews.
        with tempfile.TemporaryDirectory(prefix='workflow_reproduction_', dir=evidence_dir) as temporary:
            run('regenerate', ['-m', 'data.generate', '--output', temporary])
            regenerated = Path(temporary)
            delivered = ROOT / 'data' / 'candidates' / 'v1'
            mismatches = [p.name for p in regenerated.iterdir() if p.is_file()
                          and (not (delivered / p.name).exists() or p.read_bytes() != (delivered / p.name).read_bytes())]
            if mismatches:
                raise RuntimeError('Candidate regeneration mismatch: ' + ', '.join(mismatches))
            checks.append({'name': 'candidate_byte_reproduction', 'exit_code': 0})
            print('candidate_byte_reproduction: PASS', flush=True)
        run('independent_evaluation_checks', ['-m', 'unittest', 'evaluation.test_independent_review', '-v'])
        run('provisional_diagnostics', ['-m', 'evaluation.run_provisional'])
        status = 'available_stages_complete_human_gate_pending'
        error = None
    except Exception as exc:
        status, error = 'failed', str(exc)
    manifest = ROOT / 'data' / 'candidates' / 'v1' / 'manifest.json'
    summary = {'executed_at_utc': datetime.now(timezone.utc).isoformat(), 'status': status,
               'error': error, 'checks': checks,
               'candidate_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None,
               'official_experiments_completed': False, 'human_reviews_completed': False,
               'test_human_collected': False,
               'next_gate': 'G2: real paraphrase verification, independent human instructions and annotations, released data manifest'}
    (evidence_dir / 'run_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(status, flush=True)
    if error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
