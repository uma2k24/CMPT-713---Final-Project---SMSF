"""Inspect local FLAME model metadata and dependency presence before integration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import pickletools
import subprocess
import sys
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'external/python_packages'))


def main():
    asset_root = ROOT.parent / 'FLAME'
    result = {'asset_root': str(asset_root), 'dependencies': {name: bool(importlib.util.find_spec(name))
              for name in ('torch', 'numpy', 'scipy', 'smplx', 'trimesh', 'matplotlib', 'pyrender', 'chumpy')},
              'models': [], 'readmes': []}
    for path in sorted(asset_root.rglob('*.pkl')):
        data = path.read_bytes()
        globals_found = set()
        for op, arg, position in pickletools.genops(data):
            if op.name == 'GLOBAL':
                globals_found.add(arg)
        result['models'].append({'path': str(path.relative_to(ROOT.parent)), 'bytes': len(data),
                                 'sha256': hashlib.sha256(data).hexdigest(), 'pickle_globals': sorted(globals_found)})
    for path in sorted(asset_root.rglob('*.pdf')):
        text = '\n\n'.join(page.extract_text() for page in PdfReader(path).pages)
        label = path.parent.name + '_README.txt'
        (ROOT / 'research' / label).write_text(text, encoding='utf-8')
        result['readmes'].append({'path': str(path.relative_to(ROOT.parent)), 'extracted': 'research/' + label})
    repository = ROOT / 'external/FLAME_PyTorch'
    if (repository / '.git').exists():
        result['repository_commit'] = subprocess.check_output(['git', '-C', str(repository), 'rev-parse', 'HEAD'], text=True).strip()
    (ROOT / 'research/FLAME_PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
