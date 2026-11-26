"""Bounded, read-only inspection of downloaded assets; no pickle loading."""
import hashlib
import json
from pathlib import Path
import zipfile
from collections import Counter
from numpy.lib import format as npy_format
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT.parent


def shape(value, depth=0):
    if depth > 2:
        return type(value).__name__
    if isinstance(value, dict):
        keys = list(value)
        return {'type': 'dict', 'length': len(value), 'first_keys': keys[:8],
                'first_value_structure': shape(value[keys[0]], depth + 1) if keys else None}
    if isinstance(value, list):
        return {'type': 'list', 'length': len(value), 'first_value_structure': shape(value[0], depth + 1) if value else None}
    return {'type': type(value).__name__, 'sample': str(value)[:400]}


def main():
    dataset = ASSETS / 'CelebA-Dialog (HQ)'
    texture = ASSETS / 'FLAME' / 'TextureSpace'
    if not texture.is_dir():
        texture = ASSETS / 'TextureSpace'
    result = {'dataset_path': str(dataset), 'texture_path': str(texture),
              'dataset_exists': dataset.is_dir(), 'texture_exists': texture.is_dir(),
              'dataset_text': {}, 'texture_array_headers': [],
              'model_geometry_found': False, 'inspection': 'metadata_only_no_pickle_loading_no_model_execution'}
    if dataset.is_dir():
        result['files_by_extension'] = dict(Counter(p.suffix.lower() for p in dataset.rglob('*') if p.is_file()))
        image_ids = {p.name for p in dataset.rglob('*.jpg')}
        annotation_ids = {}
        for path in sorted((dataset / 'text').glob('*.json')):
            data = path.read_bytes()
            value = json.loads(data)
            if isinstance(value, dict):
                annotation_ids[path.name] = set(value)
            result['dataset_text'][path.name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'structure': shape(value)}
            # One bounded original example reveals attribute schema; never relabel it.
            example = next(iter(value.values())) if isinstance(value, dict) and value else value[0] if isinstance(value, list) and value else None
            result['dataset_text'][path.name]['first_example'] = json.dumps(example, ensure_ascii=False)[:2400]
            if isinstance(value, dict):
                result['dataset_text'][path.name]['image_join'] = {
                    'missing_annotation_ids': sorted(image_ids - set(value)),
                    'annotations_without_image': sorted(set(value) - image_ids)}
            if path.name == 'request_annotated_hq.json':
                result['request_attribute_counts'] = dict(Counter(r.get('attribute') for r in value.values()))
                result['request_mode_counts'] = dict(Counter(r.get('request_mode') for r in value.values()))
                result['smiling_reference_ids'] = sorted(k for k, r in value.items() if r.get('attribute') == 'Smiling')
                result['smiling_reference_note'] = 'Reference IDs only, not six-control gold labels or custom Test-Human.'
        result['annotation_id_sets_equal'] = all(ids == next(iter(annotation_ids.values())) for ids in annotation_ids.values())
        result['classification_files'] = [{'name': p.name, 'bytes': p.stat().st_size,
                                           'first_lines': p.read_text(encoding='utf-8-sig').splitlines()[:3]}
                                          for p in sorted((dataset / 'classification_label').glob('*.txt'))]
    npz = texture / 'FLAME_texture.npz'
    if npz.exists():
        result['texture_bytes'] = npz.stat().st_size
        with zipfile.ZipFile(npz) as archive:
            for item in archive.infolist():
                record = {'name': item.filename, 'uncompressed_bytes': item.file_size, 'crc32': item.CRC}
                if item.filename.endswith('.npy'):
                    with archive.open(item) as stream:
                        version = npy_format.read_magic(stream)
                        if version == (1, 0):
                            dimensions, fortran, dtype = npy_format.read_array_header_1_0(stream)
                        elif version == (2, 0):
                            dimensions, fortran, dtype = npy_format.read_array_header_2_0(stream)
                        else:
                            record['header_version_unsupported'] = list(version)
                            result['texture_array_headers'].append(record)
                            continue
                        record.update(shape=list(dimensions), dtype=str(dtype), fortran_order=fortran)
                result['texture_array_headers'].append(record)
    readme = texture / 'Readme.pdf'
    if readme.exists():
        content = '\n\n'.join(p.extract_text() for p in PdfReader(readme).pages)
        (ROOT / 'research/LOCAL_ASSET_README.txt').write_text(content, encoding='utf-8')
        result['texture_readme_text'] = content
    geometry = [p.relative_to(ASSETS).as_posix() for p in (ASSETS/'FLAME').rglob('*')
                if p.is_file() and p.suffix.lower() in ('.pkl', '.obj') and
                not any(part == '__pycache__' for part in p.parts)]
    result['geometry_candidate_files'] = geometry[:50]
    result['geometry_candidate_count'] = len(geometry)
    result['model_geometry_found'] = bool(geometry)
    output = ROOT / 'research/LOCAL_ASSET_INVENTORY.json'
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'files_by_extension': result.get('files_by_extension'),
                      'text_files': list(result['dataset_text']), 'texture_arrays': result['texture_array_headers'],
                      'geometry_candidate_count': len(geometry)}, indent=2))


if __name__ == '__main__':
    main()
