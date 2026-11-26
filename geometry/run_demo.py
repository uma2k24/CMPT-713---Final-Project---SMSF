"""Render genuine FLAME geometry checks and explicitly uncalibrated examples."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from geometry.flame_backend import FlameGeometry, illustrative_jaw_pose, ROOT


def save_obj(path, vertices, faces):
    with path.open('w') as stream:
        for vertex in vertices:
            stream.write('v ' + ' '.join(f'{v:.8f}' for v in vertex) + '\n')
        for face in faces:
            stream.write('f ' + ' '.join(str(int(v) + 1) for v in face) + '\n')


def panel(ax, vertices, faces, title, bounds):
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.array([-.4, .5, 1.])
    light /= np.linalg.norm(light)
    brightness = .25 + .75 * np.maximum(0, normals @ light)
    colors = np.array([.68, .77, .86])[None, :] * brightness[:, None]
    order = np.argsort(triangles[:, :, 2].mean(axis=1))
    collection = PolyCollection(triangles[order, :, :2], facecolors=colors[order], edgecolors='none', rasterized=True)
    ax.add_collection(collection)
    ax.set_xlim(bounds[0], bounds[1]); ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title, fontsize=11)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=ROOT.parent/'FLAME/FLAME2019/FLAME2019/generic_model.pkl')
    parser.add_argument('--output', type=Path, default=ROOT/'geometry/output_2019')
    args = parser.parse_args()
    model = FlameGeometry(args.model)
    model_label = next((part for part in args.model.parts if re.fullmatch(r'FLAME\d{4}', part)), 'FLAME custom')
    if 'no_jaw' in args.model.name:
        raise ValueError('This separate-jaw demonstration does not use the no_jaw expression variant')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    controls = torch.zeros(5, 6, device=device)
    controls[:, 4] = torch.tensor([0., .25, .5, .75, 1.], device=device)
    with torch.no_grad():
        vertices, joints = model(jaw_pose=illustrative_jaw_pose(controls))
    neutral_error = float((vertices[0] - model.v_template).abs().max())
    expression = torch.zeros(1, 100, device=device, requires_grad=True)
    expression_vertices, _ = model(expression=expression)
    expression_vertices[:, :, 1].square().sum().backward()
    gradient_finite = bool(torch.isfinite(expression.grad).all())
    gradient_nonzero = bool(expression.grad.abs().sum() > 0)
    if neutral_error > 1e-5 or not gradient_finite or not gradient_nonzero or not torch.isfinite(vertices).all():
        raise RuntimeError('FLAME geometry/gradient checks failed')
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = vertices.cpu().numpy(); faces = model.faces.cpu().numpy()
    for name, item in zip(['neutral', 'jaw_025', 'jaw_050', 'jaw_075', 'jaw_100'], mesh):
        save_obj(args.output / f'{name}.obj', item, faces)
    bounds = (mesh[:, :, 0].min() - .015, mesh[:, :, 0].max() + .015,
              mesh[:, :, 1].min() - .015, mesh[:, :, 1].max() + .015)
    fig, axes = plt.subplots(1, 5, figsize=(14, 4.8), facecolor='white')
    for ax, item, value in zip(axes, mesh, [0., .25, .5, .75, 1.]):
        panel(ax, item, faces, f'Jaw control {value:.2f}', bounds)
    fig.suptitle(f'{model_label} geometry - illustrative jaw mapping', fontsize=17)
    fig.text(.5, .04, '0.35 radians at control 1.0; scale is illustrative, not calibrated. Other five controls are not mapped.', ha='center', fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.13, left=.02, right=.98, wspace=.05)
    fig.savefig(args.output/'jaw_preview.png', dpi=160)
    plt.close(fig)
    # Unlabelled expression basis probes support later calibration without inventing semantics.
    probes = torch.zeros(12, 100, device=device)
    for i in range(6):
        probes[2*i, i] = -1.; probes[2*i+1, i] = 1.
    with torch.no_grad():
        expression_meshes = model(expression=probes)[0].cpu().numpy()
    fig, axes = plt.subplots(2, 6, figsize=(15, 8), facecolor='white')
    for i in range(6):
        for sign_index, sign in enumerate([-1, 1]):
            panel(axes[sign_index, i], expression_meshes[2*i+sign_index], faces,
                  f'Expression PC {i}: {sign:+d}', bounds)
    fig.suptitle(f'{model_label} expression basis - these PCs are not named semantic controls', fontsize=15)
    fig.tight_layout(rect=(0, .02, 1, .94))
    fig.savefig(args.output/'expression_basis_preview.png', dpi=140)
    plt.close(fig)
    result = {'model_path': str(args.model), 'model_label': model_label, 'model_sha256': model.model_sha256, 'device': device,
              'ringnet_predictor_used': False,
              'vertices': list(vertices.shape), 'faces': list(model.faces.shape),
              'neutral_template_max_abs_error': neutral_error, 'expression_gradients_finite': gradient_finite,
              'expression_gradients_nonzero': gradient_nonzero, 'semantic_mapping_complete': False,
              'jaw_mapping': 'illustrative only: positive jaw control times0.35 radians around local x',
              'unmapped_controls': ['eyebrow_elevation','eye_openness','mouth_width','smile_intensity','cheek_raise'],
              'negative_jaw': 'rejected: neutral-state feasibility requires explicit reference',
              'rendering': 'orthographic untextured software rendering; no landmark embeddings used'}
    (args.output/'run.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
