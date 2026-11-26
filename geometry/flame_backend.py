"""Geometry-only PyTorch FLAME adapter using the upstream SMPL-X LBS routine.

Matches FLAME_PyTorch's shape/expression concatenation and joint order; omits
landmarks, whose separate embeddings are not present. Original assets stay read-only.
"""
import builtins
import hashlib
import math
import pickle
from pathlib import Path
import sys
import numpy as np
import scipy.sparse
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
RINGNET_FLAME2019_SHA256 = 'e7f4f6ac7aa97045b3989893386aa94dc5245ded7105bbbf44116bfdf801a050'
sys.path.insert(0, str(ROOT / 'external/python_packages'))
from smplx.lbs import lbs


class ChumpyArrayState:
    """Read only stored Ch.x arrays, without importing/executing legacy Chumpy."""
    def __setstate__(self, state):
        if not isinstance(state, dict):
            raise ValueError('Unexpected Chumpy state')
        self.state = state


class ModelUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ('__builtin__', 'set'): builtins.set,
            ('builtins', 'set'): builtins.set,
            ('numpy', 'dtype'): np.dtype,
            ('numpy', 'ndarray'): np.ndarray,
            ('numpy.core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('numpy._core.multiarray', '_reconstruct'): np._core.multiarray._reconstruct,
            ('scipy.sparse.csc', 'csc_matrix'): scipy.sparse.csc_matrix,
            ('scipy.sparse._csc', 'csc_matrix'): scipy.sparse.csc_matrix,
            ('chumpy.ch', 'Ch'): ChumpyArrayState,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError(f'Unsupported model global: {module}.{name}')
        return allowed[module, name]


def load_model_arrays(path):
    with Path(path).open('rb') as stream:
        model = ModelUnpickler(stream, encoding='latin1').load()
    if not isinstance(model, dict):
        raise ValueError('Expected FLAME model dictionary')
    arrays = {}
    for key in ('v_template', 'f', 'shapedirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights'):
        value = model[key]
        if isinstance(value, ChumpyArrayState):
            if 'x' not in value.state or not isinstance(value.state['x'], np.ndarray):
                raise ValueError(f'{key}: unsupported computed Chumpy expression')
            value = value.state['x']
        if scipy.sparse.issparse(value):
            value = value.toarray()
        value = np.asarray(value)
        if value.dtype.hasobject or not np.isfinite(value).all():
            raise ValueError(f'{key}: invalid numeric model array')
        arrays[key] = value
    return arrays


class FlameGeometry(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model_path = Path(model_path).resolve()
        self.model_sha256 = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        data = load_model_arrays(self.model_path)
        if data['shapedirs'].shape[2] != 400:
            raise ValueError('Expected 300 shape and 100 expression components')
        if data['kintree_table'].shape != (2, 5):
            raise ValueError('Expected global/neck/jaw/left-eye/right-eye joints')
        self.register_buffer('v_template', torch.tensor(data['v_template'], dtype=torch.float32))
        self.register_buffer('shapedirs', torch.tensor(data['shapedirs'], dtype=torch.float32))
        self.register_buffer('posedirs', torch.tensor(data['posedirs'].reshape(-1, 36).T, dtype=torch.float32))
        self.register_buffer('joint_regressor', torch.tensor(data['J_regressor'], dtype=torch.float32))
        parents = torch.tensor(data['kintree_table'][0].astype(np.int64), dtype=torch.long)
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('weights', torch.tensor(data['weights'], dtype=torch.float32))
        self.register_buffer('faces', torch.tensor(data['f'].astype(np.int64), dtype=torch.long))

    def forward(self, expression=None, jaw_pose=None, shape=None):
        if any(x is not None and (not isinstance(x, torch.Tensor) or x.ndim != 2 or x.shape[0] < 1) for x in (expression, jaw_pose, shape)):
            raise ValueError('FLAME parameters must be nonempty two-dimensional tensors')
        supplied = next((x for x in (expression, jaw_pose, shape) if x is not None), None)
        batch = 1 if supplied is None else supplied.shape[0]
        device, dtype = self.v_template.device, self.v_template.dtype
        values = []
        for name, value, dimensions in [('shape', shape, 300), ('expression', expression, 100), ('jaw_pose', jaw_pose, 3)]:
            if value is None:
                value = torch.zeros(batch, dimensions, device=device, dtype=dtype)
            if value.shape != (batch, dimensions) or not torch.isfinite(value).all():
                raise ValueError(f'{name} must be finite [{batch},{dimensions}]')
            values.append(value.to(device=device, dtype=dtype))
        shape, expression, jaw_pose = values
        zeros = torch.zeros(batch, 3, device=device, dtype=dtype)
        full_pose = torch.cat([zeros, zeros, jaw_pose, zeros, zeros], dim=1)
        vertices, joints = lbs(torch.cat([shape, expression], dim=1), full_pose,
                              self.v_template.unsqueeze(0).expand(batch, -1, -1),
                              self.shapedirs, self.posedirs, self.joint_regressor, self.parents, self.weights)
        return vertices, joints

    def decode_parameter_bundle(self, bundle):
        """External coefficient bundles must name the exact geometry basis used."""
        if bundle.get('flame_model_sha256') != self.model_sha256:
            raise ValueError('FLAME coefficient/model hash mismatch; versions are not interchangeable')
        if bundle.get('parameter_source') == 'RingNet' and self.model_sha256 != RINGNET_FLAME2019_SHA256:
            raise ValueError('RingNet parameter bundles require the pinned FLAME2019 generic model')
        values = {}
        for name in ('expression', 'jaw_pose', 'shape'):
            if name in bundle:
                values[name] = torch.as_tensor(bundle[name], dtype=self.v_template.dtype, device=self.v_template.device)
        if not values:
            raise ValueError('Parameter bundle has no supported parameter arrays')
        return self(**values)


def illustrative_jaw_pose(controls, max_angle=0.35):
    """Partial visualization adapter, not a calibrated six-control geometry map."""
    if isinstance(max_angle, bool) or not isinstance(max_angle, (int, float)) or not math.isfinite(max_angle) or max_angle <= 0:
        raise ValueError('Illustrative maximum angle must be positive and finite')
    if not isinstance(controls, torch.Tensor) or controls.ndim != 2 or controls.shape[0] < 1 or controls.shape[1] != 6 or not torch.isfinite(controls).all():
        raise ValueError('Expected finite [N,6] semantic controls')
    if (controls.abs() > 1).any():
        raise ValueError('Semantic controls must be in [-1,1]')
    if controls[:, [0, 1, 2, 3, 5]].ne(0).any():
        raise ValueError('Five expression controls still need calibration; no silent geometry substitution')
    if controls[:, 4].lt(0).any():
        raise ValueError('Negative jaw edits require a defined non-neutral reference; not supported here')
    zeros = torch.zeros_like(controls[:, 4])
    return torch.stack([max_angle * controls[:, 4], zeros, zeros], dim=1)
