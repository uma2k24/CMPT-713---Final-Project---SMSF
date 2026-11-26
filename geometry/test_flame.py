"""Independent geometry checks against real local assets and upstream equations."""
import contextlib
import importlib.util
import io
import pickle
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from geometry.flame_backend import FlameGeometry, ModelUnpickler, illustrative_jaw_pose, load_model_arrays, ROOT


class GeometryChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.paths = [ROOT.parent/'FLAME/FLAME2020/FLAME2020/generic_model.pkl',
                     ROOT.parent/'FLAME/FLAME2023/FLAME2023/flame2023.pkl',
                     ROOT.parent/'FLAME/FLAME2019/FLAME2019/generic_model.pkl']
        if not all(path.exists() for path in cls.paths):
            raise unittest.SkipTest('The user-provided FLAME2019/2020/2023 assets are required')
        cls.models = [FlameGeometry(path) for path in cls.paths]

    def test_both_models_neutral_match_and_batch_inference(self):
        for model in self.models:
            with self.subTest(model=model.model_path.name), torch.no_grad():
                neutral, joints = model()
                self.assertEqual(tuple(neutral.shape), (1, 5023, 3))
                self.assertEqual(tuple(model.faces.shape), (9976, 3))
                self.assertEqual(tuple(joints.shape), (1, 5, 3))
                torch.testing.assert_close(neutral[0], model.v_template, rtol=0, atol=1e-6)
                poses = torch.tensor([[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]])
                batch, _ = model(jaw_pose=poses)
                self.assertTrue(torch.isfinite(batch).all())
                self.assertGreater(float((batch[2]-batch[0]).abs().max()), 1e-4)
                for index in range(3):
                    single, _ = model(jaw_pose=poses[index:index+1])
                    torch.testing.assert_close(single[0], batch[index], rtol=1e-5, atol=1e-6)

    def test_expression_and_jaw_gradients_are_finite_and_nonzero(self):
        expression = torch.zeros(1, 100, requires_grad=True)
        jaw = torch.tensor([[.1, 0., 0.]], requires_grad=True)
        vertices, _ = self.models[1](expression=expression, jaw_pose=jaw)
        vertices.square().sum().backward()
        for gradient in (expression.grad, jaw.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_geometry_matches_upstream_flame_pytorch(self):
        """Dummy landmark exists solely to exercise upstream mesh output in this test."""
        code = ROOT/'external/FLAME_PyTorch/flame_pytorch/flame.py'
        if not code.exists():
            self.skipTest('Pinned upstream FLAME_PyTorch checkout is needed for comparison')
        spec = importlib.util.spec_from_file_location('independent_upstream_flame', code)
        upstream = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(upstream)
        arrays = load_model_arrays(self.paths[1])
        with tempfile.TemporaryDirectory(prefix='flame_upstream_fixture_') as temporary:
            directory = Path(temporary)
            with (directory/'arrays.pkl').open('wb') as stream:
                pickle.dump(arrays, stream)
            with (directory/'dummy_landmark.pkl').open('wb') as stream:
                pickle.dump({'lmk_face_idx': np.array([0]), 'lmk_b_coords': np.array([[1., 0., 0.]])}, stream)
            config = SimpleNamespace(flame_model_path=str(directory/'arrays.pkl'), batch_size=2,
                       shape_params=300, expression_params=100, use_face_contour=False,
                       use_3D_translation=False, static_landmark_embedding_path=str(directory/'dummy_landmark.pkl'))
            with contextlib.redirect_stdout(io.StringIO()):
                reference = upstream.FLAME(config)
            shape = torch.zeros(2, 300); shape[:, :2] = torch.tensor([[.1, -.1], [-.05, .15]])
            expression = torch.zeros(2, 100); expression[:, :3] = torch.tensor([[.3, -.1, .05], [-.2, .2, -.1]])
            jaw = torch.tensor([[.08, 0., 0.], [.18, .01, 0.]])
            pose = torch.cat([torch.zeros(2, 3), jaw], dim=1)
            with torch.no_grad():
                actual = self.models[1](shape=shape, expression=expression, jaw_pose=jaw)[0]
                expected = reference(shape_params=shape, expression_params=expression, pose_params=pose)[0]
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_illustrative_mapping_rejects_unsupported_semantics(self):
        controls = torch.zeros(3, 6); controls[:, 4] = torch.tensor([0., .5, 1.])
        pose = illustrative_jaw_pose(controls)
        torch.testing.assert_close(pose[:, 0], torch.tensor([0., .175, .35]))
        self.assertTrue(pose[:, 1:].eq(0).all())
        invalid = []
        for index in (0, 1, 2, 3, 5):
            row = torch.zeros(1, 6); row[0, index] = .25; invalid.append(row)
        for value in (-.25, 1.01, float('nan'), float('inf')):
            row = torch.zeros(1, 6); row[0, 4] = value; invalid.append(row)
        invalid += [torch.zeros(6), torch.zeros(1, 5)]
        for row in invalid:
            with self.subTest(shape=tuple(row.shape)), self.assertRaises(ValueError):
                illustrative_jaw_pose(row)

    def test_mapping_scale_and_forward_inputs_fail_closed(self):
        for scale in (-.1, float('nan'), float('inf')):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                illustrative_jaw_pose(torch.zeros(1, 6), max_angle=scale)
        for expression in (torch.tensor(0.), torch.zeros(100), torch.zeros(0, 100),
                           torch.zeros(1, 99), torch.full((1, 100), float('nan'))):
            with self.subTest(shape=tuple(expression.shape)), self.assertRaises(ValueError):
                self.models[1](expression=expression)

    def test_unapproved_pickle_global_is_rejected_without_execution(self):
        with self.assertRaises(pickle.UnpicklingError):
            ModelUnpickler(io.BytesIO(b'cos\nsystem\n.')).load()

    def test_external_parameter_versions_and_ringnet_are_enforced(self):
        model2023, model2019 = self.models[1], self.models[2]
        bundle = {'parameter_source': 'RingNet', 'flame_model_sha256': model2019.model_sha256,
                  'expression': [[0.] * 100]}
        with torch.no_grad():
            expected = model2019()[0]
            actual = model2019.decode_parameter_bundle(bundle)[0]
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            model2023.decode_parameter_bundle(bundle)
        with self.assertRaises(ValueError):
            model2023.decode_parameter_bundle(dict(bundle, flame_model_sha256=model2023.model_sha256))
        with self.assertRaises(ValueError):
            model2019.decode_parameter_bundle({'expression': [[0.] * 100]})


if __name__ == '__main__':
    unittest.main()
