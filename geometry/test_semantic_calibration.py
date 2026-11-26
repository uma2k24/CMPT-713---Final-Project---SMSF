"""Integration checks using actual FLAME assets and proposed measurement controls."""
import unittest
import numpy as np
import torch
from geometry.flame_backend import FlameGeometry,ROOT
from geometry.semantic_calibration import SemanticCalibration,REFERENCE_JAW


class SemanticCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=FlameGeometry(ROOT.parent/'FLAME/FLAME2019/FLAME2019/generic_model.pkl')
        cls.calibration=SemanticCalibration(cls.model)

    def test_zero_matches_explicit_reference(self):
        mesh,record=self.calibration.decode([0.]*6)
        np.testing.assert_allclose(mesh,self.calibration.reference,atol=1e-6,rtol=0)
        self.assertEqual(record['rendered_control_indices'],list(range(6)))
        self.assertFalse(record['human_perceptual_validation'])

    def test_returned_coefficients_reproduce_mesh_without_vertex_editing(self):
        mesh,record=self.calibration.decode([.25,-.25,.25,.25,.25,-.25])
        with torch.no_grad():
            reconstructed=self.model(expression=torch.tensor([record['expression_coefficients']],dtype=torch.float32),
                jaw_pose=torch.tensor([[record['target_jaw_radians'],0,0]]))[0][0].numpy()
        np.testing.assert_allclose(mesh,reconstructed,atol=1e-7,rtol=0)

    def test_both_sides_and_unfitted_gaps_remain_valid(self):
        for index in range(6):
            for level in (-1.,1.):
                controls=[0.]*6;controls[index]=level
                mesh,_=self.calibration.decode(controls)
                points=self.calibration.landmark_matrix@mesh
                for upper,lower in [(37,41),(38,40),(43,47),(44,46)]:
                    self.assertGreater(points[upper-17,1]-points[lower-17,1],0)
                for upper,lower in [(61,67),(62,66),(63,65)]:
                    self.assertGreater(points[upper-17,1]-points[lower-17,1],.000199)

    def test_reject_invalid_values_and_model_version(self):
        for values in ([0]*5,[True]*6,[0,0,0,0,float('nan'),0],[0,0,0,0,2,0]):
            with self.assertRaises(ValueError):
                self.calibration.decode(values)
        original=self.model.model_sha256
        try:
            self.model.model_sha256='different_basis'
            with self.assertRaises(ValueError):
                SemanticCalibration(self.model)
        finally:
            self.model.model_sha256=original

    def test_excessive_combination_is_not_silently_clipped(self):
        # Recorded rejected deterministic stress case, retained as a guard check.
        values=[.7946349216909612,.8888514927435782,.5411832772981635,.5347243803142037,-.5098273359735201,-.9644207523543094]
        with self.assertRaisesRegex(ValueError,'coefficient envelope'):
            self.calibration.decode(values)


if __name__=='__main__':
    unittest.main()
