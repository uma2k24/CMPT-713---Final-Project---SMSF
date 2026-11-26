import unittest
from geometry.live_engine import jaw_projection,scalar,REFERENCE_JAW


class LiveEngineTests(unittest.TestCase):
    def test_signed_jaw_has_explicit_reference_and_valid_endpoints(self):
        for value,angle in [(-1,0),(0,.175),(1,.35)]:
            result=jaw_projection([0,0,0,0,value,0])
            self.assertAlmostEqual(result['target_jaw_radians'],angle)
            self.assertEqual(result['reference_jaw_radians'],REFERENCE_JAW)
            self.assertFalse(result['calibrated'])

    def test_unmapped_predictions_are_retained_and_explicit(self):
        raw=[.2,-.3,.4,.5,.6,-.7]
        result=jaw_projection(raw)
        self.assertEqual(result['raw_controls'],raw)
        self.assertEqual(result['unrendered_control_indices'],[0,1,2,3,5])
        self.assertEqual(result['rendered_control_indices'],[4])
        self.assertFalse(result['semantic_mapping_complete'])

    def test_invalid_parameters_rejected(self):
        for value in (True,float('nan'),float('inf'),-1,4):
            with self.assertRaises(ValueError):
                scalar(value,0,3,'coefficient')
        with self.assertRaises(ValueError):
            jaw_projection([0]*5)
        with self.assertRaises(ValueError):
            jaw_projection([0,0,0,0,1.1,0])


if __name__=='__main__':
    unittest.main()
