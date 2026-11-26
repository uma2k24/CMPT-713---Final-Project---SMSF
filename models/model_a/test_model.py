import unittest
from models.model_a.model import predict


class RuleTests(unittest.TestCase):
    def test_scope_and_preservation(self):
        p = predict(['slightly raise the brows, widen the mouth much, but keep the jaw unchanged'])[0]
        self.assertEqual(p['y'], [.25, 0, .75, 0, 0, 0])
        self.assertEqual(p['modified_mask'], [1, 0, 1, 0, 0, 0])
        self.assertEqual(p['preserved_mask'], [0, 0, 0, 0, 1, 0])

    def test_negative_and_default(self):
        p = predict(['close the eyes a bit and lower the cheeks'])[0]
        self.assertEqual(p['y'], [0, -.25, 0, 0, 0, -.5])

    def test_complement_order(self):
        a, b = predict(['raise the brows and keep all other features unchanged',
                        'keep all other features unchanged while raise the brows'])
        self.assertEqual(a['preserved_mask'], [0, 1, 1, 1, 1, 1])
        self.assertEqual(a['y'], b['y'])
        self.assertEqual(a['preserved_mask'], b['preserved_mask'])

    def test_ambiguous_negation_is_not_reversed(self):
        for text in ['do not raise the brows', 'make the smile wider', 'raise the left eyebrow',
                     'raise the brows and lower the brows', 'raise the brows and keep the brows unchanged',
                     'slightly raise the brows much', 'raise the brows twice as much']:
            with self.subTest(text=text):
                p = predict([text])[0]
                self.assertEqual(p['status'], 'unsupported')
                self.assertEqual(p['y'], [0] * 6)

    def test_redundancy_and_explicit_negation(self):
        p = predict(["Please raise the eyebrows and raise the eyebrows while don't change the smile."])[0]
        self.assertEqual(p['y'], [.5, 0, 0, 0, 0, 0])
        self.assertEqual(p['preserved_mask'][3], 1)

    def test_batch_deterministic(self):
        texts = ['maximally strengthen the smile', '', 'relax the cheeks somewhat']
        self.assertEqual(predict(texts), predict(texts))


if __name__ == '__main__':
    unittest.main()
