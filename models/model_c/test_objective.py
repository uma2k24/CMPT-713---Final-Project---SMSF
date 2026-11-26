"""Numerical/autograd tests; these do not train a language model."""
import unittest
import torch
from models.model_c.objective import consistency_loss, total_loss


class ObjectiveTests(unittest.TestCase):
    def test_reduction_and_both_gradients(self):
        left = torch.zeros((2, 6), requires_grad=True)
        right = torch.ones((2, 6), requires_grad=True)
        loss = consistency_loss(left, right)
        self.assertEqual(loss.item(), 6)
        loss.backward()
        self.assertTrue(torch.equal(left.grad, -torch.ones_like(left)))
        self.assertTrue(torch.equal(right.grad, torch.ones_like(right)))

    def test_lambda_zero_is_same_supervised_tensor(self):
        sup = torch.tensor(3.0, requires_grad=True)
        self.assertIs(total_loss(sup), sup)
        left, right = torch.zeros((1, 6)), torch.ones((1, 6))
        self.assertAlmostEqual(total_loss(sup, left, right, 0.5).item(), 6)

    def test_invalid_inputs(self):
        for left, right in [(torch.zeros((0, 6)), torch.zeros((0, 6))),
                            (torch.zeros((1, 5)), torch.zeros((1, 5))),
                            (torch.zeros((1, 6)), torch.zeros((2, 6))),
                            (torch.full((1, 6), float("nan")), torch.zeros((1, 6)))]:
            with self.assertRaises(ValueError):
                consistency_loss(left, right)
        for value in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                total_loss(torch.tensor(0.0), lambda_value=value)
        with self.assertRaises(ValueError):
            total_loss(torch.tensor(0.0), lambda_value=0.1)


if __name__ == "__main__":
    unittest.main()
