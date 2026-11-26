"""Consistency objective shared with the Model B trainer; no private pipeline."""
import math
import torch


def consistency_loss(left_y, right_y):
    """Mean over pairs of SUM over six dimensions; gradients reach both sides."""
    if left_y.ndim != 2 or left_y.shape[-1] != 6 or left_y.shape != right_y.shape:
        raise ValueError("consistency endpoints must have matching [N,6] shapes")
    if left_y.shape[0] == 0:
        raise ValueError("consistency requires at least one verified pair")
    if not torch.isfinite(left_y).all() or not torch.isfinite(right_y).all():
        raise ValueError("consistency endpoints must be finite")
    return (left_y - right_y).square().sum(dim=-1).mean()


def total_loss(supervised, left_y=None, right_y=None, lambda_value=0.0):
    if not isinstance(lambda_value, (int, float)) or not math.isfinite(lambda_value) or lambda_value < 0:
        raise ValueError("lambda must be finite and nonnegative")
    if lambda_value == 0:
        return supervised
    if left_y is None or right_y is None:
        raise ValueError("positive lambda requires verified pair endpoint outputs")
    return supervised + lambda_value * consistency_loss(left_y, right_y)
