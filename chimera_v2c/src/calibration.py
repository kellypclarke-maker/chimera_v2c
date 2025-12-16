"""
Simple Platt scaling scaffold for v2c probabilities (non-destructive).

This provides a helper to fit a logistic calibration on (p_pred, y_true) pairs,
and to apply it to future probabilities. Intended to be used offline in
scripts/notebooks before wiring into the live pipeline.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple, List


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = max(1e-6, min(1 - 1e-6, p))
    return math.log(p / (1.0 - p))


@dataclass
class PlattScaler:
    a: float
    b: float

    def predict(self, p: float) -> float:
        """
        Apply Platt scaling to a probability p in (0,1).

        We fit a logistic regression on log-odds:
          q = sigmoid(a * logit(p) + b)

        Identity calibration is therefore (a=1, b=0).
        """
        x = _logit(p)
        return _sigmoid(self.a * x + self.b)


def fit_platt(data: Iterable[Tuple[float, int]], lr: float = 0.1, iters: int = 200) -> PlattScaler:
    """
    Fit Platt scaling parameters on (p_pred, y_true) pairs.
    y_true in {0,1}. Simple gradient descent for small datasets.

    Note: p_pred is treated as a probability; we fit on logit(p_pred) so that
    (a=1, b=0) corresponds to identity calibration.
    """
    data_list: List[Tuple[float, int]] = list(data)
    if not data_list:
        return PlattScaler(a=1.0, b=0.0)

    a = 1.0
    b = 0.0
    n = len(data_list)
    for _ in range(iters):
        grad_a = 0.0
        grad_b = 0.0
        for p, y in data_list:
            x = _logit(p)
            q = _sigmoid(a * x + b)
            grad = (q - y)
            grad_a += grad * x
            grad_b += grad
        a -= lr * grad_a / n
        b -= lr * grad_b / n
    return PlattScaler(a=a, b=b)
