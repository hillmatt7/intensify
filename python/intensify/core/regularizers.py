"""Penalties for regularized maximum likelihood."""

from __future__ import annotations

import numpy as np

from .inference.multivariate_hawkes_mle_params import multivariate_hawkes_extract_alphas


class Regularizer:
    """Base class for additive penalties R(theta) in ``nll + R``."""

    def penalty(self, flat_vector: np.ndarray, M: int) -> float:
        raise NotImplementedError


def _alpha_indices(M: int) -> tuple[list[int], np.ndarray]:
    """Return (indices_in_flat_vector, mask_2d) for the alpha block.

    The intensify multivariate flat layout is
    ``[μ (M), (α_{m,k}, β_{m,k}) interleaved per cell row-major]``.
    Alpha entries live at indices ``M + 2·(m·M + k)`` for m,k in [0, M).
    """
    indices: list[int] = []
    mask = np.zeros((M, M), dtype=float)
    for m in range(M):
        for k in range(M):
            indices.append(M + 2 * (m * M + k))
            mask[m, k] = 1.0
    return indices, mask


class L1(Regularizer):
    """L1 penalty on connection strengths (alpha matrix)."""

    def __init__(self, strength: float = 0.01, *, off_diagonal_only: bool = True):
        self.strength = float(strength)
        self.off_diagonal_only = bool(off_diagonal_only)

    def penalty(self, flat_vector: np.ndarray, M: int) -> float:
        A = multivariate_hawkes_extract_alphas(flat_vector, M)
        if self.off_diagonal_only:
            mask = np.ones_like(A) - np.eye(M)
            return self.strength * float(np.sum(np.abs(A * mask)))
        return self.strength * float(np.sum(np.abs(A)))

    def gradient(self, flat_vector: np.ndarray, M: int) -> np.ndarray:
        """Subgradient of `penalty` w.r.t. the full flat vector. Zero at 0."""
        grad = np.zeros_like(flat_vector)
        indices, mask2d = _alpha_indices(M)
        if self.off_diagonal_only:
            mask2d = mask2d - np.eye(M)
        for m in range(M):
            for k in range(M):
                idx = indices[m * M + k]
                a = float(flat_vector[idx])
                if mask2d[m, k] > 0.0:
                    grad[idx] = self.strength * np.sign(a)
        return grad


class ElasticNet(Regularizer):
    """Elastic net: L1 + L2 on alpha matrix (off-diagonal optional for L1 part)."""

    def __init__(
        self,
        strength: float = 0.01,
        l1_ratio: float = 0.5,
        *,
        off_diagonal_only: bool = True,
    ):
        self.strength = float(strength)
        self.l1_ratio = float(l1_ratio)
        self.off_diagonal_only = bool(off_diagonal_only)

    def penalty(self, flat_vector: np.ndarray, M: int) -> float:
        A = multivariate_hawkes_extract_alphas(flat_vector, M)
        if self.off_diagonal_only:
            mask = np.ones_like(A) - np.eye(M)
            l1 = float(np.sum(np.abs(A * mask)))
            l2 = float(np.sum((A * mask) ** 2) * 0.5)
        else:
            l1 = float(np.sum(np.abs(A)))
            l2 = float(np.sum(A**2) * 0.5)
        return self.strength * (self.l1_ratio * l1 + (1.0 - self.l1_ratio) * l2)

    def gradient(self, flat_vector: np.ndarray, M: int) -> np.ndarray:
        grad = np.zeros_like(flat_vector)
        indices, mask2d = _alpha_indices(M)
        if self.off_diagonal_only:
            mask2d = mask2d - np.eye(M)
        for m in range(M):
            for k in range(M):
                idx = indices[m * M + k]
                a = float(flat_vector[idx])
                if mask2d[m, k] > 0.0:
                    grad[idx] = self.strength * (
                        self.l1_ratio * np.sign(a) + (1.0 - self.l1_ratio) * a
                    )
        return grad
