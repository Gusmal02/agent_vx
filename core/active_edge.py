"""
ActiveEdge: arista como transformador de señal con K resonadores cuaterniónicos.

El resonador medio r̄_ij transforma la señal del nodo j antes de transmitirla a i:
    influencia_i = r̄_ij ⊗ q̃_j

Aprendizaje hebbiano: r̄ aprende la rotación q_i ⊗ conj(q_j) que mapea j→i.
Para aristas intra-dominio: q_i ≈ q_j → r̄ → identity → transmit(q_j) ≈ q_j.
Para aristas inter-dominio: cos(q_i,q_j) ≈ 0 → no se actualiza (estabilidad).
"""

import torch
import torch.nn.functional as F

from core.fiber_bundle import quat_product, quat_conjugate


class ActiveEdge:
    def __init__(self, K: int, edge_id: int = 0):
        self.K = K
        torch.manual_seed(100 + edge_id)
        self.r = F.normalize(torch.randn(K, 4), dim=-1)  # (K, 4)
        self._last_error: float = 0.0

    def mean_resonator(self) -> torch.Tensor:
        return F.normalize(self.r.mean(dim=0), dim=-1)

    def transmit(self, q_j: torch.Tensor) -> torch.Tensor:
        """Señal transformada: r̄_ij ⊗ q̃_j → (4,)."""
        r_mean = self.mean_resonator()
        return F.normalize(
            quat_product(r_mean.unsqueeze(0), q_j.unsqueeze(0)).squeeze(0),
            dim=-1
        )

    def transmit_quality(self, q_i: torch.Tensor, q_j: torch.Tensor) -> float:
        """cos(transmit(q_j), q_i): qué tan bien llega la señal de j a i."""
        signal = self.transmit(q_j)
        return float(F.normalize(signal, dim=-1) @ F.normalize(q_i, dim=-1))

    @property
    def last_error(self) -> float:
        return self._last_error

    def update(self, q_i: torch.Tensor, q_j: torch.Tensor, eta: float = 0.05):
        """
        Hebbian: r̄ aprende la rotación q_i ⊗ conj(q_j).
        Solo actúa cuando cos(q_i, q_j) > 0 (señales coherentes).
        """
        q_i_n = F.normalize(q_i, dim=-1)
        q_j_n = F.normalize(q_j, dim=-1)
        cos_ij = float(q_i_n @ q_j_n)
        if cos_ij <= 0:
            self._last_error = 1.0
            return
        self._last_error = 1.0 - self.transmit_quality(q_i_n, q_j_n)
        q_j_inv = quat_conjugate(q_j_n)
        q_target = F.normalize(
            quat_product(q_i_n.unsqueeze(0), q_j_inv.unsqueeze(0)).squeeze(0),
            dim=-1
        )
        self.r = F.normalize(
            self.r + eta * cos_ij * (q_target.unsqueeze(0) - self.r),
            dim=-1
        )
