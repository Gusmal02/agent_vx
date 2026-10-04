"""
SubfieldNode: nodo como micro-campo de M osciladores cuaterniónicos.

q̃_i = normalize(mean({q_{i,k}})) es la posición efectiva del nodo (capa global).
q_local ∈ S³ es la posición en la fibra local del nodo.
centroid_total() = q_global ⊗ q_local — representación anidada 8D implícita.

Solo los top-k osciladores más resonantes se actualizan durante la inserción,
preservando memoria de patrones anteriores en los osciladores silenciosos.
"""

import torch
import torch.nn.functional as F

from core.fiber_bundle import quat_product


class SubfieldNode:
    def __init__(self, M: int, node_id: int = 0):
        self.M = M
        torch.manual_seed(42 + node_id)
        self.q = F.normalize(torch.randn(M, 4), dim=-1)  # (M, 4) — capa global
        # Fibra local: inicializada como cuaternión identidad para no distorsionar
        # la dinámica existente antes de que reciba calibración explícita.
        self.q_local = torch.tensor([1.0, 0.0, 0.0, 0.0])

        # Historial de activaciones para masa semántica M_i = Σ|α_{i,n}|²
        self._activation_log: list[float] = []

    def centroid(self) -> torch.Tensor:
        """q̃_i = normalize(mean(q_{i,k})) — posición en la capa global."""
        return F.normalize(self.q.mean(dim=0), dim=-1)

    def centroid_total(self) -> torch.Tensor:
        """q_total = q_global ⊗ q_local — representación anidada normalizada."""
        q_g = self.centroid().unsqueeze(0)          # (1,4)
        q_l = F.normalize(self.q_local, dim=-1).unsqueeze(0)  # (1,4)
        return F.normalize(quat_product(q_g, q_l).squeeze(0), dim=-1)

    def update_topk(self, q_target: torch.Tensor,
                    alpha: float = 0.30, k: int = None):
        """
        Actualiza solo los top-k osciladores más resonantes con q_target.
        El resto retiene su posición (memoria de patrones anteriores).
        """
        if k is None:
            k = max(1, self.M // 2)
        qt = F.normalize(q_target, dim=-1)
        cos_vals = self.q @ qt          # (M,)
        top_vals, top_idx = cos_vals.topk(k)
        self.q[top_idx] = F.normalize(
            self.q[top_idx] + alpha * (qt.unsqueeze(0) - self.q[top_idx]),
            dim=-1
        )
        # Registrar activación media de los osciladores actualizados
        self._activation_log.append(float(top_vals.mean()))

    def update_all(self, q_target: torch.Tensor, alpha: float = 0.10):
        """Actualiza todos los osciladores (propagación de señal)."""
        qt = F.normalize(q_target, dim=-1)
        cos_vals = self.q @ qt
        self.q = F.normalize(
            self.q + alpha * (qt.unsqueeze(0) - self.q), dim=-1
        )
        self._activation_log.append(float(cos_vals.mean()))

    def update_local(self, q_target: torch.Tensor, alpha: float = 0.15):
        """Actualiza la fibra local independientemente del centroide global."""
        qt = F.normalize(q_target.float(), dim=-1)
        self.q_local = F.normalize(
            self.q_local + alpha * (qt - self.q_local), dim=-1
        )

    @property
    def solitonic_mass(self) -> float:
        """M_i = Σ|α_{i,n}|² — potencia espectral de activaciones pasadas."""
        if not self._activation_log:
            return 0.0
        return float(sum(a ** 2 for a in self._activation_log))
