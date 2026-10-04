"""
ResonantTissue: tejido resonante donde cada nodo es un micro-campo de M osciladores
y cada arista es un transformador con K resonadores cuaterniónicos.

Dinámica por paso:
  1. Centroides: q̃_i = normalize(mean(q_{i,k})) para cada nodo.
  2. Propagación: influjo_i = mean_k(r̄_{ij} ⊗ q̃_j) para j∈vecinos(i).
  3. Actualización de sub-osciladores: update_all(influjo_i).
  4. Actualización de resonadores: Hebbian r̄_ij → q̃_i ⊗ conj(q̃_j).
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from core.subfield_node import SubfieldNode
from core.active_edge import ActiveEdge


class ResonantTissue:
    def __init__(self,
                 N: int,
                 M: int,
                 K: int,
                 edge_index: torch.Tensor,
                 seed: int = 42):
        torch.manual_seed(seed)
        self.N = N
        self.M = M
        self.K = K
        self.edge_index = edge_index  # (2, E)

        self.nodes: List[SubfieldNode] = [
            SubfieldNode(M, node_id=i) for i in range(N)
        ]

        # Aristas: (i←j) indexadas como (i,j)
        self.edges: Dict[Tuple[int, int], ActiveEdge] = {}
        E = edge_index.shape[1]
        for e in range(E):
            i, j = int(edge_index[0, e]), int(edge_index[1, e])
            if (i, j) not in self.edges:
                self.edges[(i, j)] = ActiveEdge(K, edge_id=e)

    # ── Centroid snapshot ────────────────────────────────────────────────────

    def centroids(self) -> torch.Tensor:
        """Devuelve (N, 4) tensor de centroides globales."""
        return torch.stack([n.centroid() for n in self.nodes])

    def centroids_total(self) -> torch.Tensor:
        """Devuelve (N, 4) tensor de centroides anidados q_global ⊗ q_local."""
        return torch.stack([n.centroid_total() for n in self.nodes])

    # ── Dinámica ─────────────────────────────────────────────────────────────

    def step(self, alpha_node: float = 0.10, eta_edge: float = 0.05):
        """Un paso de dinámica del tejido (propagación + Hebbian)."""
        c = self.centroids_total()
        influences: List[List[torch.Tensor]] = [[] for _ in range(self.N)]

        for (i, j), edge in self.edges.items():
            influences[i].append(edge.transmit(c[j]))
            edge.update(c[i], c[j], eta=eta_edge)

        for i in range(self.N):
            if influences[i]:
                mean_inf = F.normalize(
                    torch.stack(influences[i]).mean(0), dim=-1
                )
                self.nodes[i].update_all(mean_inf, alpha=alpha_node)

    # ── Inserción de patrones ────────────────────────────────────────────────

    def _omega_to_q(self, omega: torch.Tensor) -> torch.Tensor:
        omega_n = F.normalize(omega.float(), dim=-1)
        sc = float(omega_n.norm().clamp(max=0.999))
        w = (1.0 - sc ** 2) ** 0.5
        return F.normalize(
            torch.tensor([w, omega_n[0].item(), omega_n[1].item(), omega_n[2].item()]),
            dim=-1
        )

    def insert_pattern(self,
                       node_ids: List[int],
                       omega: torch.Tensor,
                       alpha: float = 0.30,
                       n_expose: int = 5,
                       k_topk: int = None,
                       layer: str = 'global'):
        """
        Inserta un patrón omega en los nodos dados.
        k_topk=None → usa M//2 (modo tejido).
        k_topk=M     → actualiza todos (modo plano/flat).
        layer='global' → solo capa global (comportamiento original).
        layer='both'   → también actualiza la fibra local de cada nodo.
        """
        qt = self._omega_to_q(omega)
        for _ in range(n_expose):
            for nid in node_ids:
                self.nodes[nid].update_topk(qt, alpha=alpha, k=k_topk)
                if layer == 'both':
                    self.nodes[nid].update_local(qt, alpha=alpha)

    # ── Métricas ─────────────────────────────────────────────────────────────

    def novelty(self, omega: torch.Tensor,
                node_ids: Optional[List[int]] = None) -> float:
        """Novelty = 1 − max_resonance usando centroides anidados (total)."""
        qt = self._omega_to_q(omega)
        c = self.centroids_total()
        if node_ids is not None:
            c = c[torch.tensor(node_ids, dtype=torch.long)]
        return float(1.0 - (c @ qt).max().clamp(-1.0, 1.0))

    def novelty_nested(self, omega: torch.Tensor,
                       layer: str = 'total',
                       node_ids: Optional[List[int]] = None) -> float:
        """
        Novelty con capa seleccionable:
          'global' → centroides globales (capa q̃_i).
          'local'  → fibras locales q_local_i.
          'total'  → centroides anidados q_global ⊗ q_local (por defecto).
        """
        qt = self._omega_to_q(omega)
        if layer == 'global':
            c = self.centroids()
        elif layer == 'local':
            c = torch.stack([
                F.normalize(n.q_local, dim=-1) for n in self.nodes
            ])
        else:
            c = self.centroids_total()
        if node_ids is not None:
            c = c[torch.tensor(node_ids, dtype=torch.long)]
        return float(1.0 - (c @ qt).max().clamp(-1.0, 1.0))

    def predictive_errors(self) -> torch.Tensor:
        """
        (N,) tensor: error predictivo medio de las aristas entrantes de cada nodo.
        Si un nodo no tiene aristas entrantes, su error es 0.0.
        """
        errors = torch.zeros(self.N)
        counts = torch.zeros(self.N)
        for (i, _), edge in self.edges.items():
            errors[i] += edge.last_error
            counts[i] += 1.0
        mask = counts > 0
        errors[mask] /= counts[mask]
        return errors

    def semantic_mass(self) -> torch.Tensor:
        """(N,) tensor: masa semántica M_i = Σ|α_{i,n}|² de cada SubfieldNode."""
        return torch.tensor([n.solitonic_mass for n in self.nodes])

    def separation_mean(self, group_a: List[int], group_b: List[int]) -> float:
        """Ángulo medio (grados) entre todos los pares de centroides de los grupos."""
        c = self.centroids()
        cos_vals = [
            float(F.normalize(c[i], dim=-1) @ F.normalize(c[j], dim=-1))
            for i in group_a for j in group_b
        ]
        mean_cos = max(-1.0, min(1.0, sum(cos_vals) / len(cos_vals)))
        return float(torch.acos(torch.tensor(mean_cos)).item() * 180.0 / 3.14159265)

    def edge_stats(self, domain_map: Dict[int, int]) -> dict:
        """
        Estadísticas de calidad de transmisión por tipo de arista.
        quality(i,j) = cos(transmit(c_j), c_i) — qué tan fielmente llega j a i.
        """
        c = self.centroids()
        intra, inter = [], []
        for (i, j), edge in self.edges.items():
            q = edge.transmit_quality(c[i], c[j])
            if domain_map.get(i, -1) == domain_map.get(j, -1):
                intra.append(q)
            else:
                inter.append(q)
        return {
            "mean_intra": sum(intra) / len(intra) if intra else 0.0,
            "mean_inter": sum(inter) / len(inter) if inter else 0.0,
            "Q_proxy":    (sum(intra) / len(intra) if intra else 0.0)
                          - (sum(inter) / len(inter) if inter else 0.0),
            "n_intra": len(intra),
            "n_inter": len(inter),
        }

    def dormant_edges(self, domain_map: Dict[int, int],
                      quality_threshold: float = 0.10) -> List[Tuple[int, int]]:
        """Aristas inter-dominio con transmit_quality < threshold (dormantes)."""
        c = self.centroids()
        dormant = []
        for (i, j), edge in self.edges.items():
            if domain_map.get(i, -1) == domain_map.get(j, -1):
                continue
            if edge.transmit_quality(c[i], c[j]) < quality_threshold:
                dormant.append((i, j))
        return dormant

    def reactivation_rate(self,
                          edge_list: List[Tuple[int, int]],
                          quality_threshold: float = 0.30) -> float:
        """Fracción de aristas con transmit_quality > threshold tras re-entrenamiento."""
        if not edge_list:
            return 0.0
        c = self.centroids()
        reactivated = sum(
            1 for (i, j) in edge_list
            if self.edges[(i, j)].transmit_quality(c[i], c[j]) > quality_threshold
        )
        return reactivated / len(edge_list)
