"""
E_FIBRADO_CANTOR — Campos Anidados (Fiber Bundle sobre S³).

Cada nodo tiene dos capas de orientación:
  q_g ∈ S³   — posición en el campo base (dominio)
  q_l ∈ S³   — posición en la fibra local (especialización dentro del dominio)

El estado total:  q_total = q_g ⊗ q_l  (producto cuaterniónico, unit-norma garantizada)

Convención fija (q_g primero, q_l segundo): NO conmutar.

La clase FiberBundle administra:
  - Un ResonantField global (base) de N nodos
  - Un ResonantField local por dominio (fibra), cada uno de n_fibers nodos
  - Proyecciones y embeddings entre los dos espacios
"""

import torch
import torch.nn.functional as F
from typing import List, Optional


# ── Aritmética cuaterniónica ──────────────────────────────────────────────────

def quat_product(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Producto cuaterniónico q1 ⊗ q2.
    Entradas: (..., 4) con convención (w, x, y, z).
    Salida:   (..., 4), unit-norma si ambas entradas lo son.
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return torch.stack([w, x, y, z], dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Conjugado cuaterniónico (w, -x, -y, -z)."""
    conj = q.clone()
    conj[..., 1:] = -q[..., 1:]
    return conj


def embed_total(q_g: torch.Tensor, q_l: torch.Tensor) -> torch.Tensor:
    """
    Combina base y fibra en un único cuaternión total.
    q_total = q_g ⊗ q_l  (N, 4)
    """
    return F.normalize(quat_product(q_g, q_l), dim=-1)


def project_base(q_total: torch.Tensor) -> torch.Tensor:
    """
    Proyecta q_total al espacio base (devuelve la parte 'gruesa').
    Aproximación: tomar los dos primeros componentes (w,x) como re-normalizado.
    En la implementación completa, se usaría la fibra conocida para despejar q_g.
    Para diagnóstico: devuelve q_total directamente normalizado en (w,x,0,0).
    """
    part = torch.zeros_like(q_total)
    part[..., 0] = q_total[..., 0]
    part[..., 1] = q_total[..., 1]
    nrm = part.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return part / nrm


def project_fiber(q_total: torch.Tensor, q_g: torch.Tensor) -> torch.Tensor:
    """
    Recupera la fibra local: q_l = q_g^{-1} ⊗ q_total.
    Requiere conocer q_g (posición base del nodo).
    """
    q_g_inv = quat_conjugate(F.normalize(q_g, dim=-1))
    return F.normalize(quat_product(q_g_inv, q_total), dim=-1)


# ── FiberBundle ───────────────────────────────────────────────────────────────

class FiberBundle:
    """
    Gestiona un campo base + campos locales por dominio.

    Parámetros
    ----------
    n_nodes   : nodos en el campo base
    n_domains : número de dominios (fibras)
    n_fibers  : nodos por campo local (fibra)
    """

    def __init__(self,
                 n_nodes: int,
                 n_domains: int,
                 n_fibers: int,
                 seed: int = 42):
        torch.manual_seed(seed)
        self.n_nodes   = n_nodes
        self.n_domains = n_domains
        self.n_fibers  = n_fibers

        # q_global: posición de cada nodo en el campo base  (N, 4)
        raw = torch.randn(n_nodes, 4)
        self.q_global = F.normalize(raw, dim=-1)

        # q_local: posición de cada nodo en su fibra de dominio  (N, 4)
        raw = torch.randn(n_nodes, 4)
        self.q_local = F.normalize(raw, dim=-1)

        # Asignación de dominio: (N,) con valores en [0, n_domains)
        self.domain_id = torch.zeros(n_nodes, dtype=torch.long)

        # Prototipos por dominio en el campo base y en la fibra
        # (n_domains, 4)
        raw_g = torch.randn(n_domains, 4)
        self.proto_global = F.normalize(raw_g, dim=-1)

        raw_l = torch.randn(n_domains, 4)
        self.proto_local  = F.normalize(raw_l, dim=-1)

    # ── Asignación ──────────────────────────────────────────────────────────

    def assign_domains(self, domain_ids: torch.Tensor):
        """Asigna cada nodo a un dominio. domain_ids: (N,) tensor long."""
        self.domain_id = domain_ids.clone()

    # ── Calibración ─────────────────────────────────────────────────────────

    def calibrate_global(self, node_ids: List[int], q_target: torch.Tensor,
                         alpha: float = 0.30, n_steps: int = 20):
        """Ancla nodos en el campo base hacia q_target."""
        qt = F.normalize(q_target, dim=-1)
        for _ in range(n_steps):
            q_now = self.q_global[node_ids]
            q_new = q_now + alpha * (qt.unsqueeze(0) - q_now)
            self.q_global[node_ids] = F.normalize(q_new, dim=-1)

    def calibrate_local(self, node_ids: List[int], q_target: torch.Tensor,
                        alpha: float = 0.30, n_steps: int = 20):
        """Ancla nodos en la fibra local hacia q_target."""
        qt = F.normalize(q_target, dim=-1)
        for _ in range(n_steps):
            q_now = self.q_local[node_ids]
            q_new = q_now + alpha * (qt.unsqueeze(0) - q_now)
            self.q_local[node_ids] = F.normalize(q_new, dim=-1)

    # ── Estado total ─────────────────────────────────────────────────────────

    def q_total(self, node_ids: Optional[List[int]] = None) -> torch.Tensor:
        """Devuelve q_total = q_g ⊗ q_l para los nodos indicados (o todos)."""
        if node_ids is None:
            return embed_total(self.q_global, self.q_local)
        idx = torch.tensor(node_ids, dtype=torch.long)
        return embed_total(self.q_global[idx], self.q_local[idx])

    # ── Similitud y separación ───────────────────────────────────────────────

    def cosine_global(self, i: int, j: int) -> float:
        qi = self.q_global[i]
        qj = self.q_global[j]
        return float(F.normalize(qi, dim=-1) @ F.normalize(qj, dim=-1))

    def cosine_local(self, i: int, j: int) -> float:
        qi = self.q_local[i]
        qj = self.q_local[j]
        return float(F.normalize(qi, dim=-1) @ F.normalize(qj, dim=-1))

    def cosine_total(self, i: int, j: int) -> float:
        qt = self.q_total()
        qi = qt[i]
        qj = qt[j]
        return float(F.normalize(qi, dim=-1) @ F.normalize(qj, dim=-1))

    def angle_deg(self, cos_val: float) -> float:
        cos_val = max(-1.0, min(1.0, cos_val))
        return float(torch.acos(torch.tensor(cos_val)).item() * 180.0 / 3.14159265)

    def separation_global(self, group_a: List[int], group_b: List[int]) -> float:
        """Ángulo mínimo entre los dos grupos en el espacio base."""
        min_cos = 2.0
        for i in group_a:
            for j in group_b:
                c = self.cosine_global(i, j)
                if c < min_cos:
                    min_cos = c
        return self.angle_deg(min_cos)

    def separation_local(self, group_a: List[int], group_b: List[int]) -> float:
        """Ángulo mínimo entre los dos grupos en la fibra."""
        min_cos = 2.0
        for i in group_a:
            for j in group_b:
                c = self.cosine_local(i, j)
                if c < min_cos:
                    min_cos = c
        return self.angle_deg(min_cos)

    def separation_total(self, group_a: List[int], group_b: List[int]) -> float:
        """Ángulo mínimo entre los dos grupos en el espacio total q_g⊗q_l."""
        qt = self.q_total()
        min_cos = 2.0
        for i in group_a:
            for j in group_b:
                qi = F.normalize(qt[i], dim=-1)
                qj = F.normalize(qt[j], dim=-1)
                c = float(qi @ qj)
                if c < min_cos:
                    min_cos = c
        return self.angle_deg(min_cos)

    # ── Inserción resonante simplificada ─────────────────────────────────────

    def insert_pattern(self, omega: torch.Tensor, domain_id: int,
                       alpha_g: float = 0.15, alpha_l: float = 0.15,
                       n_expose: int = 5):
        """
        Inserta un patrón omega (3D) en el campo.
        - Convierte omega → q_global_target  (solo parte xyz, w=√(1-|xyz|²))
        - El campo base se acerca a q_g_target para los nodos del dominio.
        - La fibra local se acerca a q_l_target = rotación ortogonal.

        omega: (3,) tensor
        """
        omega_n = F.normalize(omega.float(), dim=-1)
        # q_global_target: codificar omega en xyz del cuaternión
        scale = float(omega_n.norm().clamp(max=0.999))
        w_g = (1.0 - scale**2)**0.5
        q_g_target = torch.tensor([w_g, omega_n[0].item(),
                                   omega_n[1].item(), omega_n[2].item()])
        q_g_target = F.normalize(q_g_target, dim=-1)

        # q_local_target: rotación de 90° en xy para que sea ortogonal al global
        q_l_target = torch.tensor([w_g, -omega_n[1].item(),
                                    omega_n[0].item(), omega_n[2].item()])
        q_l_target = F.normalize(q_l_target, dim=-1)

        nodes = (self.domain_id == domain_id).nonzero(as_tuple=True)[0].tolist()
        if not nodes:
            return

        for _ in range(n_expose):
            self.calibrate_global(nodes, q_g_target, alpha=alpha_g, n_steps=3)
            self.calibrate_local(nodes, q_l_target, alpha=alpha_l, n_steps=3)

    def novelty_flat(self, omega: torch.Tensor) -> float:
        """Novelty en el campo PLANO (solo q_global, ignora fibra)."""
        omega_n = F.normalize(omega.float(), dim=-1)
        scale = float(omega_n.norm().clamp(max=0.999))
        w_g = (1.0 - scale**2)**0.5
        q_query = torch.tensor([w_g, omega_n[0].item(),
                                 omega_n[1].item(), omega_n[2].item()])
        q_query = F.normalize(q_query, dim=-1)
        res = (self.q_global @ q_query).max()
        return float(1.0 - res.clamp(-1, 1))

    def novelty_nested(self, omega: torch.Tensor, domain_id: int) -> float:
        """
        Novelty en el campo ANIDADO: resonancia en q_total restringida al dominio.
        """
        omega_n = F.normalize(omega.float(), dim=-1)
        scale = float(omega_n.norm().clamp(max=0.999))
        w_g = (1.0 - scale**2)**0.5
        q_g = torch.tensor([w_g, omega_n[0].item(),
                             omega_n[1].item(), omega_n[2].item()])
        # Fibra ortogonal
        q_l = torch.tensor([w_g, -omega_n[1].item(),
                              omega_n[0].item(), omega_n[2].item()])
        q_query = F.normalize(quat_product(
            F.normalize(q_g, dim=-1),
            F.normalize(q_l, dim=-1)
        ), dim=-1)

        nodes = (self.domain_id == domain_id).nonzero(as_tuple=True)[0]
        if len(nodes) == 0:
            return 1.0
        qt = self.q_total(nodes.tolist())
        qt_n = F.normalize(qt, dim=-1)
        res = (qt_n @ q_query).max()
        return float(1.0 - res.clamp(-1, 1))

    # ── Enrutamiento ─────────────────────────────────────────────────────────

    def route_flat(self, omega: torch.Tensor) -> int:
        """Enruta omega al dominio por máxima resonancia en q_global."""
        omega_n = F.normalize(omega.float(), dim=-1)
        scale = float(omega_n.norm().clamp(max=0.999))
        w_g = (1.0 - scale**2)**0.5
        q_query = torch.tensor([w_g, omega_n[0].item(),
                                 omega_n[1].item(), omega_n[2].item()])
        q_query = F.normalize(q_query, dim=-1)
        best_dom, best_res = 0, -2.0
        for d in range(self.n_domains):
            nodes = (self.domain_id == d).nonzero(as_tuple=True)[0]
            if len(nodes) == 0:
                continue
            res = float((self.q_global[nodes] @ q_query).max())
            if res > best_res:
                best_res = res
                best_dom = d
        return best_dom

    def route_hierarchical(self, omega: torch.Tensor) -> tuple:
        """
        Enrutamiento jerárquico: primero dominio (base), luego subtipo (fibra).
        Devuelve (domain_id, subtype_centroid_angle) para comparación.
        """
        dom = self.route_flat(omega)
        # En la fibra del dominio elegido, calcular resonancia local
        omega_n = F.normalize(omega.float(), dim=-1)
        scale = float(omega_n.norm().clamp(max=0.999))
        w_l = (1.0 - scale**2)**0.5
        # El subtipo se codifica en q_local con rotación distinta
        q_l_query = torch.tensor([w_l, -omega_n[1].item(),
                                    omega_n[0].item(), omega_n[2].item()])
        q_l_query = F.normalize(q_l_query, dim=-1)
        nodes = (self.domain_id == dom).nonzero(as_tuple=True)[0]
        if len(nodes) == 0:
            return dom, 0.0
        res_l = float((self.q_local[nodes] @ q_l_query).max())
        return dom, res_l

    # ── Norma del estado total ────────────────────────────────────────────────

    def norm_error(self) -> float:
        """Error de norma máximo en q_total (debe ser ≈0)."""
        qt = self.q_total()
        norms = qt.norm(dim=-1)
        return float((norms - 1.0).abs().max())
