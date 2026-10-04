"""
Campo completo: N osciladores acoplados sobre un grafo.
Gestiona atractor, parámetro de orden r(t), y evolución por pasos.
"""

import torch
from core.oscillator import (
    dqdt, domegadt, dmdt,
    quat_normalize, omega_to_quat, quat_mul,
)


class ResonantField:
    """
    Estado del campo: q (N,4), omega (N,3), m (N,1)
    Grafo: edge_index (2, E), edge_weight (E,) opcional
    """

    def __init__(
        self,
        N: int,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
        K: float = 1.0,
        gamma_omega: float = 0.01,
        gamma_m: float = 0.001,
        alpha_slow: float = 1e-5,
        omega_std: float = 0.3,
        dt: float = 0.05,
        device: str = "cpu",
    ):
        self.N = N
        self.K = K
        self.gamma_omega = gamma_omega
        self.gamma_m = gamma_m
        self.alpha_slow = alpha_slow
        self.dt = dt
        self.device = device

        self.edge_index = edge_index.to(device)
        if edge_weight is None:
            E = edge_index.shape[1]
            edge_weight = torch.ones(E, device=device)
        self.edge_weight = edge_weight.to(device)

        # Estado
        self.q = quat_normalize(torch.randn(N, 4, device=device))
        self.omega = torch.randn(N, 3, device=device) * omega_std
        self.m = torch.zeros(N, 1, device=device)

        # Atractor: frecuencia colectiva emergente
        self.omega_star = self.omega.mean(dim=0)  # (3,)
        self.q_star = self.q.clone()              # referencia inicial

        self.t = 0.0
        self.t_step = 0  # contador entero de pasos

        # H3 — Tiempo como dimensión
        # t_insert[i] = self.t cuando omega_i fue deformado por última vez. -1 = nunca.
        self.t_insert = torch.full((N,), -1.0, device=device)

        # H4 — Olvido
        # omega_init: copia del omega inicial (estado "en blanco" antes de cualquier inserción)
        self.omega_init = self.omega.clone()
        # f_activation: acumulado de cuánto ha resonado cada nodo (queries + inserts)
        self.f_activation = torch.zeros(N, device=device)

        # Memoria protegida — protection_factor[i] ∈ [0, 1]
        # gamma_efectivo_i = gamma_base × (1 − protection_factor[i])
        # 0 = sin protección (olvido normal); 1 = protección total (no olvida)
        self.protection_factor = torch.zeros(N, device=device)

    # ─── Promedios sobre vecindad ─────────────────────────────────────────

    def _neighbor_mean_q(self) -> torch.Tensor:
        """Promedio ponderado de q de vecinos para cada nodo. (N, 4)"""
        src, dst = self.edge_index[0], self.edge_index[1]
        w = self.edge_weight

        num = torch.zeros(self.N, 4, device=self.device)
        den = torch.zeros(self.N, 1, device=self.device)
        num.index_add_(0, dst, w.unsqueeze(-1) * self.q[src])
        den.index_add_(0, dst, w.unsqueeze(-1))

        # nodos sin vecinos reciben su propio q
        mask = (den.squeeze(-1) == 0)
        den[mask] = 1.0
        num[mask] = self.q[mask]

        return quat_normalize(num / den)

    def _neighbor_mean_omega(self) -> torch.Tensor:
        """Promedio ponderado por edge_weight de ω de vecinos. (N, 3)"""
        src, dst = self.edge_index[0], self.edge_index[1]
        w = self.edge_weight

        num = torch.zeros(self.N, 3, device=self.device)
        den = torch.zeros(self.N, 1, device=self.device)
        num.index_add_(0, dst, w.unsqueeze(-1) * self.omega[src])
        den.index_add_(0, dst, w.unsqueeze(-1))

        mask = (den.squeeze(-1) == 0)
        den[mask] = 1.0
        num[mask] = self.omega[mask]

        return num / den

    # ─── Atractor y observable ────────────────────────────────────────────

    def _attractor_q(self, t: float) -> torch.Tensor:
        """
        q_i*(t) = exp(½ [0, ω*] t) · q_i*(0)
        Aproximamos exp usando la serie de primer orden para dt pequeño.
        """
        # Para t completo usamos la fórmula exacta: exp(½ θ n̂) = cos(θ/2) + sin(θ/2) n̂
        angle_vec = 0.5 * self.omega_star * t   # (3,)
        angle = angle_vec.norm()
        if angle < 1e-10:
            rot = torch.tensor([1., 0., 0., 0.], device=self.device)
        else:
            n = angle_vec / angle
            rot = torch.cat([
                torch.cos(angle).unsqueeze(0),
                torch.sin(angle) * n,
            ])  # (4,)

        # rot ⊗ q_star para cada nodo
        rot_expanded = rot.unsqueeze(0).expand(self.N, -1)
        return quat_normalize(quat_mul(rot_expanded, self.q_star))

    def order_parameter(self) -> float:
        """
        r(t) = |(1/N) Σ_i conj(R(t)) ⊗ q_i| ∈ [0, 1]

        Mide coherencia del ensemble en el marco rotante a ω*.
        r=1: todos los q_i giran juntos en perfecta sincronía.
        r≈0: q_i apuntan en direcciones dispares.
        """
        # R_inv = exp(−½ [0,ω*] t)
        angle_vec = -0.5 * self.omega_star * self.t
        angle = angle_vec.norm()
        if angle < 1e-10:
            rot_inv = torch.tensor([1., 0., 0., 0.],
                                   dtype=self.q.dtype, device=self.device)
        else:
            n = angle_vec / angle
            rot_inv = torch.cat([
                torch.cos(angle).unsqueeze(0),
                torch.sin(angle) * n,
            ])

        rot_inv_exp = rot_inv.unsqueeze(0).expand(self.N, -1)
        q_tilde = quat_mul(rot_inv_exp, self.q)   # (N, 4) en marco rotante
        mean_q = q_tilde.mean(dim=0)              # (4,)
        return float(mean_q.norm().item())

    # ─── Paso de integración (Euler) ─────────────────────────────────────

    def step(self, xi: torch.Tensor | None = None, beta: torch.Tensor | None = None):
        """
        Un paso de integración Euler de tamaño dt.
        xi   : perturbación externa en q  (N, 4) o None
        beta : perturbación externa en ω  (N, 3) o None
        """
        q_nb = self._neighbor_mean_q()
        w_nb = self._neighbor_mean_omega()

        dq = dqdt(self.q, self.omega, q_nb, self.K, xi)
        dw = domegadt(self.omega, w_nb, self.gamma_omega, beta)

        # δq_i = q_i − q̄_i: desviación del nodo respecto a su vecindario.
        # Cero en reposo sincronizado; sube solo ante perturbaciones locales reales.
        delta_q_norm = (self.q - q_nb).norm(dim=-1, keepdim=True)
        dm = dmdt(self.m, delta_q_norm, self.gamma_m)

        self.q = quat_normalize(self.q + self.dt * dq)
        self.omega = self.omega + self.dt * dw
        self.m = (self.m + self.dt * dm).clamp(min=0.0)

        self.t += self.dt
        self.t_step += 1

    def forget_step(self, gamma_base: float = 0.001):
        """
        Olvido con γ_m adaptativo por nodo (T1 Idea 2) + memoria protegida.

        γ_m_i = gamma_base / (1 + f_activation_i) × (1 − protection_factor_i)

        protection_factor_i = 0 → olvido normal
        protection_factor_i = 1 → no olvida (protección total)
        """
        gamma_i = gamma_base / (1.0 + self.f_activation)   # (N,)
        gamma_i = gamma_i * (1.0 - self.protection_factor)  # protección reduce el olvido
        self.omega = self.omega - gamma_i.unsqueeze(-1) * (self.omega - self.omega_init)
        self.f_activation = self.f_activation * 0.995

    def merge_similar_nodes(self, theta_merge: float = 0.98) -> list[tuple[int, int]]:
        """
        Fusión de nodos similares (T2 Idea 2).

        Si resonance(i, j) = ⟨ω_i/|ω_i|, ω_j/|ω_j|⟩ > theta_merge, los nodos i y j
        apuntan casi en la misma dirección — almacenan el mismo patrón dos veces.

        Fusión: omega_i ← media ponderada por f_activation; omega_j ← omega_i.
        El nodo j queda con omega idéntico a i pero con f_activation=0 (huella borrada).
        El campo no pierde nodos — solo redistribuye la información.

        Devuelve lista de pares (i, j) fusionados.
        """
        ni = self.omega / self.omega.norm(dim=-1, keepdim=True).clamp(min=1e-12)  # (N,3)
        # Matriz de resonancias NxN
        cos_sim = ni @ ni.T   # (N, N)

        merged = []
        visited = set()
        for i in range(self.N):
            if i in visited:
                continue
            for j in range(i + 1, self.N):
                if j in visited:
                    continue
                if float(cos_sim[i, j]) > theta_merge:
                    # Fusionar: omega_i ← media ponderada por f_activation
                    act_i = float(self.f_activation[i])
                    act_j = float(self.f_activation[j])
                    total = act_i + act_j + 1e-9
                    self.omega[i] = (act_i * self.omega[i] + act_j * self.omega[j]) / total
                    # j adopta la dirección de i; su f_activation se transfiere a i
                    self.omega[j] = self.omega[i].clone()
                    self.f_activation[i] = act_i + act_j
                    self.f_activation[j] = 0.0
                    merged.append((i, j))
                    visited.add(j)
        return merged

    def step_slow(self, window_omega: torch.Tensor):
        """
        Actualización lenta del atractor (cada T_slow pasos).
        ω_i*(t+Δ) = (1−α_slow)·ω_i*(t) + α_slow·⟨ω_i⟩_ventana
        Aquí simplificamos: omega_star += alpha_slow * (media_global - omega_star)
        """
        self.omega_star = (
            (1 - self.alpha_slow) * self.omega_star
            + self.alpha_slow * window_omega
        )
        # Renovar referencia del atractor al tiempo actual
        self.q_star = self.q.clone()
