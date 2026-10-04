"""
Arquitectura de dos capas: consciente (C) + subconsciente (S).

El campo subconsciente acumula todo lo que C aprende, pero:
  - aprende más lento (delta ∈ [0.05, 0.20] de alpha_learn_C)
  - olvida extremadamente lento o nada (gamma_m_S ≈ 0)
  - no puede ser consultado directamente — solo modula a C

Acoplamiento S → C (priming):
  novelty_efectiva(c) = novelty_C(c) × (1 − beta × resonance_S(c))

Si S reconoce c aunque C lo haya olvidado:
  resonance_S(c) alta → novelty_efectiva baja → "extrañamente familiar"
  (priming implícito, intuición sin memoria explícita)

Si S nunca vio c (patrón ajeno a ambos):
  resonance_S(c) ≈ 0 → novelty_efectiva ≈ novelty_C(c)  (sin modulación)
"""

import torch
import numpy as np
import networkx as nx

from core.field import ResonantField
from memory.insert import insert, insert_multi, resonance_scores_mag, encode


def _make_field(N: int, K: float, omega_std: float,
                alpha_slow: float, seed: int) -> ResonantField:
    """Campo BA-m3 estándar."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    g = nx.barabasi_albert_graph(N, 3, seed=seed)
    ei = torch.tensor(list(g.edges()), dtype=torch.long).t().contiguous()
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    return ResonantField(N, edge_index=ei, K=K, omega_std=omega_std,
                         alpha_slow=alpha_slow)


class TwoLayerMind:
    """
    Mente de dos capas: consciente (C) + subconsciente (S).

    Parámetros de S respecto a C:
      delta       : fracción del alpha_learn de C que recibe S en cada inserción
      gamma_m_S   : tasa de olvido del subconsciente (≪ gamma_m_C)
      beta        : fuerza del acoplamiento S→C en la modulación de novelty
      K_S         : acoplamiento del campo S (más bajo = más difuso, más estados)

    Uso típico:
      mind = TwoLayerMind(N=50, ...)
      mind.warmup(400)
      mind.insert(c, alpha_learn=0.15)   # alimenta C y S automáticamente
      nov = mind.novelty_efectiva(c)     # novelty modulada por S
    """

    def __init__(
        self,
        N: int = 50,
        K_C: float = 2.5,
        K_S: float = 1.0,
        omega_std: float = 0.05,
        alpha_slow: float = 0.0,
        gamma_m_C: float = 0.001,
        gamma_m_S: float = 0.00002,
        delta: float = 0.10,
        beta: float = 0.60,
        omega_scale: float = 1.5,
        seed: int = 42,
    ):
        self.N           = N
        self.delta       = delta
        self.beta        = beta
        self.omega_scale = omega_scale
        self.gamma_m_S   = gamma_m_S

        self.C = _make_field(N, K_C, omega_std, alpha_slow, seed)
        self.S = _make_field(N, K_S, omega_std, alpha_slow, seed + 1000)

        # Anular el olvido de omega en S — la memoria subconsciente persiste
        self.S.gamma_omega = 0.0
        # gamma_m de S configura el decaimiento de m_i, pero usaremos gamma_m_S
        # por separado en forget_step manual

    # ── API pública ──────────────────────────────────────────────────────────

    def warmup(self, steps: int = 400):
        for _ in range(steps):
            self.C.step()
            self.S.step()

    def insert(self, c: torch.Tensor, alpha_learn: float = 0.15,
               top_k: int = 5, T_settle: int = 20, n_expose: int = 1):
        """Inserta c en C (alpha_learn normal) y en S (alpha_learn × delta)."""
        if n_expose == 1:
            insert(self.C, c, omega_scale=self.omega_scale,
                   alpha_learn=alpha_learn, top_k=top_k, T_settle=T_settle)
            insert(self.S, c, omega_scale=self.omega_scale,
                   alpha_learn=alpha_learn * self.delta, top_k=top_k, T_settle=T_settle)
        else:
            insert_multi(self.C, c, n_expose=n_expose,
                         omega_scale=self.omega_scale, alpha_learn=alpha_learn,
                         top_k=top_k, T_settle=T_settle)
            insert_multi(self.S, c, n_expose=n_expose,
                         omega_scale=self.omega_scale,
                         alpha_learn=alpha_learn * self.delta,
                         top_k=top_k, T_settle=T_settle)

    def step(self):
        self.C.step()
        self.S.step()

    def forget_C(self, n_steps: int, gamma_forget: float = 0.003):
        """Aplica olvido activo a C para simular el paso del tiempo."""
        for _ in range(n_steps):
            self.C.step()
            # Decaimiento explícito de omega en C
            self.C.omega *= (1 - gamma_forget)
            self.C.m *= (1 - self.C.gamma_m)

    # ── Métricas ─────────────────────────────────────────────────────────────

    def resonance_C(self, c: torch.Tensor, top_k: int = 5) -> float:
        """Resonancia máxima de C con c."""
        _, omega_c = encode(c, self.omega_scale)
        scores = resonance_scores_mag(self.C.omega, omega_c, self.omega_scale)
        return float(scores.topk(top_k).values.mean().clamp(min=-1))

    def resonance_S(self, c: torch.Tensor, top_k: int = 5) -> float:
        """Resonancia máxima de S con c."""
        _, omega_c = encode(c, self.omega_scale)
        scores = resonance_scores_mag(self.S.omega, omega_c, self.omega_scale)
        return float(scores.topk(top_k).values.mean().clamp(min=-1))

    def novelty_C(self, c: torch.Tensor, top_k: int = 5) -> float:
        """Novelty raw de C: 1 − resonance_C."""
        return float(1 - self.resonance_C(c, top_k))

    def novelty_efectiva(self, c: torch.Tensor, top_k: int = 5) -> float:
        """
        Novelty modulada por S:
          nov_ef = nov_C × (1 − beta × max(resonance_S, 0))

        Si S reconoce c aunque C lo haya olvidado:
          resonance_S alta → nov_ef < nov_C  (priming)
        Si ninguno reconoce c:
          resonance_S ≈ 0 → nov_ef ≈ nov_C  (sin efecto)
        """
        nov_c = self.novelty_C(c, top_k)
        res_s = max(self.resonance_S(c, top_k), 0.0)
        return nov_c * (1 - self.beta * res_s)

    def priming_effect(self, c: torch.Tensor, top_k: int = 5) -> dict:
        """
        Cuantifica el efecto de priming: cuánto baja la novelty_efectiva
        respecto a la novelty raw de C.

        Un efecto positivo significa que S recuerda lo que C olvidó.
        """
        nov_c  = self.novelty_C(c, top_k)
        nov_ef = self.novelty_efectiva(c, top_k)
        res_s  = max(self.resonance_S(c, top_k), 0.0)
        res_c  = max(self.resonance_C(c, top_k), 0.0)
        return {
            "novelty_C":       round(nov_c,  4),
            "novelty_efectiva":round(nov_ef, 4),
            "priming_delta":   round(nov_c - nov_ef, 4),   # cuánto bajó
            "resonance_C":     round(res_c,  4),
            "resonance_S":     round(res_s,  4),
            "priming_active":  (nov_c - nov_ef) > 0.02,
        }

    def intuition_score(self, c: torch.Tensor, top_k: int = 5) -> dict:
        """
        Familiaridad sin reconocimiento explícito.

        intuition = mean_res_S × (1 − novelty_C)
                  = resonancia_subconsciente × conciencia_de_no-saber

        Alta cuando S reconoce algo que C no puede nombrar:
          — S resonó fuertemente con c (lo ha visto a través del tiempo)
          — C tiene alta novelty (lo reporta como desconocido)

        Baja en dos casos distintos:
          — C ya reconoce c (novelty_C baja → no hay "familiaridad sin nombre")
          — S tampoco lo conoce (resonance_S baja → no hay base subconsciente)

        Returns dict con:
          intuition_score : float ∈ [0, 1]  — la métrica compuesta
          resonance_S     : resonancia media del subconsciente
          novelty_C       : novedad del consciente
          active          : bool — si la intuición es suficientemente alta (> 0.15)
          interpretation  : str — descripción del estado fenomenológico
        """
        res_s = max(self.resonance_S(c, top_k), 0.0)
        nov_c = self.novelty_C(c, top_k)
        score = res_s * nov_c

        if score > 0.30:
            interp = "déjà vu fuerte — S recuerda pero C no puede nombrar"
        elif score > 0.15:
            interp = "familiaridad vaga — leve priming subconsciente"
        elif nov_c > 0.70 and res_s < 0.10:
            interp = "territorio completamente nuevo — sin base subconsciente"
        elif nov_c < 0.20:
            interp = "reconocimiento explícito — C ya lo sabe"
        else:
            interp = "zona ambigua — señal subconsciente débil"

        return {
            "intuition_score": round(score,  4),
            "resonance_S":     round(res_s,  4),
            "novelty_C":       round(nov_c,  4),
            "active":          score > 0.15,
            "interpretation":  interp,
        }

    def seed_candidates(
        self,
        candidates: list[torch.Tensor],
        top_k: int = 5,
        n: int = 3,
    ) -> list[dict]:
        """
        Prioriza semillas de continuidad por intuition_score.

        Entre los candidatos dados (p.ej. ωs de puentes cercanos),
        devuelve los n con mayor intuición: alta resonancia subconsciente
        pero baja conciencia explícita — la frontera donde vale explorar.

        Uso en RR-RMF: reemplaza siembra aleatoria de integradores
        con siembra guiada por lo que el subconsciente ya "presiente".
        """
        scored = [
            {"omega": c, **self.intuition_score(c, top_k)}
            for c in candidates
        ]
        scored.sort(key=lambda x: x["intuition_score"], reverse=True)
        return scored[:n]

    def order_parameters(self) -> dict:
        return {
            "r_C": round(float(self.C.order_parameter()), 4),
            "r_S": round(float(self.S.order_parameter()), 4),
        }
