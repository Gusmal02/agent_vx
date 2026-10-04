"""
domain_context.py — Módulo de Contexto (memoria de trabajo)

Separa el contexto activo (episodio en curso) de la memoria persistente.
El contexto es volátil: existe solo mientras dura el episodio.
La memoria (MemoryModule) es persistente y se consulta desde aquí.

Responsabilidades:
  - Mantener el estado del episodio activo (pasos, recompensas, omegas)
  - Proveer resumen vectorial del contexto actual para que los especialistas lo usen
  - Limpiar entre episodios
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class EpisodeStep:
    omega: torch.Tensor      # estado observado (S²)
    action_idx: int          # acción tomada
    winner: torch.Tensor     # omega ganadora (S²)
    reward: float            # recompensa obtenida
    regime: str              # "fast" | "resonant" | "random"


class ContextModule:
    """
    Memoria de trabajo episódica.

    Mantiene el historial del episodio activo y construye un vector de
    contexto resumido que los especialistas pueden consultar para
    tomar decisiones informadas por el pasado reciente.
    """

    def __init__(self, window_size: int = 10, dim: int = 3):
        self.window_size = window_size
        self.dim = dim
        self._steps: List[EpisodeStep] = []
        self._episode_count: int = 0
        self._total_reward: float = 0.0

    # ── Escritura ─────────────────────────────────────────────────────────────

    def record_step(self,
                    omega: torch.Tensor,
                    action_idx: int,
                    winner: torch.Tensor,
                    reward: float,
                    regime: str = "resonant") -> None:
        step = EpisodeStep(omega.clone(), action_idx, winner.clone(), reward, regime)
        self._steps.append(step)
        self._total_reward += reward
        if len(self._steps) > self.window_size * 4:
            self._steps = self._steps[-self.window_size * 2:]

    def close_episode(self) -> float:
        """Cierra el episodio actual. Devuelve la recompensa total."""
        total = self._total_reward
        self._episode_count += 1
        self._steps = []
        self._total_reward = 0.0
        return total

    # ── Lectura ───────────────────────────────────────────────────────────────

    def context_vector(self) -> Optional[torch.Tensor]:
        """
        Vector de contexto: promedio ponderado de las omegas recientes.
        Pasos más recientes tienen mayor peso (decaimiento exponencial).
        Devuelve None si no hay pasos registrados.
        """
        if not self._steps:
            return None
        window = self._steps[-self.window_size:]
        weights = torch.exp(torch.linspace(-2.0, 0.0, len(window)))
        stacked = torch.stack([s.omega for s in window])       # (T, 3)
        weighted = (stacked * weights.unsqueeze(1)).sum(0)     # (3,)
        return F.normalize(weighted, dim=-1)

    def recent_rewards(self, n: int = 5) -> List[float]:
        return [s.reward for s in self._steps[-n:]]

    def momentum(self) -> float:
        """Tendencia de recompensa reciente: positiva si está mejorando."""
        rr = self.recent_rewards(6)
        if len(rr) < 2:
            return 0.0
        mid = len(rr) // 2
        return float(sum(rr[mid:]) - sum(rr[:mid]))

    def current_step_count(self) -> int:
        return len(self._steps)

    def summary(self) -> dict:
        return {
            "episode": self._episode_count,
            "steps_this_ep": len(self._steps),
            "reward_so_far": round(self._total_reward, 2),
            "momentum": round(self.momentum(), 3),
        }
