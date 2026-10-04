"""
Memoria contextual para E_REVALORIZACION.

ContextualIdea: idea con su contexto de creación y estado.
BufferLimbo: pool de ideas muertas que pueden revivir si el contexto cambia.
"""

import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field as dc_field


@dataclass
class ContextualIdea:
    omega: torch.Tensor         # vector ω de la idea (3D)
    context_omega: torch.Tensor # ω del contexto cuando fue registrada/rechazada
    status: str                 # 'alive' | 'dead' | 'revived'
    label: str = ""
    domain: str = ""
    revival_count: int = 0


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    cos = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


class BufferLimbo:
    """
    Pool de ideas muertas con memoria de contexto.

    Revive una idea si:
      1. d_ctx = angle(c_actual, c_fallo) > ctx_threshold_deg  (contexto cambió)
      2. cos(omega_idea, c_actual) > compat_threshold          (compatible con lo nuevo)
    """

    def __init__(self, ctx_threshold_deg: float = 30.0,
                 compat_threshold: float = 0.0):
        self.ideas: list[ContextualIdea] = []
        self.ctx_threshold_deg = ctx_threshold_deg
        self.compat_threshold = compat_threshold

    def add_dead(self, omega: torch.Tensor, context_omega: torch.Tensor,
                 label: str = "", domain: str = "") -> ContextualIdea:
        idea = ContextualIdea(
            omega=omega.clone(),
            context_omega=context_omega.clone(),
            status="dead",
            label=label,
            domain=domain,
        )
        self.ideas.append(idea)
        return idea

    def context_changed(self, idea: ContextualIdea,
                        c_actual: torch.Tensor) -> tuple[bool, float]:
        angle = _angle_deg(c_actual, idea.context_omega)
        return angle > self.ctx_threshold_deg, angle

    def compatible(self, idea: ContextualIdea,
                   c_actual: torch.Tensor) -> tuple[bool, float]:
        cos = _cos_sim(idea.omega, c_actual)
        return cos > self.compat_threshold, cos

    def check_revival(self, c_actual: torch.Tensor,
                      require_compat: bool = True
                      ) -> list[tuple[ContextualIdea, float, float]]:
        """Devuelve ideas muertas que pueden revivir: (idea, angle_ctx, cos_compat)."""
        result = []
        for idea in self.ideas:
            if idea.status != "dead":
                continue
            changed, angle = self.context_changed(idea, c_actual)
            if not changed:
                continue
            compat, cos = self.compatible(idea, c_actual)
            if require_compat and not compat:
                continue
            result.append((idea, angle, cos))
        return result

    def mark_winner(self, idea: ContextualIdea, context_omega: torch.Tensor) -> None:
        idea.status = "revived"
        idea.context_omega = context_omega.clone()
        idea.revival_count += 1

    def summary(self) -> dict:
        return {
            "total":   len(self.ideas),
            "dead":    sum(1 for i in self.ideas if i.status == "dead"),
            "revived": sum(1 for i in self.ideas if i.status == "revived"),
            "alive":   sum(1 for i in self.ideas if i.status == "alive"),
        }
