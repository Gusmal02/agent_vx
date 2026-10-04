"""
core/satisfaction_meter.py — Medidor de satisfacción / insatisfacción

Detecta cuándo el agente está estancado en un problema y necesita:
  1. Pivotar de perspectiva (cambiar enfoque)
  2. Mejorar sus herramientas (auto-mejora)
  3. Cambiar el subproblema atacado

Métricas observadas:
  - Progreso en hipótesis verificadas
  - Diversidad de enfoques intentados
  - Tiempo sin novedad (stagnation)
  - Calidad de las herramientas inventadas

"Después de cierto tiempo que vea que hay una cierta insatisfacción,
poder perseguirlo desde otra perspectiva."
"""

import time
from typing import List, Optional
from enum import Enum


class SatisfactionState(Enum):
    EXPLORING   = "exploring"      # activamente explorando, progreso normal
    STAGNATING  = "stagnating"     # sin progreso reciente
    PIVOTING    = "pivoting"       # cambiando de perspectiva
    IMPROVING   = "improving"      # mejorando sus herramientas


class SatisfactionMeter:
    """
    Monitorea el nivel de satisfacción del agente con su progreso.
    Emite señales de pivot cuando detecta estancamiento.
    """

    def __init__(self,
                 stagnation_window: int = 5,       # ciclos sin mejora → stagnation
                 pivot_threshold: int = 8,          # ciclos stagnating → forzar pivot
                 min_novelty: float = 0.05,         # mínima mejora para no contar como estancado
                 ):
        self.stagnation_window  = stagnation_window
        self.pivot_threshold    = pivot_threshold
        self.min_novelty        = min_novelty

        self._state             = SatisfactionState.EXPLORING
        self._progress_history: List[float] = []
        self._stagnation_count  = 0
        self._pivot_count       = 0
        self._last_pivot_time   = time.time()
        self._cycle             = 0

    # ── Actualización ─────────────────────────────────────────────────────────

    def update(self, progress_score: float) -> SatisfactionState:
        """
        Actualiza con el score de progreso del ciclo actual.
        progress_score: 0.0 (sin progreso) a 1.0 (máximo progreso).
        Devuelve el estado actual.
        """
        self._cycle += 1
        self._progress_history.append(progress_score)

        # Calcular mejora respecto a la ventana anterior
        window = self._progress_history[-self.stagnation_window:]
        if len(window) >= 2:
            improvement = window[-1] - min(window[:-1])
        else:
            improvement = progress_score

        if improvement < self.min_novelty:
            self._stagnation_count += 1
        else:
            self._stagnation_count = max(0, self._stagnation_count - 1)
            self._state = SatisfactionState.EXPLORING

        if self._stagnation_count >= self.pivot_threshold:
            self._state = SatisfactionState.PIVOTING
            self._stagnation_count = 0
            self._pivot_count += 1
            self._last_pivot_time = time.time()
        elif self._stagnation_count >= self.stagnation_window:
            self._state = SatisfactionState.STAGNATING

        return self._state

    # ── Consulta ──────────────────────────────────────────────────────────────

    @property
    def state(self) -> SatisfactionState:
        return self._state

    def needs_pivot(self) -> bool:
        return self._state == SatisfactionState.PIVOTING

    def needs_tool_improvement(self) -> bool:
        return self._state == SatisfactionState.STAGNATING

    def satisfaction_score(self) -> float:
        """0.0 = completamente insatisfecho, 1.0 = satisfecho."""
        if not self._progress_history:
            return 0.5
        recent = self._progress_history[-self.stagnation_window:]
        raw = sum(recent) / len(recent)
        stag_penalty = min(1.0, self._stagnation_count / self.pivot_threshold)
        return max(0.0, raw - 0.4 * stag_penalty)

    # ── Perspectivas de pivot ─────────────────────────────────────────────────

    def next_perspective(self, current_perspective: str,
                         available: List[str]) -> str:
        """
        Elige la siguiente perspectiva a intentar.
        Evita la perspectiva actual y elige de manera rotatoria.
        """
        options = [p for p in available if p != current_perspective]
        if not options:
            return current_perspective
        idx = self._pivot_count % len(options)
        return options[idx]

    def summary(self) -> dict:
        return {
            "state":            self._state.value,
            "cycle":            self._cycle,
            "stagnation_count": self._stagnation_count,
            "pivot_count":      self._pivot_count,
            "satisfaction":     round(self.satisfaction_score(), 3),
        }
