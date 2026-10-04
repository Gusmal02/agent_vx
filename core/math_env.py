"""
core/math_env.py — Entorno de refuerzo matemático (v0.0.9)

Reemplaza CartPole. El tejido aprende QUÉ acción matemática tomar
basándose en el historial de progreso actual.

Estado: vector de features del progreso matemático
Acción: índice en la lista de acciones del problema
Recompensa: progress_score del sandbox para esa acción

El tejido ahora piensa matemáticamente, no equilibra postes.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
from typing import List, Optional


HISTORY_LEN = 10    # ciclos de historia en el estado


class MathEnv(gym.Env):
    """
    Entorno de refuerzo donde el agente elige qué acción matemática ejecutar.

    observation_space (Box, float32):
      - [0:N_actions]          score promedio reciente por acción
      - [N_actions:2*N_actions] veces ejecutada cada acción (normalizado)
      - [2*N_actions]           hipótesis registradas (normalizado)
      - [2*N_actions+1]         confianza promedio de hipótesis
      - [2*N_actions+2]         ciclo actual (normalizado 0-1 sobre 500)
      - [2*N_actions+3]         stagnation_count normalizado
      - [2*N_actions+4]         tools_active normalizado

    action_space: Discrete(N_actions)
    """

    metadata = {"render_modes": []}

    def __init__(self,
                 actions: List[str],
                 sandbox,
                 problem: str,
                 meta_tracker,
                 satisfaction_meter,
                 tool_registry,
                 max_cycles: int = 500,
                 ):
        super().__init__()
        self.actions          = actions
        self.sandbox          = sandbox
        self.problem          = problem
        self.meta             = meta_tracker
        self.meter            = satisfaction_meter
        self.tool_registry    = tool_registry
        self.max_cycles       = max_cycles

        self._n               = len(actions)
        self._cycle           = 0
        self._score_history   = {a: deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
                                  for a in actions}
        self._exec_count      = {a: 0 for a in actions}

        # Dimensión del estado
        obs_dim = self._n * 2 + 5
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self._n)

        # Cache del último resultado para el bucle exterior
        self.last_result: dict = {}

    # ── Gymnasium API ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._cycle = 0
        for a in self.actions:
            self._score_history[a] = deque([0.0] * HISTORY_LEN, maxlen=HISTORY_LEN)
            self._exec_count[a]    = 0
        self.last_result = {}
        return self._observe(), {}

    def step(self, action_idx: int):
        action_name = self.actions[action_idx]
        self._cycle += 1
        self._exec_count[action_name] += 1

        # Ejecutar la acción en el sandbox (llamada al atacador externo)
        score, result = self._execute_action(action_name)

        self._score_history[action_name].append(score)
        self.last_result = {"action": action_name, "score": score, "result": result}

        obs        = self._observe()
        reward     = float(score)
        terminated = self._cycle >= self.max_cycles
        truncated  = False
        info       = {"action_name": action_name, "cycle": self._cycle}

        return obs, reward, terminated, truncated, info

    # ── Observación ──────────────────────────────────────────────────────────

    def _observe(self) -> np.ndarray:
        avg_scores = np.array([
            np.mean(self._score_history[a]) for a in self.actions
        ], dtype=np.float32)

        max_exec = max(self._exec_count.values()) or 1
        exec_norm = np.array([
            self._exec_count[a] / max_exec for a in self.actions
        ], dtype=np.float32)

        n_hyp      = min(1.0, len(self.meta._hypotheses) / 20.0)
        conf_avg   = float(np.mean([h["confidence"] for h in self.meta._hypotheses])
                           if self.meta._hypotheses else 0.0)
        cycle_norm = min(1.0, self._cycle / self.max_cycles)
        stag_norm  = min(1.0, self.meter._stagnation_count / self.meter.pivot_threshold)
        tools_norm = min(1.0, self.tool_registry.summary()["active"] / 10.0)

        return np.concatenate([
            avg_scores, exec_norm,
            [n_hyp, conf_avg, cycle_norm, stag_norm, tools_norm]
        ])

    # ── Ejecución de acción ──────────────────────────────────────────────────

    def _execute_action(self, action_name: str):
        """
        Delegado al atacador externo (run_v009.py inject via set_attacker).
        Por defecto devuelve 0 — el bucle principal inyecta la función real.
        """
        if self._attacker is None:
            return 0.0, {}
        return self._attacker(action_name, self._cycle)

    def set_attacker(self, fn):
        """Inyectar la función que ejecuta la acción en el sandbox."""
        self._attacker = fn

    _attacker = None

    def action_name(self, idx: int) -> str:
        return self.actions[idx]

    def summary(self) -> dict:
        return {
            "cycle":       self._cycle,
            "action_stats": {
                a: {
                    "exec":     self._exec_count[a],
                    "avg_score": round(float(np.mean(self._score_history[a])), 3),
                }
                for a in self.actions
            },
        }
