"""
domain_memory.py — Módulo de Memoria (wrapper semántico sobre la biblioteca)

Separa dos tipos de memoria:
  - EPISÓDICA: qué pasó en episodios anteriores (ResonantLibrary)
  - SEMÁNTICA: patrones generalizados aprendidos (vectores de centroide)

El ContextModule (memoria de trabajo) consulta este módulo cuando necesita
recordar qué funciona en situaciones similares a la actual.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple


class MemoryModule:
    """
    Interfaz semántica sobre la ResonantLibrary.

    En lugar de que el agente consulte la biblioteca directamente,
    este módulo centraliza:
      - qué tipo de memoria existe (episódica vs semántica)
      - si la memoria es fresca o stale
      - si hay suficiente para confiar en ella
    """

    def __init__(self, library, env_name: str = "unknown"):
        self._lib = library
        self._env_name = env_name
        self._semantic_centroids: List[Tuple[torch.Tensor, float]] = []
        self._last_consolidation_count: int = 0

    # ── Configuración ─────────────────────────────────────────────────────────

    def set_env(self, env_name: str) -> None:
        self._env_name = env_name

    # ── Consulta episódica ────────────────────────────────────────────────────

    def query_episodic(self,
                       omega: torch.Tensor,
                       min_samples: int = 5,
                       ) -> Optional[Tuple[int, float]]:
        """
        Consulta la memoria episódica para omega dado.
        Devuelve (action_idx, confidence) o None si no hay suficiente memoria.
        """
        if self._lib is None:
            return None
        result = self._lib.query_resonance(
            omega, min_samples=min_samples, env_name=self._env_name
        )
        if result is None:
            return None
        action_idx, confidence = result
        return (action_idx, confidence)

    def episodic_count(self) -> int:
        if self._lib is None:
            return 0
        return self._lib.count(env_name=self._env_name)

    # ── Estado de la memoria ──────────────────────────────────────────────────

    def memory_status(self) -> str:
        """
        Diagnóstica el estado de la memoria episódica.
        Devuelve: 'empty' | 'sparse' | 'developing' | 'mature' | 'dominant'
        """
        n = self.episodic_count()
        if n == 0:
            return "empty"
        elif n < 10:
            return "sparse"
        elif n < 30:
            return "developing"
        elif n < 100:
            return "mature"
        else:
            return "dominant"

    def is_trustworthy(self, confidence_threshold: float = 0.93) -> bool:
        """¿Tiene la memoria suficiente calidad para confiar en ella?"""
        return self.memory_status() in ("mature", "dominant")

    # ── Memoria semántica (centroides) ────────────────────────────────────────

    def consolidate_semantic(self,
                             recent_omegas: List[torch.Tensor],
                             recent_rewards: List[float],
                             min_reward: float = 0.0) -> None:
        """
        Actualiza centroides semánticos con los omegas recientes que obtuvieron
        recompensa positiva. Llamar al final de episodios buenos.
        """
        if not recent_omegas:
            return
        good = [
            (o, r) for o, r in zip(recent_omegas, recent_rewards)
            if r > min_reward
        ]
        if not good:
            return
        stacked = torch.stack([o for o, _ in good])
        centroid = F.normalize(stacked.mean(0), dim=-1)
        avg_r = sum(r for _, r in good) / len(good)
        self._semantic_centroids.append((centroid, avg_r))
        # Mantener solo los mejores 20 centroides
        if len(self._semantic_centroids) > 20:
            self._semantic_centroids.sort(key=lambda x: x[1], reverse=True)
            self._semantic_centroids = self._semantic_centroids[:20]

    def semantic_hint(self, omega: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Busca el centroide semántico más cercano a omega.
        Devuelve el centroide si la similitud es alta, None si no hay match.
        """
        if not self._semantic_centroids:
            return None
        best_cos, best_c = -1.0, None
        for centroid, _ in self._semantic_centroids:
            cos = float(F.cosine_similarity(omega.unsqueeze(0), centroid.unsqueeze(0)))
            if cos > best_cos:
                best_cos, best_c = cos, centroid
        if best_cos > 0.85:
            return best_c
        return None

    def summary(self) -> dict:
        return {
            "env": self._env_name,
            "episodic_count": self.episodic_count(),
            "status": self.memory_status(),
            "semantic_centroids": len(self._semantic_centroids),
        }
