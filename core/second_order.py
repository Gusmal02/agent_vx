"""
core/second_order.py — Agente de Segundo Orden (v0.0.7)

El maestro (segundo orden) no corre episodios directamente.
Orquesta agentes de primer orden, evalúa sus resultados
y absorbe el tejido del ganador.

Flujo:
  1. Maestro define N variantes de configuración
  2. Lanza N agentes1° en paralelo (comparten biblioteca en solo-lectura)
  3. Evalúa cada uno con la misma semilla y biblioteca caliente
  4. Ganador = mejor media SIN regresión respecto a la línea base
  5. Absorción: exec_ids del ganador → exec_ids del maestro
     (el pensamiento no cambia, solo cómo actúa)

"No es que cambie tanto el que piensa sino el que actúa."
"""

from __future__ import annotations

import copy
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass, field

from core.proto_tissue_agent_v2 import ProtoTissueAgentV2


@dataclass
class FirstOrderResult:
    variant_name: str
    config: dict
    media: float
    max_reward: float
    fast_pct: float
    failures: dict
    agent: ProtoTissueAgentV2       # agente con el tejido entrenado


class SecondOrderAgent:
    """
    Orquestador de segundo orden.

    El maestro mantiene su propio tejido (pesos base) y lanza
    agentes de primer orden para explorar mejoras. El ganador
    transfiere su subcampo exec al maestro — el maestro absorbe,
    no se reinicia.
    """

    def __init__(self,
                 base_config: dict,
                 library,
                 seed: int = 42):
        self.base_config = copy.deepcopy(base_config)
        self.library     = library
        self.seed        = seed
        self._baseline_media: float = 0.0
        self._generation:     int   = 0
        self.absorption_log: List[dict] = []

        # Agente maestro (segundo orden)
        self.master = self._build_agent(base_config, seed)

    # ── Construcción ─────────────────────────────────────────────────────────

    def _build_agent(self, cfg: dict, seed: int) -> ProtoTissueAgentV2:
        return ProtoTissueAgentV2(
            N=cfg["N"], M=cfg["M"], K=cfg["K"],
            seed=seed,
            stagnation_threshold=cfg["stagnation_threshold"],
            filter_threshold=cfg["filter_threshold"],
            library=self.library,
            confidence_threshold=cfg["confidence_threshold"],
            t_settle=5,
            warmup_steps=50,
            subconscious_update_freq=5,
        )

    # ── Generación de variantes ───────────────────────────────────────────────

    def generate_variants(self,
                          diagnostico: str,
                          n_variants: int = 3,
                          ) -> List[dict]:
        """
        Genera N configuraciones variantes según el diagnóstico.
        Cada variante es una perturbación diferente de base_config.
        """
        cfg = self.base_config
        variants = []

        if diagnostico == "library_dominance":
            # Variante 1: bajar confidence para forzar más resonancia
            v = copy.deepcopy(cfg)
            v["confidence_threshold"] = max(0.72, cfg["confidence_threshold"] - 0.08)
            v["_variant_label"] = "conf_thr-0.08"
            variants.append(v)
            # Variante 2: más exploración epsilon
            v = copy.deepcopy(cfg)
            v["epsilon_start"] = min(0.85, cfg.get("epsilon_start", 0.5) + 0.15)
            v["_variant_label"] = "eps+0.15"
            variants.append(v)
            # Variante 3: tejido más grande para romper el patrón dominante
            v = copy.deepcopy(cfg)
            v["N"] = min(150, cfg["N"] + 20)
            v["confidence_threshold"] = max(0.72, cfg["confidence_threshold"] - 0.05)
            v["_variant_label"] = f"N+20_conf-0.05"
            variants.append(v)

        elif diagnostico == "limit":
            deltas = [(25, 0.0), (0, -0.05), (25, -0.05)]
            for dN, df in deltas:
                v = copy.deepcopy(cfg)
                v["N"]                = min(150, cfg["N"] + dN)
                v["filter_threshold"] = max(0.25, cfg["filter_threshold"] + df)
                v["_variant_label"]   = f"N+{dN}_f{df:+.2f}"
                variants.append(v)

        elif diagnostico in ("stale", "ignorance"):
            for de in [0.10, 0.15, 0.20]:
                v = copy.deepcopy(cfg)
                v["epsilon_start"]    = min(0.90, cfg["epsilon_start"] + de)
                v["_variant_label"]   = f"eps+{de:.2f}"
                variants.append(v)

        else:
            # Sin diagnóstico claro — exploración diversificada
            for de in [0.08, 0.12]:
                v = copy.deepcopy(cfg)
                v["epsilon_start"] = min(0.85, cfg["epsilon_start"] + de)
                v["_variant_label"] = f"eps+{de:.2f}"
                variants.append(v)
            v2 = copy.deepcopy(cfg)
            v2["confidence_threshold"] = max(0.75, cfg["confidence_threshold"] - 0.05)
            v2["_variant_label"] = "conf_thr-0.05"
            variants.append(v2)

        return variants[:n_variants]

    # ── Evaluación en caliente ────────────────────────────────────────────────

    def evaluate_variant(self,
                         cfg: dict,
                         evaluator_fn,
                         env_name: str,
                         n_episodes: int,
                         seed: int,
                         ) -> FirstOrderResult:
        """
        Construye un agente de primer orden con `cfg` y lo evalúa.
        El agente comparte la biblioteca (solo lectura: _clone_mode=True).
        """
        agent = self._build_agent(cfg, seed)
        agent._clone_mode = True   # no escribe en biblioteca

        result = evaluator_fn(agent, env_name, n_episodes)

        r = result["reporte_agente"]
        total_dec = r['n_fast_decisions'] + r['n_resonant_decisions']
        fast_pct  = r['n_fast_decisions'] / max(1, total_dec)

        return FirstOrderResult(
            variant_name = cfg.get("_variant_label", "?"),
            config       = copy.deepcopy(cfg),
            media        = result["recompensa_media"],
            max_reward   = result["recompensa_max"],
            fast_pct     = fast_pct,
            failures     = r.get("failures", {}),
            agent        = agent,
        )

    # ── Selección del ganador ─────────────────────────────────────────────────

    def select_winner(self,
                      results: List[FirstOrderResult],
                      baseline_media: float,
                      regression_tolerance: float = 0.90,
                      ) -> Optional[FirstOrderResult]:
        """
        Ganador = mejor media entre los que no regresan más del
        `regression_tolerance` respecto a la línea base.
        """
        floor = baseline_media * regression_tolerance
        candidates = [r for r in results if r.media >= floor]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.media)

    # ── Absorción ─────────────────────────────────────────────────────────────

    def absorb(self, winner: FirstOrderResult, alpha: float = 0.40) -> None:
        """
        El maestro absorbe el subcampo exec del agente ganador.

        Mezcla alpha: nuevo_exec = (1-α)*maestro_exec + α*ganador_exec
        α=1.0 sería reemplazo total; α=0.4 es absorción gradual.

        El pensamiento (perc, imag, sub, mem) permanece intacto en el maestro.
        Solo actúa diferente — exec cambia.
        """
        master_tissue  = self.master.tissue
        winner_tissue  = winner.agent.tissue
        exec_ids       = self.master.exec_ids

        with torch.no_grad():
            for nid in exec_ids:
                q_master = master_tissue.nodes[nid].q
                q_winner = winner_tissue.nodes[nid].q
                q_new    = F.normalize(
                    (1 - alpha) * q_master + alpha * q_winner, dim=-1
                )
                master_tissue.nodes[nid].q = q_new

        # Actualizar config del maestro con la del ganador
        old_cfg = copy.deepcopy(self.base_config)
        self.base_config.update({
            k: v for k, v in winner.config.items()
            if not k.startswith("_")
        })
        self.master.confidence_threshold = self.base_config["confidence_threshold"]
        self.master.filter_threshold     = self.base_config["filter_threshold"]

        self._generation += 1
        self.absorption_log.append({
            "generation":  self._generation,
            "winner":      winner.variant_name,
            "media_antes": self._baseline_media,
            "media_ganador": winner.media,
            "config_before": old_cfg,
            "config_after":  copy.deepcopy(self.base_config),
        })

    # ── Ciclo completo ────────────────────────────────────────────────────────

    def run_generation(self,
                       diagnostico: str,
                       evaluator_fn,
                       env_name: str,
                       n_episodes: int,
                       seed: int,
                       verbose: bool = True,
                       ) -> dict:
        """
        Un ciclo completo: generar variantes → evaluar → absorber al ganador.
        Devuelve un resumen del ciclo.
        """
        variants = self.generate_variants(diagnostico, n_variants=3)

        if verbose:
            print(f"\n  [2°orden gen {self._generation+1}] "
                  f"Evaluando {len(variants)} variantes "
                  f"(diagnóstico: {diagnostico})...")

        results: List[FirstOrderResult] = []
        for cfg in variants:
            r = self.evaluate_variant(cfg, evaluator_fn, env_name, n_episodes, seed + 10)
            results.append(r)
            if verbose:
                print(f"    Variante '{r.variant_name}': "
                      f"media={r.media:.1f}  fast={r.fast_pct:.0%}")

        winner = self.select_winner(results, self._baseline_media)

        if winner is not None and winner.media > self._baseline_media:
            if verbose:
                print(f"  → Absorbiendo '{winner.variant_name}' "
                      f"(media {winner.media:.1f} > base {self._baseline_media:.1f})")
            self.absorb(winner)
            promoted = True
        else:
            if verbose:
                nm = winner.variant_name if winner else "ninguno"
                print(f"  → Ninguna variante mejora la línea base "
                      f"({self._baseline_media:.1f}). Mantener maestro.")
            promoted = False

        return {
            "generation":      self._generation,
            "diagnostico":     diagnostico,
            "variants_tried":  [(r.variant_name, round(r.media, 1)) for r in results],
            "winner":          winner.variant_name if winner else None,
            "promoted":        promoted,
            "media_ganador":   winner.media if winner else None,
        }

    def set_baseline(self, media: float) -> None:
        self._baseline_media = media
