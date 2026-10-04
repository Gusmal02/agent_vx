"""
run_v011.py — Agente vX v0.1.1

Cambios respecto a v0.1.0:
  - EpistemicState: cobertura / incertidumbre / anomalía por acción
  - Máquina de fases: SURVEY → FOCUS → CONSOLIDATE
  - Selector de acción dirigido por curiosidad (no round-robin en FOCUS)
  - Ciclo de vida de hipótesis sin SubagentPool (evidence_bank propio)
  - Oracle se dispara en momentos de síntesis, no de estancamiento
  - Para cuando produce finding_document (solución o inconclusivo)
  - Safety net: --max-hours (default 8h)

Uso:
    uv run python run_v011.py --problem causal
    uv run python run_v011.py --problem continual --max-hours 8
    uv run python run_v011.py --problem causal --resume
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from library.store              import ResonantLibrary
from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from core.second_order          import SecondOrderAgent
from core.direct_executor       import DirectExecutor
from core.satisfaction_meter    import SatisfactionMeter
from core.checkpoint            import CheckpointManager
from core.tool_registry         import ToolRegistry
from core.math_env              import MathEnv
from core.oracle_client         import OracleClient
from core.subagent_pool         import SubagentPool
from core.hypothesis_tracker    import HypothesisTracker
from core.research_memory       import ResearchMemory
from core.action_inventor       import ActionInventor
from core.corpus.ml_problems_corpus import (
    get_ml_corpus, list_ml_corpus_for_domain
)

load_dotenv()

VERSION = "v0.1.1"

# ── Configuración ─────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "N": 100, "M": 3, "K": 3,
    "stagnation_threshold": 8,
    "filter_threshold":     0.35,
    "confidence_threshold": 0.88,
    "epsilon_start":        0.50,
    "n_episodios":          15,
}

PROBLEM_ACTIONS = {
    "causal": [
        "backdoor_adjustment",
        "do_calculus_test",
        "irm_vs_erm",
        "causal_discovery",
        "counterfactual_bounds",
    ],
    "continual": [
        "gradient_interference",
        "ewc_retention",
        "fisher_geometry",
        "task_similarity",
        "replay_vs_finetune",
    ],
}

PERSPECTIVE_EMPHASIS = {
    "mathematician":   [0, 1, 2, 3, 4],
    "physicist":       [2, 3, 1, 0, 4],
    "algorithmist":    [4, 0, 2, 1, 3],
    "python_engineer": [1, 0, 3, 2, 4],
}

ORACLE_BUDGET = 2.50

# ── Fases ─────────────────────────────────────────────────────────────────────

PHASE_SURVEY      = "SURVEY"
PHASE_FOCUS       = "FOCUS"
PHASE_CONSOLIDATE = "CONSOLIDATE"

# ── Thresholds ────────────────────────────────────────────────────────────────

SURVEY_MIN_CYCLES_PER_ACTION = 3    # ciclos por acción para salir de SURVEY
FOCUS_MAX_CYCLES             = 40   # ciclos en FOCUS antes de forzar CONSOLIDATE
SUPPORTED_THRESHOLD_SCORE    = 0.65 # score mínimo para contar evidencia
SUPPORTED_EVIDENCE_WINDOW    = 5    # ciclos consecutivos para declarar SUPPORTED
SUPPORTED_EVIDENCE_VARIANCE  = 0.04 # varianza máxima del bloque
ANOMALY_SATURATION           = 0.92 # score saturado (acción ya aprendida)
ANOMALY_FLAT_VAR             = 0.001 # varianza cero = acción bloqueada
SOLUTION_MIN_SUPPORTED       = 2    # hipótesis soportadas mínimas
SOLUTION_MIN_RICHNESS        = 0.55 # riqueza mínima del finding
MIN_CYCLE_SEC                = 15.0
MEM_NOVELTY_THRESHOLD        = 0.12


# ══════════════════════════════════════════════════════════════════════════════
# EpistemicState
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicState:
    """Modelo del estado del conocimiento del agente por acción."""

    def __init__(self, actions: list[str]):
        self.actions     = actions
        self.phase       = PHASE_SURVEY
        self.coverage    = {a: 0.0 for a in actions}   # 0–1: fracción con memoria
        self.uncertainty = {a: 1.0 for a in actions}   # varianza reciente
        self.saturated   = {a: False for a in actions}  # score > 0.92 (resuelta)
        self.flat        = {a: False for a in actions}  # var < 0.001 (atascada)
        self.focus_action: str | None = None
        self._cycles_in_phase  = 0
        self._survey_count     = {a: 0 for a in actions}
        self._focus_cycles     = 0
        self._last_action: str | None = None
        self._consecutive: int = 0  # veces consecutivas que se eligió la misma acción

    def update(self, action: str, score: float,
               hist: list[float], memory_added: bool) -> None:
        self._survey_count[action] = self._survey_count.get(action, 0) + 1
        self._cycles_in_phase += 1

        if memory_added:
            self.coverage[action] = min(1.0, self.coverage[action] + 0.12)
        else:
            self.coverage[action] = max(0.0, self.coverage[action] - 0.01)

        if len(hist) >= 3:
            window = hist[-min(5, len(hist)):]
            self.uncertainty[action] = float(np.var(window))

        if len(hist) >= 5:
            avg = sum(hist[-5:]) / 5
            var = float(np.var(hist[-5:]))
            self.saturated[action] = avg > ANOMALY_SATURATION
            self.flat[action]      = var < ANOMALY_FLAT_VAR

        if self.phase == PHASE_FOCUS:
            self._focus_cycles += 1

        # Actualizar contador de consecutivos
        if action == self._last_action:
            self._consecutive += 1
        else:
            self._consecutive = 1
            self._last_action = action

    @property
    def anomaly(self) -> dict[str, bool]:
        return {a: self.saturated[a] or self.flat[a] for a in self.actions}

    def most_curious_action(self, active: list[str]) -> str:
        """
        Acción con mayor curiosidad epistémica.
        - Saturada (resuelta): sin bonus — ya sabemos lo que hace
        - Plana (atascada): penalización — el Limbo la manejará
        - Incertidumbre alta: bonus — hay algo que aprender
        Además: si llevamos >3 ciclos consecutivos en la misma acción, se fuerza rotación.
        """
        # Forzar rotación si llevamos demasiados ciclos seguidos en la misma
        if self._consecutive >= 3 and len(active) > 1:
            candidates = [a for a in active if a != self._last_action]
        else:
            candidates = active

        best, best_score = candidates[0], -999.0
        for a in candidates:
            s  = self.uncertainty.get(a, 1.0) * 2.0
            if self.flat.get(a, False):
                s -= 0.6           # plana: deprioritizar (Limbo la pondrá dormante)
            elif self.saturated.get(a, False):
                s -= 0.1           # saturada: leve penalización (ya aprendida)
            s -= self.coverage.get(a, 0.0) * 0.3
            if s > best_score:
                best, best_score = a, s
        return best

    def all_surveyed(self) -> bool:
        return all(
            self._survey_count.get(a, 0) >= SURVEY_MIN_CYCLES_PER_ACTION
            for a in self.actions
        )

    def should_transition(self, n_supported: int) -> str | None:
        if self.phase == PHASE_SURVEY:
            if self.all_surveyed():
                if any(self.anomaly.values()) or n_supported >= 1:
                    return PHASE_FOCUS
        elif self.phase == PHASE_FOCUS:
            if n_supported >= SOLUTION_MIN_SUPPORTED:
                return PHASE_CONSOLIDATE
            if self._focus_cycles >= FOCUS_MAX_CYCLES:
                return PHASE_CONSOLIDATE
        return None

    def enter_phase(self, phase: str) -> None:
        self.phase = phase
        self._cycles_in_phase = 0
        if phase == PHASE_FOCUS:
            self._focus_cycles = 0
        print(f"\n  [EpistemicState] → FASE {phase}")

    def summary(self) -> dict:
        return {
            "phase":        self.phase,
            "focus_cycles": self._focus_cycles,
            "coverage":     {a: round(v, 3) for a, v in self.coverage.items()},
            "uncertainty":  {a: round(v, 5) for a, v in self.uncertainty.items()},
            "saturated":    dict(self.saturated),
            "flat":         dict(self.flat),
            "survey_count": dict(self._survey_count),
            "consecutive":  self._consecutive,
            "last_action":  self._last_action,
        }


# ══════════════════════════════════════════════════════════════════════════════
# EvidenceBank — ciclo de vida de hipótesis sin SubagentPool
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceBank:
    """
    Acumula evidencia por acción y declara hipótesis SUPPORTED cuando
    hay una racha consistente de scores altos.
    No depende de SubagentPool.
    """

    def __init__(self, actions: list[str]):
        self.actions    = actions
        # scores recientes (ventana deslizante)
        self._scores: dict[str, list[float]] = {a: [] for a in actions}
        # hipótesis soportadas: {action: {"score_mean": f, "evidence_cycles": n}}
        self.supported:  dict[str, dict] = {}
        self.rejected:   dict[str, dict] = {}
        self._contradiction: tuple[str, str] | None = None

    def record(self, action: str, score: float) -> None:
        buf = self._scores[action]
        buf.append(score)
        if len(buf) > 20:
            buf.pop(0)
        self._check_supported(action)
        self._check_contradiction()

    def _check_supported(self, action: str) -> None:
        buf = self._scores[action]
        if len(buf) < SUPPORTED_EVIDENCE_WINDOW:
            return
        window = buf[-SUPPORTED_EVIDENCE_WINDOW:]
        mean_s = sum(window) / len(window)
        var_s  = float(np.var(window))
        if mean_s >= SUPPORTED_THRESHOLD_SCORE and var_s <= SUPPORTED_EVIDENCE_VARIANCE:
            if action not in self.supported:
                self.supported[action] = {
                    "score_mean": round(mean_s, 4),
                    "score_var":  round(var_s, 5),
                    "evidence_cycles": SUPPORTED_EVIDENCE_WINDOW,
                    "window": [round(s, 4) for s in window],
                }
                print(f"  [EvidenceBank] SUPPORTED: '{action}' "
                      f"mean={mean_s:.3f} var={var_s:.4f}")

    def _check_contradiction(self) -> None:
        high = [a for a, s in self._scores.items()
                if len(s) >= 3 and sum(s[-3:]) / 3 > 0.70]
        low  = [a for a, s in self._scores.items()
                if len(s) >= 3 and sum(s[-3:]) / 3 < 0.30]
        if high and low:
            pair = (high[0], low[0])
            if pair != self._contradiction:
                self._contradiction = pair
                print(f"  [EvidenceBank] CONTRADICCIÓN: '{high[0]}' alto vs '{low[0]}' bajo")

    def has_contradiction(self) -> bool:
        return self._contradiction is not None

    def pop_contradiction(self) -> tuple[str, str] | None:
        c = self._contradiction
        self._contradiction = None
        return c

    def n_supported(self) -> int:
        return len(self.supported)

    def richness(self) -> float:
        """
        Métrica de riqueza del hallazgo:
          0.4 * (n_supported / SOLUTION_MIN_SUPPORTED)
          0.3 * cross_action (¿el finding cruza dominios de acción?)
          0.3 * evidence_consistency (1 - mean_var)
        """
        if not self.supported:
            return 0.0
        n_frac = min(1.0, self.n_supported() / SOLUTION_MIN_SUPPORTED)
        # cross_action: si hay ≥2 acciones soportadas de tipos distintos
        cross  = min(1.0, (self.n_supported() - 1) / 2) if self.n_supported() > 1 else 0.0
        mean_var = (sum(v["score_var"] for v in self.supported.values())
                    / len(self.supported))
        consistency = max(0.0, 1.0 - mean_var * 20)  # 0.05 var → 0 consistency
        return round(0.4 * n_frac + 0.3 * cross + 0.3 * consistency, 4)

    def summary(self) -> dict:
        return {
            "supported": self.supported,
            "rejected":  self.rejected,
            "richness":  self.richness(),
            "n_supported": self.n_supported(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# FindingDocument
# ══════════════════════════════════════════════════════════════════════════════

def build_finding_document(
    result_type: str,           # "convergent" | "inconclusive" | "contradiction"
    evidence_bank: EvidenceBank,
    memory,
    tracker,
    meter,
    cycles: int,
    oracle_synthesis: str = "",
) -> dict:
    top_memory = memory.to_corpus_knowledge(top_k=5)
    report = tracker.self_report()
    tracked_hyps = report.get("hypotheses", [])
    # self_report puede devolver int (conteo) en lugar de lista
    if not isinstance(tracked_hyps, list):
        open_questions = []
    else:
        open_questions = [
            h for h in tracked_hyps
            if h not in evidence_bank.supported
        ][:5]

    return {
        "version":             VERSION,
        "result":              result_type,
        "cycles_to_finding":   cycles,
        "finding_richness":    evidence_bank.richness(),
        "supported_hypotheses": evidence_bank.supported,
        "rejected_hypotheses":  evidence_bank.rejected,
        "open_questions":       open_questions,
        "meter_summary":        meter.summary(),
        "memory_top5":          top_memory,
        "oracle_synthesis":     oracle_synthesis,
        "timestamp":            datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Causalidad (igual que v0.1.0)
# ══════════════════════════════════════════════════════════════════════════════

def _causal_action(action: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker,
                   log: dict) -> float:

    if action == "backdoor_adjustment":
        code = f"""
import numpy as np
np.random.seed({cycle * 17 + 3})
results = []
for trial in range(5):
    n = 1500
    alpha_zx = 0.6 + np.random.uniform(-0.2, 0.2)
    alpha_zy = 0.4 + np.random.uniform(-0.2, 0.2)
    alpha_xy = 0.5 + np.random.uniform(-0.3, 0.3)
    Z = np.random.randn(n)
    X = alpha_zx * Z + np.random.randn(n) * 0.4
    Y = alpha_xy * X + alpha_zy * Z + np.random.randn(n) * 0.4
    b_naive = np.cov(X, Y)[0,1] / (np.var(X) + 1e-9)
    A = np.column_stack([X, Z, np.ones(n)])
    b_adj = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    bias_n = abs(b_naive - alpha_xy)
    bias_a = abs(b_adj   - alpha_xy)
    results.append({{"true": round(float(alpha_xy),4),
                     "naive": round(float(b_naive),4),
                     "adjusted": round(float(b_adj),4),
                     "bias_naive": round(float(bias_n),4),
                     "bias_adj":   round(float(bias_a),4),
                     "reduction":  round(float(max(0, bias_n-bias_a)/(bias_n+1e-9)),4)}})
mean_red = sum(r["reduction"] for r in results) / len(results)
score = min(1.0, 0.2 + mean_red * 0.8)
_result = {{"action": "backdoor_adjustment", "trials": results,
            "mean_bias_reduction": round(float(mean_red),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"backdoor_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.6:
            tracker.record(
                f"Ciclo {cycle}: ajuste backdoor reduce sesgo causal "
                f"{res.get('mean_bias_reduction',0):.0%} en promedio",
                domain="causal", confidence=0.72,
                evidence="5 trials, ajuste por Z",
            )
        return score

    elif action == "do_calculus_test":
        code = f"""
import numpy as np
import networkx as nx
np.random.seed({cycle * 13 + 7})
n_dags = 8
results = []
for _ in range(n_dags):
    structure = np.random.choice(["simple", "mediator", "chain"], p=[0.4, 0.3, 0.3])
    n = 800
    Z = np.random.randn(n)
    W = np.random.randn(n)
    if structure == "simple":
        X = 0.7*Z + np.random.randn(n)*0.4
        Y = 0.5*X + 0.4*Z + np.random.randn(n)*0.4
        true_ace = 0.5
    elif structure == "mediator":
        X = np.random.randn(n)
        W = 0.8*X + np.random.randn(n)*0.3
        Y = 0.6*W + np.random.randn(n)*0.4
        true_ace = 0.8 * 0.6
    else:
        X = 0.7*Z + np.random.randn(n)*0.3
        V = 0.5*X + np.random.randn(n)*0.4
        Y = V.copy()
        true_ace = 0.5
    b_naive = np.cov(X, Y)[0,1] / (np.var(X)+1e-9)
    A = np.column_stack([X, Z, np.ones(n)])
    b_adj = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    err_naive = abs(b_naive - true_ace)
    err_adj   = abs(b_adj   - true_ace)
    results.append({{"structure": structure, "true_ace": round(true_ace,3),
                     "naive": round(float(b_naive),3), "adjusted": round(float(b_adj),3),
                     "err_naive": round(float(err_naive),4), "err_adj": round(float(err_adj),4),
                     "adj_better": bool(err_adj < err_naive)}})
frac_better = sum(1 for r in results if r["adj_better"]) / len(results)
score = min(1.0, 0.3 + frac_better * 0.7)
_result = {{"action": "do_calculus_test", "n_dags": n_dags, "results": results,
            "frac_adj_better": round(frac_better,3), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"docalc_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.65:
            tracker.record(
                f"Ciclo {cycle}: do-calculus identifica efecto causal en "
                f"{res.get('frac_adj_better',0):.0%} de DAGs aleatorios",
                domain="causal", confidence=0.70,
                evidence=f"n_dags={res.get('n_dags')}, ajuste backdoor",
            )
        return score

    elif action == "irm_vs_erm":
        code = f"""
import numpy as np
np.random.seed({cycle * 11 + 5})
n_trials = 6
trial_results = []
for trial in range(n_trials):
    d = 10
    n_env = 400
    true_coef = np.zeros(d); true_coef[0] = 1.5; true_coef[1] = -1.0
    env_errors = []
    for gamma in [0.9, -0.9]:
        X = np.random.randn(n_env, d)
        X[:, 2] = gamma * (true_coef[:2] @ X[:, :2].T) + np.random.randn(n_env)*0.1
        Y = X @ true_coef + np.random.randn(n_env)*0.5
        b_erm = np.linalg.lstsq(np.column_stack([X, np.ones(n_env)]), Y, rcond=None)[0][:-1]
        env_errors.append(float(np.mean((X @ b_erm - Y)**2)))
    X_test = np.random.randn(n_env, d)
    Y_test = X_test @ true_coef + np.random.randn(n_env)*0.5
    X_all = np.vstack([np.random.randn(n_env, d) for _ in range(2)])
    gammas = [0.9, -0.9]
    for i, g in enumerate(gammas):
        X_all[i*n_env:(i+1)*n_env, 2] = g * (true_coef[:2] @ X_all[i*n_env:(i+1)*n_env, :2].T)
    Y_all = X_all @ true_coef + np.random.randn(2*n_env)*0.5
    b_erm_all = np.linalg.lstsq(np.column_stack([X_all, np.ones(2*n_env)]), Y_all, rcond=None)[0][:-1]
    mse_erm_test = float(np.mean((X_test @ b_erm_all - Y_test)**2))
    b_irm = np.linalg.lstsq(np.column_stack([X_all[:,:2], np.ones(2*n_env)]), Y_all, rcond=None)[0]
    mse_irm_test = float(np.mean((X_test[:,:2] @ b_irm[:2] + b_irm[2] - Y_test)**2))
    trial_results.append({{"mse_erm_test": round(mse_erm_test,4),
                           "mse_irm_test": round(mse_irm_test,4),
                           "irm_wins": bool(mse_irm_test < mse_erm_test),
                           "gap": round(float(mse_erm_test - mse_irm_test),4)}})
frac_irm_wins = sum(1 for r in trial_results if r["irm_wins"]) / len(trial_results)
mean_gap = sum(r["gap"] for r in trial_results) / len(trial_results)
score = min(1.0, 0.2 + frac_irm_wins * 0.5 + max(0, mean_gap)*0.3)
_result = {{"action": "irm_vs_erm", "trials": trial_results,
            "frac_irm_wins": round(frac_irm_wins,3),
            "mean_gap": round(float(mean_gap),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"irm_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.6:
            tracker.record(
                f"Ciclo {cycle}: IRM supera ERM en entorno test en "
                f"{res.get('frac_irm_wins',0):.0%} de trials "
                f"(gap medio={res.get('mean_gap',0):.3f})",
                domain="causal", confidence=0.68,
                evidence="entornos gamma=±0.9, test gamma=0",
            )
        return score

    elif action == "causal_discovery":
        code = f"""
import numpy as np
np.random.seed({cycle * 19 + 11})
n_graphs = 5
results = []
for _ in range(n_graphs):
    n = 600
    d = 5
    true_adj = (np.random.rand(d, d) > 0.6).astype(float)
    true_adj = np.triu(true_adj, k=1)
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(true_adj[:, j] > 0)[0]
        noise = np.random.randn(n) * 0.5
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ true_adj[parents, j] + noise
        else:
            X[:, j] = noise
    estimated_adj = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            others = [k for k in range(d) if k != i and k != j]
            if others:
                Z = X[:, others]
                Xi_res = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
                Xj_res = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
                corr = np.corrcoef(Xi_res, Xj_res)[0, 1]
            else:
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if abs(corr) > 0.15:
                estimated_adj[i, j] = 1
                estimated_adj[j, i] = 1
    true_skeleton = ((true_adj + true_adj.T) > 0).astype(float)
    np.fill_diagonal(true_skeleton, 0)
    np.fill_diagonal(estimated_adj, 0)
    shd = int(np.sum(np.abs(true_skeleton - estimated_adj)))
    max_edges = d * (d-1)
    shd_norm = shd / (max_edges + 1e-9)
    results.append({{"n_true_edges": int(true_skeleton.sum()//2),
                     "n_estimated": int(estimated_adj.sum()//2),
                     "shd": shd, "shd_norm": round(float(shd_norm),4)}})
mean_shd = sum(r["shd_norm"] for r in results) / len(results)
score = min(1.0, max(0.0, 1.0 - mean_shd * 3))
_result = {{"action": "causal_discovery", "n_graphs": n_graphs, "results": results,
            "mean_shd_norm": round(float(mean_shd),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"discovery_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: PC algorithm recupera esqueleto causal "
                f"SHD={res.get('mean_shd_norm',1):.3f} normalizado",
                domain="causal", confidence=0.65,
                evidence=f"n_graphs={res.get('n_graphs')}, d=5 variables",
            )
        return score

    else:  # counterfactual_bounds
        code = f"""
import numpy as np
np.random.seed({cycle * 23 + 13})
results = []
for trial in range(6):
    n = 1000
    p_u = 0.4
    U = (np.random.rand(n) < p_u).astype(int)
    X = ((np.random.rand(n) < 0.3 + 0.4 * U)).astype(int)
    Y = ((np.random.rand(n) < 0.2 + 0.5 * X + 0.3 * U)).astype(int)
    p_y_x0 = Y[X==0].mean() if (X==0).sum() > 0 else 0.5
    p_y_x1 = Y[X==1].mean() if (X==1).sum() > 0 else 0.5
    p_x1   = X.mean()
    lb = max(p_y_x1 * p_x1 - (1-p_y_x0)*(1-p_x1),
             -(p_y_x0 * (1-p_x1) + (1-p_y_x1) * p_x1))
    ub = min(p_y_x1 * p_x1 + p_y_x0 * (1-p_x1),
             p_x1 + p_y_x0 - 2 * p_y_x0 * p_x1 + (1 - p_y_x1 + p_y_x0) * p_x1)
    Y_do1 = (np.random.rand(n) < 0.2 + 0.5 * 1 + 0.3 * U).astype(int)
    Y_do0 = (np.random.rand(n) < 0.2 + 0.5 * 0 + 0.3 * U).astype(int)
    true_ace = float(Y_do1.mean() - Y_do0.mean())
    in_bounds = float(lb) <= true_ace <= float(ub)
    width = float(ub) - float(lb)
    results.append({{"lb": round(float(lb),4), "ub": round(float(ub),4),
                     "true_ace": round(true_ace,4), "in_bounds": in_bounds,
                     "width": round(width,4)}})
frac_valid = sum(1 for r in results if r["in_bounds"]) / len(results)
mean_width = sum(r["width"] for r in results) / len(results)
score = min(1.0, 0.2 + frac_valid * 0.5 + max(0, 0.5 - mean_width) * 0.6)
_result = {{"action": "counterfactual_bounds", "trials": results,
            "frac_bounds_valid": round(frac_valid,3),
            "mean_width": round(float(mean_width),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"ctf_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: cotas de Tian-Pearl válidas en "
                f"{res.get('frac_bounds_valid',0):.0%} de trials "
                f"(ancho medio={res.get('mean_width',1):.3f})",
                domain="causal", confidence=0.67,
                evidence="SCM binario con confounder U no observable",
            )
        return score


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Olvido Catastrófico (igual que v0.1.0)
# ══════════════════════════════════════════════════════════════════════════════

def _continual_action(action: str, cycle: int,
                      executor: DirectExecutor,
                      tracker: HypothesisTracker,
                      log: dict) -> float:

    if action == "gradient_interference":
        code = f"""
import numpy as np
np.random.seed({cycle * 17 + 3})
d_in = 20
n    = 300
results = []
for pair_idx in range(8):
    W = np.random.randn(1, d_in) * 0.1
    X_A = np.random.randn(n, d_in); X_A[:, 1::2] *= 0.05
    y_A = (X_A[:, ::2].sum(axis=1) > 0).astype(float)
    angle = ({cycle} * 0.1 + pair_idx * 0.3) % np.pi
    X_B = np.random.randn(n, d_in); X_B[:, ::2] *= 0.05
    y_B = (np.cos(angle)*X_B[:, 1::2].sum(1) + np.sin(angle)*X_B[:, ::2].sum(1) > 0).astype(float)
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def grad(X, y, W):
        p = sigmoid(X @ W.T)[:,0]; e = p - y
        return (e[:,None]*X).mean(axis=0)
    gA = grad(X_A, y_A, W)
    gB = grad(X_B, y_B, W)
    cos = float(np.dot(gA,gB) / (np.linalg.norm(gA)*np.linalg.norm(gB)+1e-9))
    results.append({{"pair": pair_idx, "cosine_sim": round(cos,4),
                     "interfere": bool(cos < 0), "angle_deg": round(float(np.degrees(angle)),1)}})
n_conflict = sum(1 for r in results if r["interfere"])
mean_cos   = sum(r["cosine_sim"] for r in results) / len(results)
score = min(1.0, 0.3 + (0.5 - mean_cos) * 0.7) if mean_cos < 0.5 else 0.35
_result = {{"action": "gradient_interference", "n_pairs": len(results),
            "n_conflict": n_conflict, "mean_cosine": round(float(mean_cos),4),
            "results": results[:4], "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"grad_int_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: {res.get('n_conflict',0)}/{res.get('n_pairs',8)} pares "
                f"muestran interferencia de gradientes (cos={res.get('mean_cosine',0):.3f})",
                domain="continual", confidence=0.70,
                evidence="8 pares de tareas, d_in=20",
            )
        return score

    elif action == "ewc_retention":
        lam_base = 1.0 + (cycle % 10) * 2.0
        code = f"""
import numpy as np
np.random.seed({cycle * 13 + 7})
d_in = 30; n = 500
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def acc(W, X, y): return float(np.mean((sigmoid(X@W.T)[:,0]>0.5)==y))
def train(W, X, y, lr=0.05, steps=300, ewc_W=None, ewc_F=None, lam=0.0):
    W = W.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        g = np.zeros_like(W); g[0] = (e[:,None]*X).mean(0)
        if ewc_W is not None and lam > 0: g += lam * ewc_F * (W - ewc_W)
        W -= lr * g
    return W
def fisher(W, X, y):
    p = sigmoid(X@W.T)[:,0]; e = p-y
    g_per = e[:,None]*X
    F = np.zeros_like(W); F[0] = (g_per**2).mean(0)
    return F
W_init = np.random.randn(2, d_in) * 0.05
w_A = np.random.randn(d_in); w_A[d_in//2:] *= 0.1
w_B = w_A.copy(); w_B[:d_in//2] *= -1.0
X_shared = np.random.randn(n, d_in)
y_A = (X_shared @ w_A > 0).astype(float)
y_B = (X_shared @ w_B > 0).astype(float)
conflict_rate = float(np.mean(y_A != y_B))
W_A = train(W_init, X_shared, y_A, steps=400)
acc_A_base = acc(W_A, X_shared, y_A)
F_A = fisher(W_A, X_shared, y_A)
lambda_results = []
for lam in [0.0, {lam_base:.2f}, {lam_base*3:.2f}, {lam_base*10:.2f}]:
    W_ft = train(W_A, X_shared, y_B, ewc_W=W_A, ewc_F=F_A, lam=lam)
    a_A  = acc(W_ft, X_shared, y_A)
    a_B  = acc(W_ft, X_shared, y_B)
    lambda_results.append({{"lambda": round(lam,3), "acc_A": round(a_A,3),
                            "acc_B": round(a_B,3), "forgetting": round(float(acc_A_base-a_A),4)}})
best_lam = min(lambda_results[1:], key=lambda x: x["forgetting"])
improvement = lambda_results[0]["forgetting"] - best_lam["forgetting"]
score = min(1.0, 0.2 + max(0, improvement) * 2.5)
_result = {{"action": "ewc_retention", "acc_A_base": round(float(acc_A_base),3),
            "conflict_rate": round(float(conflict_rate),3),
            "lambda_sweep": lambda_results, "best_lambda": best_lam,
            "retention_improvement": round(float(improvement),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"ewc_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.5:
            best = res.get("best_lambda", {})
            tracker.record(
                f"Ciclo {cycle}: EWC λ={best.get('lambda')} reduce olvido "
                f"{res.get('retention_improvement',0):.3f} "
                f"(acc_A={best.get('acc_A')}, acc_B={best.get('acc_B')})",
                domain="continual", confidence=0.72,
                evidence=f"sweep λ en {lam_base:.2f}–{lam_base*10:.2f}",
            )
        return score

    elif action == "fisher_geometry":
        code = f"""
import numpy as np
np.random.seed({cycle * 11 + 5})
d_in = 20; n = 400
results = []
for scenario in range(5):
    W = np.random.randn(2, d_in) * 0.1
    X_A = np.random.randn(n, d_in)
    mask_A = np.zeros(d_in); mask_A[:d_in//2] = 1.0
    y_A = (X_A @ mask_A > 0).astype(float)
    X_B = np.random.randn(n, d_in)
    mask_B = np.zeros(d_in); mask_B[d_in//2:] = 1.0
    y_B = (X_B @ mask_B > 0).astype(float)
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def train_simple(X, y):
        W_ = W.copy()
        for _ in range(150):
            p = sigmoid(X@W_.T)[:,0]; e = p-y
            W_[0] -= 0.07*(e[:,None]*X).mean(0)
        return W_
    def fisher_diag(X, y, W_):
        p = sigmoid(X@W_.T)[:,0]; e = p-y
        return ((e[:,None]*X)**2).mean(0)
    W_A = train_simple(X_A, y_A)
    W_B = train_simple(X_B, y_B)
    F_A = fisher_diag(X_A, y_A, W_A)
    F_B = fisher_diag(X_B, y_B, W_B)
    critical_A = F_A > np.percentile(F_A, 70)
    safe_B     = F_B < np.percentile(F_B, 30)
    safe_to_protect = int((critical_A & safe_B).sum())
    overlap = float((critical_A & ~safe_B).sum()) / (d_in + 1e-9)
    results.append({{"scenario": scenario, "safe_to_protect": safe_to_protect,
                     "overlap": round(overlap, 3),
                     "mean_F_A": round(float(F_A.mean()),4),
                     "mean_F_B": round(float(F_B.mean()),4)}})
mean_safe = sum(r["safe_to_protect"] for r in results) / len(results)
mean_overlap = sum(r["overlap"] for r in results) / len(results)
score = min(1.0, 0.3 + (mean_safe/d_in)*0.5 + (1-mean_overlap)*0.2)
_result = {{"action": "fisher_geometry", "n_scenarios": len(results),
            "mean_safe_params": round(float(mean_safe),2),
            "mean_overlap": round(float(mean_overlap),4),
            "results": results, "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"fisher_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: FIM identifica {res.get('mean_safe_params',0):.1f} "
                f"parámetros seguros (críticos A, planos en B), overlap={res.get('mean_overlap',1):.3f}",
                domain="continual", confidence=0.68,
                evidence="d_in=20, 5 escenarios",
            )
        return score

    elif action == "task_similarity":
        code = f"""
import numpy as np
np.random.seed({cycle * 19 + 11})
d_in = 30; n = 600
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def train_sgd(X, y, W_init, lr=0.04, steps=400):
    W = W_init.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        W[0] -= lr*(e[:,None]*X).mean(0)
    return W
task_pairs = []
for pair_idx in range(8):
    sim = np.random.uniform(0.0, 1.0)
    w_A = np.random.randn(d_in); w_A /= np.linalg.norm(w_A)
    w_orth = np.random.randn(d_in)
    w_orth -= np.dot(w_orth, w_A)*w_A; w_orth /= np.linalg.norm(w_orth)+1e-9
    sign = 1.0 if sim >= 0.5 else -1.0
    w_B = sim * w_A + (1-sim) * sign * w_orth
    w_B /= np.linalg.norm(w_B)+1e-9
    X = np.random.randn(n, d_in)
    y_A = (X @ w_A > 0).astype(float)
    y_B = (X @ w_B > 0).astype(float)
    W_init = np.random.randn(1, d_in) * 0.05
    g_A = ((sigmoid(X@W_init.T)[:,0]-y_A)[:,None]*X).mean(0)
    g_B = ((sigmoid(X@W_init.T)[:,0]-y_B)[:,None]*X).mean(0)
    grad_cos = float(np.dot(g_A,g_B)/(np.linalg.norm(g_A)*np.linalg.norm(g_B)+1e-9))
    W_A  = train_sgd(X, y_A, W_init)
    acc_A_before = float(np.mean((sigmoid(X@W_A.T)[:,0]>0.5)==y_A))
    W_AB = train_sgd(X, y_B, W_A, steps=300)
    acc_A_after  = float(np.mean((sigmoid(X@W_AB.T)[:,0]>0.5)==y_A))
    forgetting = acc_A_before - acc_A_after
    task_pairs.append({{"sim": round(sim,3), "grad_cos": round(grad_cos,4),
                        "forgetting": round(forgetting,4),
                        "acc_A_before": round(acc_A_before,3),
                        "acc_A_after": round(acc_A_after,3)}})
sims = [r["sim"] for r in task_pairs]
fgts = [r["forgetting"] for r in task_pairs]
corr = float(np.corrcoef(sims, [-f for f in fgts])[0,1])
score = min(1.0, 0.3 + max(0, corr) * 0.7)
_result = {{"action": "task_similarity", "n_pairs": len(task_pairs),
            "sim_forgetting_corr": round(corr,4),
            "mean_forgetting": round(float(np.mean(fgts)),4),
            "pairs": task_pairs, "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"tasksim_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.5:
            tracker.record(
                f"Ciclo {cycle}: correlación similitud-olvido = {res.get('sim_forgetting_corr',0):.3f} "
                f"(mayor similitud → menor olvido)",
                domain="continual", confidence=0.65,
                evidence="8 pares de tareas con similitud controlada",
            )
        return score

    else:  # replay_vs_finetune
        replay_frac = 0.1 + (cycle % 8) * 0.05
        code = f"""
import numpy as np
np.random.seed({cycle * 23 + 13})
d_in = 30; n = 500; n_replay = int(n * {replay_frac:.2f})
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def acc(W, X, y): return float(np.mean((sigmoid(X@W.T)[:,0]>0.5)==y))
def train(X, y, W_init, lr=0.04, steps=350):
    W = W_init.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        W[0] -= lr*(e[:,None]*X).mean(0)
    return W
W_init = np.random.randn(2, d_in) * 0.05
w_A = np.random.randn(d_in); w_A[d_in//2:] *= 0.1
w_B = w_A.copy(); w_B[:d_in//2] *= -1.2
X_shared = np.random.randn(n, d_in)
y_A = (X_shared @ w_A > 0).astype(float)
y_B = (X_shared @ w_B > 0).astype(float)
W_A = train(X_shared, y_A, W_init, steps=500)
acc_A_base = acc(W_A, X_shared, y_A)
W_noreplay = train(X_shared, y_B, W_A)
replay_idx = np.random.choice(n, n_replay, replace=False)
X_mix = np.vstack([X_shared, X_shared[replay_idx]])
y_mix = np.concatenate([y_B, y_A[replay_idx]])
W_replay = train(X_mix, y_mix, W_A)
acc_A_norepay = acc(W_noreplay, X_shared, y_A)
acc_A_replay  = acc(W_replay,   X_shared, y_A)
acc_B_replay  = acc(W_replay,   X_shared, y_B)
retention_gain = acc_A_replay - acc_A_norepay
score = min(1.0, 0.2 + max(0, retention_gain) * 3.0)
_result = {{"action": "replay_vs_finetune", "replay_frac": round({replay_frac:.2f},3),
            "acc_A_base": round(float(acc_A_base),3),
            "acc_A_norepay": round(float(acc_A_norepay),3),
            "acc_A_replay": round(float(acc_A_replay),3),
            "acc_B_replay": round(float(acc_B_replay),3),
            "retention_gain": round(float(retention_gain),4),
            "replay_helps": bool(retention_gain > 0.02), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"replay_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.45:
            tracker.record(
                f"Ciclo {cycle}: replay {replay_frac:.0%} mejora retención de A "
                f"{res.get('retention_gain',0):.3f} sobre fine-tune puro",
                domain="continual", confidence=0.66,
                evidence=f"replay_frac={replay_frac:.2f}",
            )
        return score


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def execute_action(action: str, problem: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker) -> tuple[float, dict]:
    log = {"action": action, "cycle": cycle, "executor_results": []}
    if problem == "causal":
        score = _causal_action(action, cycle, executor, tracker, log)
    else:
        score = _continual_action(action, cycle, executor, tracker, log)
    tracker.record_attempt(description=action, result=f"score={score:.3f}", domain=problem)
    log["progress_score"] = score
    return score, log


# ══════════════════════════════════════════════════════════════════════════════
# Guardado
# ══════════════════════════════════════════════════════════════════════════════

class _Enc(json.JSONEncoder):
    def default(self, o):
        try:
            return float(o)
        except Exception:
            return str(o)

import json as _json

def _save_final(session_id, all_logs, tracker, meter, tool_reg,
                oracle, pool, memory, inventor, epi, evidence_bank,
                finding=None, verbose=True):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results_v011_{session_id}.json"
    try:
        data = {
            "version":       VERSION,
            "session_id":    session_id,
            "timestamp":     ts,
            "tracker":       tracker.self_report(),
            "meter":         meter.summary(),
            "tools":         tool_reg.summary(),
            "oracle":        oracle.summary(),
            "pool":          pool.summary(),
            "memory":        memory.summary(),
            "inventor":      inventor.summary(),
            "epistemic":     epi.summary(),
            "evidence_bank": evidence_bank.summary(),
            "finding":       finding,
            "logs":          all_logs[-50:],
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, cls=_Enc, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n  [Guardado] → {path}")
    except Exception as e:
        if verbose:
            print(f"\n  [Error guardado] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Síntesis oracle para CONSOLIDATE
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_consolidate(oracle, tracker, memory, evidence_bank,
                        problem: str, cycle: int) -> str:
    """
    Síntesis oracle al entrar en CONSOLIDATE.
    Usa request_oracle_direction() que ya maneja presupuesto, errores y logging.
    """
    if not oracle.available:
        return "(oracle no disponible)"
    supported_summary = "\n".join(
        f"  - {a}: score_mean={v['score_mean']:.3f}"
        for a, v in evidence_bank.supported.items()
    )
    # Construir recent_scores con lo que sabemos del evidence_bank
    recent_scores = {
        a: {"avg": v["score_mean"], "n": v["evidence_cycles"]}
        for a, v in evidence_bank.supported.items()
    }
    # Inyectar contexto en el campo failed_actions (lo usa para la síntesis)
    context_note = [
        f"CONSOLIDATE ciclo={cycle}",
        f"Soportadas: {supported_summary}",
        f"Corpus: {memory.to_corpus_knowledge(top_k=3)[:300]}",
    ]
    try:
        result = tracker.request_oracle_direction(
            problem, recent_scores, context_note
        )
        if not result:
            return "(oracle sin respuesta)"
        # result puede ser {"direction": "..."} o {"raw": "```json\n{...}"}
        if "direction" in result:
            return str(result["direction"])
        raw = result.get("raw", "")
        # intentar parsear JSON embebido en raw
        try:
            import re as _re
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if m:
                parsed = json.loads(m.group())
                return str(parsed.get("direction", parsed.get("synthesis", raw[:300])))
        except Exception:
            pass
        return str(raw)[:400] if raw else str(result)[:400]
    except Exception as e:
        return f"(error oracle: {e})"


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ACTION_TAGS = {
    "backdoor_adjustment":  ["causal", "backdoor", "confunder", "ajuste"],
    "do_calculus_test":     ["causal", "do-calculus", "dag", "identificacion"],
    "irm_vs_erm":           ["causal", "irm", "invarianza", "entornos"],
    "causal_discovery":     ["causal", "descubrimiento", "esqueleto", "dag"],
    "counterfactual_bounds":["causal", "contrafactual", "cotas", "tian-pearl"],
    "gradient_interference":["continual", "gradiente", "interferencia", "coseno"],
    "ewc_retention":        ["continual", "ewc", "fisher", "retencion"],
    "fisher_geometry":      ["continual", "fisher", "geometria", "parametros"],
    "task_similarity":      ["continual", "similitud", "tareas", "olvido"],
    "replay_vs_finetune":   ["continual", "replay", "memoria", "retencion"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",   choices=["causal","continual"], default="causal")
    parser.add_argument("--max-hours", type=float, default=8.0,
                        help="Safety net: máximo de horas (default 8)")
    parser.add_argument("--seed",      type=int,   default=42)
    parser.add_argument("--resume",    action="store_true")
    args = parser.parse_args()

    api_key  = os.getenv("ANTHROPIC_API_KEY")
    deadline = time.time() + args.max_hours * 3600

    session_id = datetime.now().strftime(f"v011_{args.problem}_%Y%m%d_%H%M")
    Path("results").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  agente vX {VERSION}  |  problema: {args.problem}  |  max {args.max_hours}h")
    print(f"  Para: finding_document ≥ {SOLUTION_MIN_SUPPORTED} hipótesis + richness ≥ {SOLUTION_MIN_RICHNESS}")
    print(f"{'='*70}\n")

    # ── Setup ────────────────────────────────────────────────────────────────
    executor     = DirectExecutor(timeout_sec=60.0)
    sub_executor = DirectExecutor(timeout_sec=50.0)

    oracle  = OracleClient(api_key, budget_usd=ORACLE_BUDGET)
    pool    = SubagentPool(sub_executor, max_workers=3)
    tracker = HypothesisTracker(subagent_pool=pool, oracle_client=oracle,
                                sandbox=executor)
    meter   = SatisfactionMeter(stagnation_window=5, pivot_threshold=10)

    library  = ResonantLibrary(f"library_{args.problem}_{args.seed}.pkl")
    maestro  = SecondOrderAgent(BASE_CONFIG, library, seed=args.seed)
    tool_reg = ToolRegistry()

    mem_path = f"results/memory_{args.problem}_{session_id}.json"
    memory   = ResearchMemory(domain=args.problem, persist_path=mem_path)
    inventor = ActionInventor(oracle)

    # ── Corpus ──────────────────────────────────────────────────────────────
    print(f"[Corpus] Cargando para '{args.problem}'...")
    all_tools: set[str] = set()
    for corpus_name in list_ml_corpus_for_domain(args.problem):
        corpus = get_ml_corpus(corpus_name)
        for td in corpus.get("initial_tools", []):
            if td["name"] not in all_tools:
                ok, _ = tool_reg.invent(td["name"], td["code"], td["description"])
                if ok:
                    all_tools.add(td["name"])
                    print(f"  [Tool] {td['name']}")
    print(f"  Tools: {list(all_tools)}\n")

    # ── Acciones + estado epistémico ────────────────────────────────────────
    actions      = PROBLEM_ACTIONS[args.problem]
    epi          = EpistemicState(actions)
    evidence_bank = EvidenceBank(actions)

    # ── Estado del loop ──────────────────────────────────────────────────────
    global_cycle = 0
    perspective  = "mathematician"
    all_logs: list = []
    last_oracle  = 0.0
    crisis_count = 0
    finding_document: dict | None = None

    _action_stagnation: dict[str, int] = {a: 0 for a in actions}
    _dormant:     set[str]  = set()
    _dormant_avg: dict[str, float] = {}
    _action_score_hist: dict[str, list[float]] = {a: [] for a in actions}

    # ── Resume ───────────────────────────────────────────────────────────────
    if args.resume:
        import glob as _glob
        candidates = sorted(
            _glob.glob(f"results_v01*_{args.problem}_*.json"),
            reverse=True,
        )
        if candidates:
            _resume_file = candidates[0]
            try:
                with open(_resume_file, encoding="utf-8") as _rf:
                    _prev = json.load(_rf)
                _prev_logs = _prev.get("logs", [])
                if _prev_logs:
                    global_cycle = max(l.get("cycle", 0) for l in _prev_logs)
                    for _l in _prev_logs:
                        _a = _l.get("action")
                        _s = _l.get("progress_score", 0.0)
                        if _a in _action_score_hist:
                            _action_score_hist[_a].append(_s)
                            if len(_action_score_hist[_a]) > 20:
                                _action_score_hist[_a].pop(0)
                _prev_meter = _prev.get("meter", {})
                print(f"[Resume] archivo={_resume_file}")
                print(f"[Resume] ciclo_anterior={global_cycle}  "
                      f"satisfaccion={_prev_meter.get('satisfaction','?')}")
            except Exception as _e:
                print(f"[Resume] Error: {_e} — inicio frío")
                global_cycle = 0
        else:
            print(f"[Resume] No se encontró resultado previo — inicio frío")

    print(f"[Inicio] fase={epi.phase}  actions={actions}\n")

    # ── Bucle principal ───────────────────────────────────────────────────────
    try:
        while time.time() < deadline:
            t_cycle_start = time.time()
            global_cycle += 1
            t_rem = (deadline - t_cycle_start) / 3600

            print(f"\n{'─'*70}")
            print(f"  CICLO {global_cycle}  |  {t_rem:.2f}h restantes  "
                  f"|  FASE: {epi.phase}  |  {perspective}")
            print(f"  Satisfacción: {meter.satisfaction_score():.2f}  "
                  f"mem: {memory.summary()['total']}  "
                  f"supported: {evidence_bank.n_supported()}/{len(actions)}  "
                  f"richness: {evidence_bank.richness():.3f}")
            print(f"{'─'*70}")

            # ── CONSOLIDATE: sin experimentos, solo síntesis ──────────────────
            if epi.phase == PHASE_CONSOLIDATE:
                print(f"\n  [CONSOLIDATE] Sintetizando hallazgos...")
                oracle_text = _oracle_consolidate(
                    oracle, tracker, memory, evidence_bank,
                    args.problem, global_cycle
                )
                last_oracle = time.time()
                print(f"  [Oracle/síntesis] {oracle_text[:150]}")

                richness = evidence_bank.richness()
                n_sup    = evidence_bank.n_supported()
                if n_sup >= SOLUTION_MIN_SUPPORTED and richness >= SOLUTION_MIN_RICHNESS:
                    result_type = "convergent"
                elif n_sup >= 1:
                    result_type = "inconclusive"
                else:
                    result_type = "inconclusive"

                finding_document = build_finding_document(
                    result_type, evidence_bank, memory, tracker,
                    meter, global_cycle, oracle_text
                )
                print(f"\n  {'★'*60}")
                print(f"  FINDING: {result_type.upper()}")
                print(f"  richness={richness:.3f}  supported={n_sup}  ciclos={global_cycle}")
                print(f"  {'★'*60}\n")
                break  # parar — encontramos un resultado

            # ── Elegir acción según fase ─────────────────────────────────────
            emphasis     = PERSPECTIVE_EMPHASIS.get(perspective, list(range(5)))
            active_indices = [i for i in emphasis if actions[i] not in _dormant]
            if not active_indices:
                _dormant.clear(); _dormant_avg.clear()
                active_indices = list(emphasis)
                print("  [Limbo] Todas dormantes — reviviendo todas")
            active_actions = [actions[i] for i in active_indices]

            if epi.phase == PHASE_SURVEY:
                # Round-robin: mapear cobertura uniformemente
                action_idx = active_indices[global_cycle % len(active_indices)]
                action     = actions[action_idx]
            else:
                # FOCUS: curiosidad dirigida
                action = epi.most_curious_action(active_actions)

            # ── Ejecutar acción ──────────────────────────────────────────────
            score, attack_log = execute_action(
                action, args.problem, global_cycle, executor, tracker
            )
            all_logs.append(attack_log)
            for res in attack_log["executor_results"]:
                print(f"  [{action}] {str(res)[:140]}")

            # ── Indexar en ResearchMemory ────────────────────────────────────
            tags = ACTION_TAGS.get(action, [action, args.problem])
            hist = _action_score_hist.setdefault(action, [])
            _is_novel = (
                len(hist) < 3
                or abs(score - sum(hist[-5:]) / len(hist[-5:])) > MEM_NOVELTY_THRESHOLD
            )
            if _is_novel:
                mem_id = memory.index(
                    summary = f"{action}: score={score:.3f}, "
                              f"{str(attack_log['executor_results'][:1])[:120]}",
                    score   = score, cycle=global_cycle,
                    source  = "main", tags=tags,
                    result  = attack_log["executor_results"][0]
                               if attack_log["executor_results"] else {},
                )
            else:
                mem_id = "(skip-redundante)"
            hist.append(score)
            if len(hist) > 20:
                hist.pop(0)

            # ── Actualizar EpistemicState ────────────────────────────────────
            epi.update(action, score, hist, _is_novel)

            # ── Acumular evidencia ───────────────────────────────────────────
            evidence_bank.record(action, score)

            # ── Stagnation por acción + Limbo ────────────────────────────────
            if len(hist) >= 3 and all(
                abs(hist[-1-i] - hist[-1]) < 0.02
                for i in range(1, min(3, len(hist)))
            ):
                _action_stagnation[action] = _action_stagnation.get(action, 0) + 1
            else:
                _action_stagnation[action] = 0

            if _action_stagnation.get(action, 0) >= 15 and action not in _dormant:
                _dormant.add(action)
                _dormant_avg[action] = meter.satisfaction_score()
                print(f"  [Limbo] '{action}' dormante "
                      f"(stagnation={_action_stagnation[action]}, score≈{score:.3f})")

            for _da in list(_dormant):
                if _da == action:
                    continue
                if epi.saturated.get(_da, False):
                    continue  # saturada (≈1.0) → no revival, ya está resuelta
                if meter.satisfaction_score() > _dormant_avg.get(_da, 0.0) + 0.05:
                    _dormant.discard(_da)
                    _action_stagnation[_da] = 0
                    print(f"  [Limbo/revival] '{_da}' reactivada")

            # ── Medidor de satisfacción ──────────────────────────────────────
            state = meter.update(score)
            print(f"  [Meter] score={score:.3f}  state={state.value}  "
                  f"mem={mem_id}  curious={epi.uncertainty.get(action,0):.4f}")

            # ── Subagentes (recoger) ─────────────────────────────────────────
            completed = pool.collect_completed()
            if completed:
                tracker.absorb_subagent_results(completed)
                for r in completed:
                    memory.index(
                        summary=f"Sub {r.task_id}: {r.hypothesis[:80]}",
                        score=r.score, cycle=global_cycle,
                        source="subagent", tags=["verification", r.domain],
                        result=r.result,
                    )

            # ── Verificación de hipótesis cada 3 ciclos ──────────────────────
            if global_cycle % 3 == 0:
                for h in list(tracker._hyps):
                    tracker._maybe_verify(h)

            # ── Creative crisis cada 8 ciclos ────────────────────────────────
            if global_cycle % 8 == 0:
                crisis = tracker.check_creative_crisis(research_memory=memory)
                if crisis:
                    crisis_count += 1
                    print(f"\n  [CreativeCrisis #{crisis_count}]")
                    print(f"    H tensión: {crisis['h1'][:60]}")
                    print(f"    Síntesis:  {crisis['synthesis_direction'][:80]}")

            # ── Rotación de perspectiva cada 8 ciclos ────────────────────────
            if global_cycle % 8 == 0:
                opts = [p for p in PERSPECTIVE_EMPHASIS if p != perspective]
                perspective = opts[meter._pivot_count % len(opts)]
                meter._pivot_count += 1
                print(f"\n  [Rotación] → {perspective}")
            elif meter.needs_pivot():
                perspective = meter.next_perspective(
                    perspective, list(PERSPECTIVE_EMPHASIS.keys())
                )
                print(f"\n  [Pivot] → {perspective}")

            # ── Oracle: contradicción detectada ──────────────────────────────
            now = time.time()
            if (oracle.available
                    and evidence_bank.has_contradiction()
                    and (now - last_oracle) > 300):
                pair = evidence_bank.pop_contradiction()
                if pair:
                    try:
                        contra_scores = {
                            pair[0]: {"avg": round(sum(evidence_bank._scores.get(pair[0],[0])[-3:]) / 3, 3), "n": 3},
                            pair[1]: {"avg": round(sum(evidence_bank._scores.get(pair[1],[0])[-3:]) / 3, 3), "n": 3},
                        }
                        result = tracker.request_oracle_direction(
                            args.problem, contra_scores,
                            [f"CONTRADICCIÓN: '{pair[0]}' alto vs '{pair[1]}' bajo"]
                        )
                        if result and "direction" in result:
                            oracle_text = str(result["direction"])
                        elif result and "raw" in result:
                            try:
                                import re as _re2
                                m2 = _re2.search(r'\{.*\}', result["raw"], _re2.DOTALL)
                                oracle_text = json.loads(m2.group()).get("direction","") if m2 else result["raw"][:200]
                            except Exception:
                                oracle_text = str(result.get("raw",""))[:200]
                        else:
                            oracle_text = ""
                        last_oracle = now
                        print(f"\n  [Oracle/contradicción] {oracle_text[:120]}")
                        if oracle_text:
                            memory.index(
                                summary=f"Contradicción {pair[0]} vs {pair[1]}: {oracle_text[:100]}",
                                score=0.6, cycle=global_cycle, source="oracle",
                                tags=["contradiction", "oracle", args.problem],
                                result={},
                            )
                    except Exception as _e:
                        print(f"  [Oracle/error] {_e}")

            # ── ActionInventor ────────────────────────────────────────────────
            if (global_cycle % 15 == 0
                    and inventor.can_invent
                    and sum(_rsc := [l["progress_score"] for l in all_logs[-8:]])
                       / max(1, len(_rsc)) < 0.50):
                avg_r = sum(_rsc) / len(_rsc)
                print(f"\n  [ActionInventor] avg_reciente={avg_r:.2f} < 0.50, inventando...")
                tool_sum = "\n".join(f"  {n}" for n in all_tools) or "(ninguna)"
                invention = inventor.try_invent(
                    problem=args.problem, n_cycles=global_cycle,
                    tool_summary=tool_sum, failed_actions=[],
                    memory=memory,
                )
                if invention:
                    ok, _ = tool_reg.invent(
                        invention["tool_name"], invention["code"],
                        invention["description"]
                    )
                    if ok:
                        all_tools.add(invention["tool_name"])
                        memory.index(
                            summary=f"Invención: {invention['tool_name']} — "
                                    f"{invention['description']}",
                            score=0.7, cycle=global_cycle, source="invention",
                            tags=["invention", "new_tool", args.problem],
                        )
                        print(f"  [ActionInventor] ✓ {invention['tool_name']}")

            # ── Transición de fase ────────────────────────────────────────────
            new_phase = epi.should_transition(evidence_bank.n_supported())
            if new_phase:
                epi.enter_phase(new_phase)
                if new_phase == PHASE_CONSOLIDATE:
                    # La siguiente iteración ejecuta la síntesis
                    print(f"  [Fase] Entrando a CONSOLIDATE en ciclo {global_cycle+1}")

            # ── Memory snapshot cada 10 ciclos ────────────────────────────────
            if global_cycle % 10 == 0:
                ms = memory.summary()
                print(f"\n  [Memory] total={ms['total']} avg={ms['avg_score']:.3f}")
                top3 = memory.to_corpus_knowledge(top_k=3)
                print(f"  [Memory/top3]:\n{top3}")
                print(f"  [Epistemic] {epi.summary()}")

            # ── Guardado periódico cada 25 ciclos ────────────────────────────
            if global_cycle % 25 == 0:
                _save_final(session_id, all_logs, tracker, meter, tool_reg,
                            oracle, pool, memory, inventor, epi, evidence_bank,
                            finding=None, verbose=False)

            # ── Pausa mínima ──────────────────────────────────────────────────
            elapsed_cycle = time.time() - t_cycle_start
            if elapsed_cycle < MIN_CYCLE_SEC:
                time.sleep(MIN_CYCLE_SEC - elapsed_cycle)

    except KeyboardInterrupt:
        print("\n  [Interrupción manual]")
    finally:
        pool.shutdown()
        # Si no se llegó a finding, construir uno inconclusivo
        if finding_document is None:
            finding_document = build_finding_document(
                "inconclusive", evidence_bank, memory, tracker,
                meter, global_cycle,
                oracle_synthesis="(safety net — sin síntesis oracle)"
            )
        _save_final(session_id, all_logs, tracker, meter, tool_reg,
                    oracle, pool, memory, inventor, epi, evidence_bank,
                    finding=finding_document)
        print(f"\n[Fin] ciclos={global_cycle}  "
              f"resultado={finding_document['result']}  "
              f"richness={finding_document['finding_richness']:.3f}")
