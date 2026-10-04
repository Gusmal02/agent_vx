"""
run_v012.py — Agente vX v0.1.2

Cambios respecto a v0.1.1:
  1. CrossActionAnalyzer: cada acción exporta key_metric; correlaciones cruzadas
     (|r| > 0.70, N ≥ 5) se proponen como hipótesis inter-acción.
  2. Acciones explore-mode (una por dominio):
       causal:   discovery_threshold_sweep  (busca τ* óptimo para PC skeleton)
       continual: interference_angle_sweep  (encuentra ángulo de cruce-cero de coseno)
     Estas reemplazan causal_discovery y gradient_interference.
  3. Fase VERIFY: tras CONSOLIDATE, el oracle propone 2 experimentos ejecutables;
     el agente los corre, evalúa confirmación/refutación, y lo incluye en el finding.

Uso:
    uv run python run_v012.py --problem causal
    uv run python run_v012.py --problem continual --max-hours 8
    uv run python run_v012.py --problem causal --resume
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

VERSION = "v0.1.2"

# ── Configuración ─────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "N": 100, "M": 3, "K": 3,
    "stagnation_threshold": 8,
    "filter_threshold":     0.35,
    "confidence_threshold": 0.88,
    "epsilon_start":        0.50,
    "n_episodios":          15,
}

# discovery_threshold_sweep reemplaza causal_discovery
# interference_angle_sweep  reemplaza gradient_interference
PROBLEM_ACTIONS = {
    "causal": [
        "backdoor_adjustment",
        "do_calculus_test",
        "irm_vs_erm",
        "discovery_threshold_sweep",
        "counterfactual_bounds",
    ],
    "continual": [
        "interference_angle_sweep",
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

ORACLE_BUDGET = 3.50  # +$1 para experimentos VERIFY

# ── Campo key_metric por acción ───────────────────────────────────────────────
KEY_METRIC_FIELDS = {
    "backdoor_adjustment":      "mean_bias_reduction",
    "do_calculus_test":         "frac_adj_better",
    "irm_vs_erm":               "frac_irm_wins",
    "discovery_threshold_sweep":"optimal_threshold",
    "counterfactual_bounds":    "frac_bounds_valid",
    "interference_angle_sweep": "zero_crossing_angle",
    "ewc_retention":            "retention_improvement",
    "fisher_geometry":          "mean_safe_params",
    "task_similarity":          "sim_forgetting_corr",
    "replay_vs_finetune":       "retention_gain",
}

# ── Fases ─────────────────────────────────────────────────────────────────────

PHASE_SURVEY      = "SURVEY"
PHASE_FOCUS       = "FOCUS"
PHASE_CONSOLIDATE = "CONSOLIDATE"
PHASE_VERIFY      = "VERIFY"

# ── Thresholds ────────────────────────────────────────────────────────────────

SURVEY_MIN_CYCLES_PER_ACTION = 3
FOCUS_MAX_CYCLES             = 40
SUPPORTED_THRESHOLD_SCORE    = 0.65
SUPPORTED_EVIDENCE_WINDOW    = 5
SUPPORTED_EVIDENCE_VARIANCE  = 0.04
ANOMALY_SATURATION           = 0.92
ANOMALY_FLAT_VAR             = 0.001
SOLUTION_MIN_SUPPORTED       = 2
SOLUTION_MIN_RICHNESS        = 0.55
FOCUS_MIN_CYCLES_BEFORE_STOP = 12   # mínimo de ciclos FOCUS antes de CONSOLIDATE
MIN_CYCLE_SEC                = 15.0
MEM_NOVELTY_THRESHOLD        = 0.12

CROSS_ANALYSIS_INTERVAL      = 3    # analizar cada N ciclos FOCUS (antes: 5)
CROSS_CORR_THRESHOLD         = 0.70
CROSS_MIN_POINTS             = 4    # puntos mínimos por acción (antes: 5)
ORACLE_RESERVE_BUDGET        = 1.50 # USD reservados para CONSOLIDATE + VERIFY


# ══════════════════════════════════════════════════════════════════════════════
# EpistemicState  (idéntico a v0.1.1)
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicState:
    """Modelo del estado del conocimiento del agente por acción."""

    def __init__(self, actions: list[str]):
        self.actions     = actions
        self.phase       = PHASE_SURVEY
        self.coverage    = {a: 0.0 for a in actions}
        self.uncertainty = {a: 1.0 for a in actions}
        self.saturated   = {a: False for a in actions}
        self.flat        = {a: False for a in actions}
        self.focus_action: str | None = None
        self._cycles_in_phase  = 0
        self._survey_count     = {a: 0 for a in actions}
        self._focus_cycles     = 0
        self._last_action: str | None = None
        self._consecutive: int = 0

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

        if action == self._last_action:
            self._consecutive += 1
        else:
            self._consecutive = 1
            self._last_action = action

    @property
    def anomaly(self) -> dict[str, bool]:
        return {a: self.saturated[a] or self.flat[a] for a in self.actions}

    def most_curious_action(self, active: list[str]) -> str:
        if self._consecutive >= 3 and len(active) > 1:
            candidates = [a for a in active if a != self._last_action]
        else:
            candidates = active

        best, best_score = candidates[0], -999.0
        for a in candidates:
            s  = self.uncertainty.get(a, 1.0) * 2.0
            if self.flat.get(a, False):
                s -= 0.6
            elif self.saturated.get(a, False):
                s -= 0.1
            s -= self.coverage.get(a, 0.0) * 0.3
            if s > best_score:
                best, best_score = a, s
        return best

    def all_surveyed(self) -> bool:
        return all(
            self._survey_count.get(a, 0) >= SURVEY_MIN_CYCLES_PER_ACTION
            for a in self.actions
        )

    def should_transition(self, n_supported: int,
                          n_new: int = 0, has_prior: bool = False) -> str | None:
        if self.phase == PHASE_SURVEY:
            if self.all_surveyed():
                if any(self.anomaly.values()) or n_supported >= 1:
                    return PHASE_FOCUS
        elif self.phase == PHASE_FOCUS:
            enough_focus = self._focus_cycles >= FOCUS_MIN_CYCLES_BEFORE_STOP
            if enough_focus:
                if has_prior:
                    # Con estado previo: exigir al menos 1 hallazgo nuevo
                    # o haber agotado el máximo de ciclos
                    if n_new >= 1 and n_supported >= SOLUTION_MIN_SUPPORTED:
                        return PHASE_CONSOLIDATE
                else:
                    if n_supported >= SOLUTION_MIN_SUPPORTED:
                        return PHASE_CONSOLIDATE
            if self._focus_cycles >= FOCUS_MAX_CYCLES:
                return PHASE_CONSOLIDATE
        # CONSOLIDATE → VERIFY se maneja en el bucle principal
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
# EvidenceBank  (idéntico a v0.1.1)
# ══════════════════════════════════════════════════════════════════════════════

class EvidenceBank:
    def __init__(self, actions: list[str]):
        self.actions    = actions
        self._scores: dict[str, list[float]] = {a: [] for a in actions}
        self.supported:  dict[str, dict] = {}
        self.rejected:   dict[str, dict] = {}
        self._contradiction: tuple[str, str] | None = None
        self._new_this_session: set[str] = set()  # hipótesis encontradas en esta sesión

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
                self._new_this_session.add(action)
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

    def n_new_this_session(self) -> int:
        return len(self._new_this_session)

    def richness(self) -> float:
        if not self.supported:
            return 0.0
        n_frac = min(1.0, self.n_supported() / SOLUTION_MIN_SUPPORTED)
        cross  = min(1.0, (self.n_supported() - 1) / 2) if self.n_supported() > 1 else 0.0
        mean_var = (sum(v["score_var"] for v in self.supported.values())
                    / len(self.supported))
        consistency = max(0.0, 1.0 - mean_var * 20)
        return round(0.4 * n_frac + 0.3 * cross + 0.3 * consistency, 4)

    def summary(self) -> dict:
        return {
            "supported": self.supported,
            "rejected":  self.rejected,
            "richness":  self.richness(),
            "n_supported": self.n_supported(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# CrossActionAnalyzer  ← NUEVO v0.1.2
# ══════════════════════════════════════════════════════════════════════════════

class CrossActionAnalyzer:
    """
    Acumula key_metric por acción y detecta correlaciones cruzadas.
    Una correlación |r| > 0.70 con N ≥ 5 puntos se registra como
    hipótesis inter-acción.
    """

    def __init__(self, actions: list[str]):
        self.key_metrics: dict[str, list[float]] = {a: [] for a in actions}
        self.cross_hypotheses: list[dict] = []
        self._reported: set[tuple[str, str]] = set()

    def record(self, action: str, key_metric_value) -> None:
        if key_metric_value is not None:
            try:
                self.key_metrics[action].append(float(key_metric_value))
            except (TypeError, ValueError):
                pass

    def analyze(self) -> list[dict]:
        """Retorna hipótesis cruzadas nuevas (no reportadas antes)."""
        new_hyps = []
        actions_with_data = [
            a for a, v in self.key_metrics.items()
            if len(v) >= CROSS_MIN_POINTS
        ]
        for i, a1 in enumerate(actions_with_data):
            for a2 in actions_with_data[i + 1:]:
                pair = (a1, a2)
                if pair in self._reported:
                    continue
                v1 = self.key_metrics[a1]
                v2 = self.key_metrics[a2]
                n  = min(len(v1), len(v2))
                if n < CROSS_MIN_POINTS:
                    continue
                arr1 = np.array(v1[-n:])
                arr2 = np.array(v2[-n:])
                if np.std(arr1) < 1e-6 or np.std(arr2) < 1e-6:
                    continue  # constante → correlación no informativa
                corr = float(np.corrcoef(arr1, arr2)[0, 1])
                if abs(corr) >= CROSS_CORR_THRESHOLD:
                    f1 = KEY_METRIC_FIELDS.get(a1, a1)
                    f2 = KEY_METRIC_FIELDS.get(a2, a2)
                    hyp = {
                        "action1":     a1,
                        "action2":     a2,
                        "metric1":     f1,
                        "metric2":     f2,
                        "correlation": round(corr, 3),
                        "n":           n,
                        "direction":   "positiva" if corr > 0 else "negativa",
                        "hypothesis":  (
                            f"{'↑' if corr > 0 else '↓'}{f1} correlaciona con "
                            f"{f2} (r={corr:.2f}, n={n})"
                        ),
                    }
                    new_hyps.append(hyp)
                    self._reported.add(pair)
                    self.cross_hypotheses.append(hyp)
                    print(f"  [CrossAnalyzer] CORRELACIÓN: {hyp['hypothesis']}")
        return new_hyps

    def summary(self) -> dict:
        return {
            "cross_hypotheses":    self.cross_hypotheses,
            "n_cross":             len(self.cross_hypotheses),
            "key_metric_counts":   {a: len(v) for a, v in self.key_metrics.items()},
        }


# ══════════════════════════════════════════════════════════════════════════════
# EpistemicStateManager  ← NUEVO v0.1.3  (persistencia entre sesiones)
# ══════════════════════════════════════════════════════════════════════════════

class EpistemicStateManager:
    """
    Persiste y restaura el estado epistémico del agente entre sesiones.

    Almacena en results/epistemic_state_{problem}.json:
      - supported_hypotheses: hipótesis ya confirmadas (evita re-descubrir)
      - cross_hypotheses: correlaciones cruzadas conocidas
      - key_metric_history: historial de key_metrics para CrossAnalyzer
      - action_survey_counts: cuántas veces fue explorada cada acción

    Al restaurar, el agente arranca con conocimiento previo:
      - Acciones ya soportadas no necesitan re-encolar
      - CrossAnalyzer tiene datos históricos → puede detectar cambios
      - Si N_supported >= SOLUTION_MIN_SUPPORTED al arrancar → salta a FOCUS
    """

    MAX_KEY_METRIC_HISTORY = 30  # puntos máximos a guardar por acción

    def __init__(self, problem: str, state_dir: str = "results",
                 agent_id: str | None = None):
        Path(state_dir).mkdir(exist_ok=True)
        suffix = f"_{agent_id}" if agent_id else ""
        self.path     = Path(state_dir) / f"epistemic_state_{problem}{suffix}.json"
        self.problem  = problem
        self.agent_id = agent_id

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            with open(self.path, encoding="utf-8") as f:
                state = json.load(f)
            print(f"\n[Estado previo] cargado: {self.path}")
            print(f"  sesiones={state.get('sessions', 0)}  "
                  f"supported={len(state.get('supported_hypotheses', {}))}  "
                  f"cross={len(state.get('cross_hypotheses', []))}  "
                  f"última_actualización={state.get('last_updated','?')[:10]}")
            return state
        except Exception as e:
            print(f"[Estado previo] error al cargar ({e}) — inicio frío")
            return None

    def apply(self, prior: dict,
              evidence_bank: "EvidenceBank",
              cross_analyzer: "CrossActionAnalyzer",
              epi: "EpistemicState") -> int:
        """
        Inyecta estado previo en los objetos activos.
        Retorna número de hipótesis soportadas restauradas.
        """
        n_loaded = 0

        # 1. Hipótesis soportadas → poblar EvidenceBank.supported directamente
        for action, hyp_data in prior.get("supported_hypotheses", {}).items():
            if action not in evidence_bank.actions:
                continue
            if action not in evidence_bank.supported:
                evidence_bank.supported[action] = hyp_data
                n_loaded += 1
                print(f"  [Estado previo] SUPPORTED restaurado: '{action}' "
                      f"mean={hyp_data.get('score_mean', '?')}")

        # 2. Historia de key_metrics → CrossActionAnalyzer
        for action, history in prior.get("key_metric_history", {}).items():
            if action in cross_analyzer.key_metrics:
                cross_analyzer.key_metrics[action] = list(history)

        # 3. Cross_hypotheses conocidas → evitar re-anunciarlas
        existing_pairs = {(h["action1"], h["action2"])
                          for h in cross_analyzer.cross_hypotheses}
        for ch in prior.get("cross_hypotheses", []):
            pair = (ch["action1"], ch["action2"])
            if pair not in existing_pairs:
                cross_analyzer.cross_hypotheses.append(ch)
                cross_analyzer._reported.add(pair)  # no re-anunciar
                existing_pairs.add(pair)

        # 4. Survey counts → EpistemicState (evita re-SURVEY de lo ya explorado)
        for action, count in prior.get("action_survey_counts", {}).items():
            if action in epi._survey_count:
                epi._survey_count[action] = max(epi._survey_count[action], count)
                # Restaurar cobertura aproximada para acciones muy exploradas
                if count >= SURVEY_MIN_CYCLES_PER_ACTION:
                    epi.coverage[action] = min(0.9, epi.coverage[action] + 0.3 * count)

        return n_loaded

    def save(self, evidence_bank: "EvidenceBank",
             cross_analyzer: "CrossActionAnalyzer",
             epi: "EpistemicState",
             prior: dict | None = None) -> dict:
        """Merge estado actual con previo y guarda."""
        base = prior or {}

        # Hipótesis soportadas: merge (actualizamos si ya existe, añadimos si es nueva)
        all_supported = dict(base.get("supported_hypotheses", {}))
        all_supported.update(evidence_bank.supported)

        # Cross_hypotheses: merge por par de acciones (actualizar r si ya existe)
        all_cross: list[dict] = []
        seen_pairs: dict[tuple, int] = {}
        for ch in base.get("cross_hypotheses", []):
            pair = (ch["action1"], ch["action2"])
            seen_pairs[pair] = len(all_cross)
            all_cross.append(ch)
        for ch in cross_analyzer.cross_hypotheses:
            pair = (ch["action1"], ch["action2"])
            if pair in seen_pairs:
                all_cross[seen_pairs[pair]] = ch  # actualizar con r más reciente
            else:
                seen_pairs[pair] = len(all_cross)
                all_cross.append(ch)

        # Key_metric history: extender y truncar
        all_km: dict[str, list] = dict(base.get("key_metric_history", {}))
        for action, new_pts in cross_analyzer.key_metrics.items():
            prev = all_km.get(action, [])
            combined = prev + new_pts
            all_km[action] = combined[-self.MAX_KEY_METRIC_HISTORY:]

        # Survey counts: máximo
        all_survey: dict[str, int] = dict(base.get("action_survey_counts", {}))
        for action, count in epi._survey_count.items():
            all_survey[action] = max(all_survey.get(action, 0), count)

        # Anotar agent_id en cada hipótesis soportada para trazabilidad en merge
        if self.agent_id:
            for hyp in all_supported.values():
                hyp.setdefault("agent_id", self.agent_id)

        state = {
            "problem":               self.problem,
            "version":               VERSION,
            "agent_id":              self.agent_id,
            "last_updated":          datetime.now().isoformat(),
            "sessions":              base.get("sessions", 0) + 1,
            "supported_hypotheses":  all_supported,
            "cross_hypotheses":      all_cross,
            "key_metric_history":    all_km,
            "action_survey_counts":  all_survey,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"\n[Estado] guardado → {self.path}  "
              f"(sesión {state['sessions']}  "
              f"supported={len(all_supported)}  cross={len(all_cross)})")
        return state


# ══════════════════════════════════════════════════════════════════════════════
# FindingDocument  (ampliado para verify_results y cross_hypotheses)
# ══════════════════════════════════════════════════════════════════════════════

def build_finding_document(
    result_type: str,
    evidence_bank: EvidenceBank,
    memory,
    tracker,
    meter,
    cycles: int,
    oracle_synthesis: str = "",
    verify_results: list | None = None,
    cross_hypotheses: list | None = None,
) -> dict:
    top_memory = memory.to_corpus_knowledge(top_k=5)
    report = tracker.self_report()
    tracked_hyps = report.get("hypotheses", [])
    if not isinstance(tracked_hyps, list):
        open_questions = []
    else:
        open_questions = [
            h for h in tracked_hyps
            if h not in evidence_bank.supported
        ][:5]

    return {
        "version":               VERSION,
        "result":                result_type,
        "cycles_to_finding":     cycles,
        "finding_richness":      evidence_bank.richness(),
        "supported_hypotheses":  evidence_bank.supported,
        "rejected_hypotheses":   evidence_bank.rejected,
        "cross_hypotheses":      cross_hypotheses or [],
        "open_questions":        open_questions,
        "meter_summary":         meter.summary(),
        "memory_top5":           top_memory,
        "oracle_synthesis":      oracle_synthesis,
        "verify_results":        verify_results or [],
        "timestamp":             datetime.now().isoformat(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Causalidad
# ══════════════════════════════════════════════════════════════════════════════

def _causal_action(action: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker,
                   log: dict) -> tuple[float, float | None]:
    """
    Retorna (score, key_metric).
    key_metric es el campo más informativo del resultado para CrossActionAnalyzer.
    """

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
        return score, res.get("mean_bias_reduction")

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
        return score, res.get("frac_adj_better")

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
        return score, res.get("frac_irm_wins")

    elif action == "discovery_threshold_sweep":
        # ── Acción explore-mode (v0.1.2) ────────────────────────────────────
        # Pregunta real: ¿cuál es el umbral τ* óptimo para PC skeleton recovery?
        # Varía con densidad del grafo, N, ruido → genuinamente desconocido por ciclo.
        code = f"""
import numpy as np
np.random.seed({cycle * 19 + 11})
n_graphs  = 6
thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
shd_by_tau = {{t: [] for t in thresholds}}
graph_stats = []
for gi in range(n_graphs):
    n = 600; d = 5
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
    true_skeleton = ((true_adj + true_adj.T) > 0).astype(float)
    np.fill_diagonal(true_skeleton, 0)
    n_true_edges = int(true_skeleton.sum() // 2)
    graph_stats.append(n_true_edges)
    for tau in thresholds:
        est = np.zeros((d, d))
        for i in range(d):
            for j in range(i+1, d):
                others = [k for k in range(d) if k != i and k != j]
                if others:
                    Z = X[:, others]
                    Xi_r = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
                    Xj_r = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
                    corr = np.corrcoef(Xi_r, Xj_r)[0, 1]
                else:
                    corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
                if abs(corr) > tau:
                    est[i, j] = est[j, i] = 1
        np.fill_diagonal(est, 0)
        shd = int(np.sum(np.abs(true_skeleton - est)))
        shd_by_tau[tau].append(shd)
mean_shd = {{t: float(np.mean(v)) for t, v in shd_by_tau.items()}}
optimal_tau = min(mean_shd, key=mean_shd.get)
worst_shd   = max(mean_shd.values())
best_shd    = mean_shd[optimal_tau]
sharpness   = (worst_shd - best_shd) / (worst_shd + 1e-9)  # qué tan claro es el óptimo
score = min(1.0, 0.2 + sharpness * 0.5 + max(0, 1 - best_shd/8) * 0.3)
_result = {{
    "action": "discovery_threshold_sweep",
    "n_graphs": n_graphs,
    "mean_shd_by_tau": {{str(k): round(v,3) for k,v in mean_shd.items()}},
    "optimal_threshold": round(float(optimal_tau), 2),
    "best_shd": round(float(best_shd), 3),
    "sharpness": round(float(sharpness), 4),
    "mean_graph_edges": round(float(np.mean(graph_stats)), 2),
    "score": round(score, 4),
}}
"""
        r   = executor.run(code, label=f"threshold_sweep_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.40:
            tracker.record(
                f"Ciclo {cycle}: τ* óptimo para PC skeleton = "
                f"{res.get('optimal_threshold',0.15):.2f} "
                f"(SHD={res.get('best_shd',0):.2f}, sharpness={res.get('sharpness',0):.3f})",
                domain="causal", confidence=0.65,
                evidence=f"sweep τ∈[0.05,0.40], n_graphs={res.get('n_graphs')}, d=5",
            )
        return score, res.get("optimal_threshold")

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
        return score, res.get("frac_bounds_valid")


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Olvido Catastrófico
# ══════════════════════════════════════════════════════════════════════════════

def _continual_action(action: str, cycle: int,
                      executor: DirectExecutor,
                      tracker: HypothesisTracker,
                      log: dict) -> tuple[float, float | None]:
    """Retorna (score, key_metric)."""

    if action == "interference_angle_sweep":
        # ── Acción explore-mode (v0.1.2) ────────────────────────────────────
        # Pregunta real: ¿a qué ángulo entre tareas el coseno de gradientes
        # cruza de positivo a negativo (interferencia constructiva → destructiva)?
        code = f"""
import numpy as np
np.random.seed({cycle * 17 + 3})
d_in = 20; n = 400
angles_deg = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
sweep_results = []
for angle_deg in angles_deg:
    angle = np.radians(angle_deg)
    W_init = np.random.randn(1, d_in) * 0.1
    X_A = np.random.randn(n, d_in); X_A[:, 1::2] *= 0.05
    y_A = (X_A[:, ::2].sum(axis=1) > 0).astype(float)
    # tarea B orientada a angle_deg respecto a tarea A
    base = np.zeros(d_in); base[::2] = 1.0; base /= np.linalg.norm(base)+1e-9
    orth = np.zeros(d_in); orth[1::2] = 1.0; orth /= np.linalg.norm(orth)+1e-9
    w_B = np.cos(angle)*base + np.sin(angle)*orth
    X_B = np.random.randn(n, d_in)
    y_B = (X_B @ w_B > 0).astype(float)
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def grad_fn(X, y, W):
        p = sigmoid(X @ W.T)[:,0]; e = p - y
        return (e[:,None]*X).mean(axis=0)
    gA = grad_fn(X_A, y_A, W_init)
    gB = grad_fn(X_B, y_B, W_init)
    cos_sim = float(np.dot(gA, gB) / (np.linalg.norm(gA)*np.linalg.norm(gB)+1e-9))
    # medir olvido real
    def train(X, y, W, lr=0.05, steps=200):
        W = W.copy()
        for _ in range(steps):
            p = sigmoid(X@W.T)[:,0]; e = p-y
            W[0] -= lr*(e[:,None]*X).mean(0)
        return W
    W_A = train(X_A, y_A, W_init, steps=300)
    acc_A_pre = float(np.mean((sigmoid(X_A@W_A.T)[:,0]>0.5)==y_A))
    W_AB = train(X_B, y_B, W_A)
    acc_A_post = float(np.mean((sigmoid(X_A@W_AB.T)[:,0]>0.5)==y_A))
    forgetting = acc_A_pre - acc_A_post
    sweep_results.append({{
        "angle_deg": angle_deg,
        "cosine_sim": round(cos_sim, 4),
        "forgetting": round(float(forgetting), 4),
        "interferes": bool(cos_sim < 0),
    }})
# Encontrar ángulo de cruce-cero: primer ángulo donde coseno < 0
zero_crossing = None
for r in sweep_results:
    if r["cosine_sim"] < 0:
        zero_crossing = r["angle_deg"]
        break
if zero_crossing is None:
    zero_crossing = 180  # nunca cruza (tareas siempre complementarias)
# Sharpness: qué tan claro es el cruce
cos_values = [r["cosine_sim"] for r in sweep_results]
cos_range = max(cos_values) - min(cos_values)
sharpness = min(1.0, cos_range * 2)
score = min(1.0, 0.3 + sharpness * 0.4 + (1 - abs(zero_crossing - 90)/90) * 0.3)
_result = {{
    "action": "interference_angle_sweep",
    "sweep": sweep_results,
    "zero_crossing_angle": float(zero_crossing),
    "cosine_range": round(float(cos_range), 4),
    "sharpness": round(float(sharpness), 4),
    "score": round(score, 4),
}}
"""
        r   = executor.run(code, label=f"angle_sweep_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.40:
            tracker.record(
                f"Ciclo {cycle}: cruce-cero de coseno de gradientes en "
                f"{res.get('zero_crossing_angle',90):.0f}° "
                f"(sharpness={res.get('sharpness',0):.3f})",
                domain="continual", confidence=0.68,
                evidence="sweep ángulo 0°–180°, d_in=20",
            )
        return score, res.get("zero_crossing_angle")

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
        return score, res.get("retention_improvement")

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
        return score, res.get("mean_safe_params")

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
        return score, res.get("sim_forgetting_corr")

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
        return score, res.get("retention_gain")


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher (ahora retorna key_metric)
# ══════════════════════════════════════════════════════════════════════════════

def execute_action(action: str, problem: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker) -> tuple[float, dict, float | None]:
    log = {"action": action, "cycle": cycle, "executor_results": []}
    if problem == "causal":
        score, key_metric = _causal_action(action, cycle, executor, tracker, log)
    else:
        score, key_metric = _continual_action(action, cycle, executor, tracker, log)
    tracker.record_attempt(description=action, result=f"score={score:.3f}", domain=problem)
    log["progress_score"] = score
    log["key_metric"]     = key_metric
    return score, log, key_metric


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
                cross_analyzer, finding=None, verbose=True):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results_v012_{session_id}.json"
    try:
        data = {
            "version":        VERSION,
            "session_id":     session_id,
            "timestamp":      ts,
            "tracker":        tracker.self_report(),
            "meter":          meter.summary(),
            "tools":          tool_reg.summary(),
            "oracle":         oracle.summary(),
            "pool":           pool.summary(),
            "memory":         memory.summary(),
            "inventor":       inventor.summary(),
            "epistemic":      epi.summary(),
            "evidence_bank":  evidence_bank.summary(),
            "cross_analyzer": cross_analyzer.summary(),
            "finding":        finding,
            "logs":           all_logs[-50:],
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, cls=_Enc, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n  [Guardado] → {path}")
    except Exception as e:
        if verbose:
            print(f"\n  [Error guardado] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Oracle helpers
# ══════════════════════════════════════════════════════════════════════════════

def _oracle_consolidate(oracle, tracker, memory, evidence_bank,
                        problem: str, cycle: int) -> str:
    """Síntesis oracle al entrar en CONSOLIDATE."""
    if not oracle.available or oracle._spent >= oracle._budget - 0.50:
        return "(oracle no disponible)"
    supported_summary = "\n".join(
        f"  - {a}: score_mean={v['score_mean']:.3f}"
        for a, v in evidence_bank.supported.items()
    )
    recent_scores = {
        a: {"avg": v["score_mean"], "n": v["evidence_cycles"]}
        for a, v in evidence_bank.supported.items()
    }
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
        if "direction" in result:
            return str(result["direction"])
        raw = result.get("raw", "")
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


def _oracle_propose_experiments(oracle, tracker, memory, evidence_bank,
                                cross_analyzer, problem: str, cycle: int) -> list[dict]:
    """
    Pide al oracle 2 experimentos específicos y ejecutables.
    Retorna lista de {"hypothesis": str, "code": str, "expected": str}.
    """
    if not oracle.available or oracle._spent >= oracle._budget - 0.20:
        return []
    supported_summary = "\n".join(
        f"  - {a}: score_mean={v['score_mean']:.3f}"
        for a, v in evidence_bank.supported.items()
    )
    cross_summary = ""
    if cross_analyzer.cross_hypotheses:
        cross_summary = "\nCorrelaciones cruzadas detectadas:\n" + "\n".join(
            f"  - {h['hypothesis']}" for h in cross_analyzer.cross_hypotheses[:3]
        )
    try:
        result = oracle.propose_verify_experiments(problem, supported_summary, cross_summary)
        if not result:
            return []
        exps = result.get("experiments", [])
        if isinstance(exps, list):
            return [e for e in exps if isinstance(e, dict) and "code" in e]
        return []
    except Exception as e:
        print(f"  [Oracle/verify] error: {e}")
        return []


_SCIPY_STUB = """\
# ── scipy stub (solo numpy) ──────────────────────────────────────────────────
import sys, types as _types
_scipy_mod  = _types.ModuleType("scipy")
_stats_mod  = _types.ModuleType("scipy.stats")

def _pearsonr(x, y):
    import numpy as _np
    r = float(_np.corrcoef(x, y)[0, 1])
    return r, None

def _spearmanr(x, y):
    import numpy as _np
    rx = _np.argsort(_np.argsort(x)); ry = _np.argsort(_np.argsort(y))
    r = float(_np.corrcoef(rx, ry)[0, 1])
    return r, None

def _linregress(x, y):
    import numpy as _np
    x, y = _np.array(x, float), _np.array(y, float)
    mx, my = x.mean(), y.mean()
    slope = float(_np.sum((x-mx)*(y-my)) / (_np.sum((x-mx)**2)+1e-12))
    inter = float(my - slope*mx)
    class _R:
        pass
    r = _R(); r.slope=slope; r.intercept=inter; r.rvalue=0.0; r.pvalue=1.0; r.stderr=0.0
    return r

class _NormDist:
    def cdf(self, x, loc=0, scale=1):
        import numpy as _np
        z = (_np.asarray(x)-loc)/scale
        return float(0.5*(1+_np.sign(z)*_np.sqrt(1-_np.exp(-2*z**2/_np.pi))))
    def ppf(self, p): return float(np.sqrt(2)*np.sign(p-0.5)*np.sqrt(-np.log(4*np.minimum(p,1-p)**2+1e-12)))

_stats_mod.pearsonr   = _pearsonr
_stats_mod.spearmanr  = _spearmanr
_stats_mod.linregress = _linregress
_stats_mod.norm       = _NormDist()
_scipy_mod.stats      = _stats_mod
sys.modules["scipy"]        = _scipy_mod
sys.modules["scipy.stats"]  = _stats_mod
stats = _stats_mod
# ─────────────────────────────────────────────────────────────────────────────
"""

def _sanitize_verify_code(code: str) -> str:
    """
    Prepend scipy stub; remove scipy import lines so el código oracle
    (que frecuentemente usa scipy.stats) funcione en DirectExecutor (solo numpy).
    """
    import re
    # Eliminar líneas de import scipy (el stub ya lo provee vía sys.modules)
    code = re.sub(r"^\s*(?:import scipy[^\n]*|from scipy[^\n]*)\n", "", code, flags=re.MULTILINE)
    return _SCIPY_STUB + code


def _run_verify_experiments(experiments: list[dict], executor: DirectExecutor,
                            cycle: int, memory) -> list[dict]:
    """Ejecuta cada experimento propuesto por el oracle. Retorna resultados."""
    verify_results = []
    for i, exp in enumerate(experiments):
        hypothesis   = exp.get("hypothesis", f"exp_{i}")
        code         = exp.get("code", "")
        expected     = exp.get("expected_if_true", "")
        if not code or len(code) < 10:
            verify_results.append({
                "hypothesis": hypothesis,
                "success": False,
                "result": {},
                "error": "código vacío",
                "confirmed": False,
            })
            continue
        code = _sanitize_verify_code(code)
        r   = executor.run(code, label=f"verify_{cycle}_{i}")
        res = r.get("result") or {}
        err = r.get("error", "")
        success   = not err and bool(res)
        confirmed = bool(res.get("confirms", success))  # el código decide si confirma
        entry = {
            "hypothesis":   hypothesis[:120],
            "expected":     expected[:120],
            "success":      success,
            "confirmed":    confirmed,
            "result":       {k: v for k, v in res.items() if k != "code"}
                            if isinstance(res, dict) else str(res)[:200],
            "error":        str(err)[:120] if err else "",
        }
        verify_results.append(entry)
        status = "✓" if success else "✗"
        verdict = "CONFIRMA" if confirmed else ("REFUTA" if success else "ERROR")
        print(f"  [VERIFY/{status}/{verdict}] {hypothesis[:70]}")
        if success:
            memory.index(
                summary=f"VERIFY: {hypothesis[:80]} → {verdict}",
                score=0.85 if confirmed else 0.75,
                cycle=cycle, source="verify",
                tags=["verify", "oracle"],
                result=entry["result"] if isinstance(entry["result"], dict) else {},
            )
    return verify_results


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

ACTION_TAGS = {
    "backdoor_adjustment":       ["causal", "backdoor", "confunder", "ajuste"],
    "do_calculus_test":          ["causal", "do-calculus", "dag", "identificacion"],
    "irm_vs_erm":                ["causal", "irm", "invarianza", "entornos"],
    "discovery_threshold_sweep": ["causal", "descubrimiento", "esqueleto", "umbral", "sweep"],
    "counterfactual_bounds":     ["causal", "contrafactual", "cotas", "tian-pearl"],
    "interference_angle_sweep":  ["continual", "gradiente", "angulo", "interferencia", "sweep"],
    "ewc_retention":             ["continual", "ewc", "fisher", "retencion"],
    "fisher_geometry":           ["continual", "fisher", "geometria", "parametros"],
    "task_similarity":           ["continual", "similitud", "tareas", "olvido"],
    "replay_vs_finetune":        ["continual", "replay", "memoria", "retencion"],
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",    choices=["causal","continual"], default="causal")
    parser.add_argument("--max-hours",  type=float, default=8.0)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--cold-start", action="store_true",
                        help="Ignorar estado epistémico previo — inicio desde cero")
    parser.add_argument("--agent-id",   type=str, default=None,
                        help="ID único para runs paralelos (afecta nombre del epistemic_state)")
    args = parser.parse_args()

    api_key  = os.getenv("ANTHROPIC_API_KEY")
    deadline = time.time() + args.max_hours * 3600

    _agent_suffix = f"_{args.agent_id}" if args.agent_id else ""
    session_id = datetime.now().strftime(f"v012_{args.problem}_%Y%m%d_%H%M%S") + _agent_suffix
    Path("results").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  agente vX {VERSION}  |  problema: {args.problem}  |  max {args.max_hours}h")
    print(f"  Fases: SURVEY → FOCUS → CONSOLIDATE → VERIFY → STOP")
    print(f"  Para: ≥{SOLUTION_MIN_SUPPORTED} hipótesis + richness ≥ {SOLUTION_MIN_RICHNESS}")
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
    actions       = PROBLEM_ACTIONS[args.problem]
    epi           = EpistemicState(actions)
    evidence_bank = EvidenceBank(actions)
    cross_analyzer = CrossActionAnalyzer(actions)

    # ── Persistencia epistémica (v0.1.3) ────────────────────────────────────
    state_mgr  = EpistemicStateManager(args.problem, agent_id=args.agent_id)
    prior_state = state_mgr.load()
    if prior_state and not args.cold_start:
        n_prior = state_mgr.apply(prior_state, evidence_bank, cross_analyzer, epi)
        if evidence_bank.n_supported() >= SOLUTION_MIN_SUPPORTED:
            # Ya tenemos suficientes hipótesis — saltamos SURVEY y arrancamos en FOCUS.
            # _focus_cycles arranca en 0: el agente debe trabajar FOCUS_MIN_CYCLES_BEFORE_STOP
            # ciclos buscando hallazgos NUEVOS antes de poder transicionar.
            epi.phase = PHASE_FOCUS
            epi._focus_cycles = 0
            print(f"[Estado previo] {n_prior} hipótesis restauradas — "
                  f"saltando a FOCUS (buscando hallazgos nuevos)")
        else:
            print(f"[Estado previo] {n_prior} hipótesis restauradas — "
                  f"continuando exploración normal")
    else:
        if args.cold_start:
            print("[Inicio frío] --cold-start: ignorando estado previo")
        else:
            print("[Inicio frío] no hay estado previo — empezando desde cero")

    # ── Estado del loop ──────────────────────────────────────────────────────
    global_cycle   = 0
    perspective    = "mathematician"
    all_logs: list = []
    last_oracle    = 0.0
    crisis_count   = 0
    finding_document: dict | None = None
    _verify_experiments: list[dict] = []   # experimentos propuestos por oracle
    _focus_cycle_count = 0                 # para disparar CrossAnalyzer

    _action_stagnation: dict[str, int]    = {a: 0 for a in actions}
    _dormant:           set[str]          = set()
    _dormant_avg:       dict[str, float]  = {}
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
                  f"richness: {evidence_bank.richness():.3f}  "
                  f"cross: {len(cross_analyzer.cross_hypotheses)}")
            print(f"{'─'*70}")

            # ── VERIFY: ejecutar experimentos propuestos por oracle ───────────
            if epi.phase == PHASE_VERIFY:
                print(f"\n  [VERIFY] Ejecutando {len(_verify_experiments)} experimentos oracle...")
                verify_results = _run_verify_experiments(
                    _verify_experiments, executor, global_cycle, memory
                )
                n_confirmed = sum(1 for r in verify_results if r.get("confirmed"))
                n_run       = sum(1 for r in verify_results if r.get("success"))
                print(f"  [VERIFY] ejecutados={n_run}/{len(_verify_experiments)}  "
                      f"confirmados={n_confirmed}")

                richness  = evidence_bank.richness()
                n_sup     = evidence_bank.n_supported()
                if n_confirmed >= 1:
                    result_type = "verified"
                elif n_sup >= SOLUTION_MIN_SUPPORTED and richness >= SOLUTION_MIN_RICHNESS:
                    result_type = "convergent"
                else:
                    result_type = "inconclusive"

                finding_document = build_finding_document(
                    result_type, evidence_bank, memory, tracker, meter,
                    global_cycle,
                    oracle_synthesis=str(_oracle_synthesis_text),
                    verify_results=verify_results,
                    cross_hypotheses=cross_analyzer.cross_hypotheses,
                )
                print(f"\n  {'★'*60}")
                print(f"  FINDING: {result_type.upper()}")
                print(f"  richness={richness:.3f}  supported={n_sup}  ciclos={global_cycle}")
                if cross_analyzer.cross_hypotheses:
                    print(f"  cross_hypotheses={len(cross_analyzer.cross_hypotheses)}")
                print(f"  {'★'*60}\n")
                break

            # ── CONSOLIDATE: síntesis + propuesta de experimentos ────────────
            if epi.phase == PHASE_CONSOLIDATE:
                print(f"\n  [CONSOLIDATE] Sintetizando hallazgos...")
                _oracle_synthesis_text = _oracle_consolidate(
                    oracle, tracker, memory, evidence_bank,
                    args.problem, global_cycle
                )
                last_oracle = time.time()
                print(f"  [Oracle/síntesis] {_oracle_synthesis_text[:150]}")

                # Proponer experimentos para VERIFY
                print(f"\n  [CONSOLIDATE→VERIFY] Solicitando experimentos al oracle...")
                _verify_experiments = _oracle_propose_experiments(
                    oracle, tracker, memory, evidence_bank,
                    cross_analyzer, args.problem, global_cycle
                )
                print(f"  [VERIFY] {len(_verify_experiments)} experimentos propuestos")

                # Transición directa a VERIFY
                epi.enter_phase(PHASE_VERIFY)
                continue  # siguiente iteración ejecuta VERIFY

            # ── Elegir acción según fase ─────────────────────────────────────
            emphasis       = PERSPECTIVE_EMPHASIS.get(perspective, list(range(5)))
            active_indices = [i for i in emphasis if actions[i] not in _dormant]
            if not active_indices:
                _dormant.clear(); _dormant_avg.clear()
                active_indices = list(emphasis)
                print("  [Limbo] Todas dormantes — reviviendo todas")
            active_actions = [actions[i] for i in active_indices]

            if epi.phase == PHASE_SURVEY:
                action_idx = active_indices[global_cycle % len(active_indices)]
                action     = actions[action_idx]
            else:
                action = epi.most_curious_action(active_actions)

            # ── Ejecutar acción ──────────────────────────────────────────────
            score, attack_log, key_metric = execute_action(
                action, args.problem, global_cycle, executor, tracker
            )
            all_logs.append(attack_log)
            for res in attack_log["executor_results"]:
                print(f"  [{action}] {str(res)[:140]}")

            # ── Registrar key_metric en CrossActionAnalyzer ─────────────────
            cross_analyzer.record(action, key_metric)

            # ── Análisis de correlaciones cruzadas cada N ciclos FOCUS ───────
            if epi.phase == PHASE_FOCUS:
                _focus_cycle_count += 1
                if _focus_cycle_count % CROSS_ANALYSIS_INTERVAL == 0:
                    new_cross = cross_analyzer.analyze()
                    for ch in new_cross:
                        tracker.record(
                            f"Ciclo {global_cycle}: correlación cruzada — {ch['hypothesis']}",
                            domain=args.problem, confidence=0.60,
                            evidence=f"CrossActionAnalyzer r={ch['correlation']:.2f} n={ch['n']}",
                        )
                        memory.index(
                            summary=f"Cross-correlación: {ch['hypothesis']}",
                            score=0.75, cycle=global_cycle,
                            source="cross_analyzer",
                            tags=["cross_action", "correlation", args.problem],
                            result=ch,
                        )

            # ── Indexar en ResearchMemory ────────────────────────────────────
            tags  = ACTION_TAGS.get(action, [action, args.problem])
            hist  = _action_score_hist.setdefault(action, [])
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
                    continue
                if meter.satisfaction_score() > _dormant_avg.get(_da, 0.0) + 0.05:
                    _dormant.discard(_da)
                    _action_stagnation[_da] = 0
                    print(f"  [Limbo/revival] '{_da}' reactivada")

            # ── Medidor de satisfacción ──────────────────────────────────────
            state = meter.update(score)
            print(f"  [Meter] score={score:.3f}  state={state.value}  "
                  f"mem={mem_id}  key_metric={key_metric}  "
                  f"curious={epi.uncertainty.get(action,0):.4f}")

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
                    and oracle._spent < oracle._budget - ORACLE_RESERVE_BUDGET
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
            new_phase = epi.should_transition(
                evidence_bank.n_supported(),
                n_new=evidence_bank.n_new_this_session(),
                has_prior=prior_state is not None,
            )
            if new_phase:
                epi.enter_phase(new_phase)
                if new_phase == PHASE_CONSOLIDATE:
                    print(f"  [Fase] Entrando a CONSOLIDATE en ciclo {global_cycle+1}")

            # ── Memory snapshot cada 10 ciclos ────────────────────────────────
            if global_cycle % 10 == 0:
                ms = memory.summary()
                print(f"\n  [Memory] total={ms['total']} avg={ms['avg_score']:.3f}")
                top3 = memory.to_corpus_knowledge(top_k=3)
                print(f"  [Memory/top3]:\n{top3}")
                print(f"  [Epistemic] {epi.summary()}")
                print(f"  [CrossAnalyzer] {cross_analyzer.summary()}")

            # ── Guardado periódico cada 25 ciclos ────────────────────────────
            if global_cycle % 25 == 0:
                _save_final(session_id, all_logs, tracker, meter, tool_reg,
                            oracle, pool, memory, inventor, epi, evidence_bank,
                            cross_analyzer, finding=None, verbose=False)

            # ── Pausa mínima ──────────────────────────────────────────────────
            elapsed_cycle = time.time() - t_cycle_start
            if elapsed_cycle < MIN_CYCLE_SEC:
                time.sleep(MIN_CYCLE_SEC - elapsed_cycle)

    except KeyboardInterrupt:
        print("\n  [Interrupción manual]")
    finally:
        pool.shutdown()
        if finding_document is None:
            oracle_synthesis_fallback = getattr(locals(), "_oracle_synthesis_text", "(sin síntesis)")
            finding_document = build_finding_document(
                "inconclusive", evidence_bank, memory, tracker,
                meter, global_cycle,
                oracle_synthesis=oracle_synthesis_fallback,
                cross_hypotheses=cross_analyzer.cross_hypotheses,
            )
        _save_final(session_id, all_logs, tracker, meter, tool_reg,
                    oracle, pool, memory, inventor, epi, evidence_bank,
                    cross_analyzer, finding=finding_document)

        # ── Guardar estado epistémico para próxima sesión ────────────────────
        if not args.cold_start:
            state_mgr.save(evidence_bank, cross_analyzer, epi, prior_state)

        print(f"\n[Fin] ciclos={global_cycle}  "
              f"resultado={finding_document['result']}  "
              f"richness={finding_document['finding_richness']:.3f}  "
              f"cross_hyps={len(cross_analyzer.cross_hypotheses)}")
