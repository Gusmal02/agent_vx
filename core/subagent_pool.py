"""
core/subagent_pool.py — Pool de subagentes paralelos (v0.0.9)

Aprovecha Python 3.14t free-threading (sin GIL) para exploración
genuinamente paralela de hipótesis indirectas.

Cada subagente es un thread que recibe:
  - Una hipótesis a explorar
  - El sandbox matemático compartido (thread-safe en 3.14t)
  - Un presupuesto de tiempo

Los subagentes son locales (no API) — rápidos y gratuitos.
El oráculo se invoca SOLO desde el agente principal para hipótesis críticas.

Hipótesis "indirectas": conexiones que el agente principal no está persiguiendo
directamente pero que tienen justificación matemática suficiente.
Ejemplo: mientras el agente ataca los ceros de ζ, un subagente explora
si la distribución espectral de matrices de Gram de vectores SAT también sigue GUE.
"""

from __future__ import annotations

import time
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class SubagentTask:
    task_id:    str
    hypothesis: str
    code:       str          # código Python a ejecutar en el sandbox
    domain:     str
    submitted:  float = field(default_factory=time.time)
    timeout_sec: float = 45.0


@dataclass
class SubagentResult:
    task_id:    str
    hypothesis: str
    domain:     str
    ok:         bool
    result:     Optional[dict]
    error:      Optional[str]
    elapsed:    float
    score:      float        # estimado del resultado


class SubagentPool:
    """
    Pool de workers thread para exploración paralela de hipótesis.

    En Python 3.14t (sin GIL) los threads corren en paralelo real —
    el sandbox puede ejecutar código simultáneamente en múltiples cores.
    """

    def __init__(self,
                 sandbox,
                 max_workers: int = 3,
                 max_queue:   int = 10,
                 ):
        self._sandbox     = sandbox
        self._executor    = ThreadPoolExecutor(max_workers=max_workers,
                                               thread_name_prefix="subagent")
        self._max_queue   = max_queue
        self._pending:    dict[str, Future] = {}
        self._results:    list[SubagentResult] = []
        self._lock        = threading.Lock()
        self._task_count  = 0

        print(f"  [SubagentPool] {max_workers} workers (3.14t free-threading)")

    # ── Envío de tareas ───────────────────────────────────────────────────────

    def submit(self, hypothesis: str, code: str, domain: str,
               timeout_sec: float = 45.0) -> Optional[str]:
        """
        Lanza un subagente para explorar `hypothesis` ejecutando `code`.
        Devuelve el task_id o None si la cola está llena.
        """
        with self._lock:
            if len(self._pending) >= self._max_queue:
                return None
            self._task_count += 1
            task_id = f"sub_{self._task_count:04d}"

        task   = SubagentTask(task_id, hypothesis, code, domain, timeout_sec=timeout_sec)
        future = self._executor.submit(self._run_task, task)

        with self._lock:
            self._pending[task_id] = future

        print(f"  [SubagentPool] lanzado {task_id}: {hypothesis[:60]}...")
        return task_id

    def submit_from_oracle(self, oracle_result: dict, domain: str) -> Optional[str]:
        """
        Lanza un subagente directamente desde la respuesta del oráculo.
        """
        code = oracle_result.get("exploration_code") or oracle_result.get("code")
        if not code:
            return None
        hypothesis = (oracle_result.get("connection")
                      or oracle_result.get("direction")
                      or "Exploración oracle")
        return self.submit(hypothesis, code, domain)

    # ── Ejecución interna ─────────────────────────────────────────────────────

    def _run_task(self, task: SubagentTask) -> SubagentResult:
        start = time.time()
        try:
            r = self._sandbox.run(task.code, label=task.task_id)
            elapsed = time.time() - start
            result  = r.get("result") or {}
            ok      = r["ok"]
            error   = r.get("error") if not ok else None
            # Estimar score del resultado
            score   = self._estimate_score(result, task.domain)
            return SubagentResult(
                task_id=task.task_id, hypothesis=task.hypothesis,
                domain=task.domain, ok=ok, result=result,
                error=error, elapsed=round(elapsed, 2), score=score,
            )
        except Exception as e:
            elapsed = time.time() - start
            return SubagentResult(
                task_id=task.task_id, hypothesis=task.hypothesis,
                domain=task.domain, ok=False, result=None,
                error=traceback.format_exc(limit=3),
                elapsed=round(elapsed, 2), score=0.0,
            )

    def _estimate_score(self, result: dict, domain: str) -> float:
        """
        Score basado en métricas numéricas reales del resultado.
        Prioriza campos cuantitativos antes que palabras clave.
        """
        if not result:
            return 0.0

        # ── Campo score explícito ─────────────────────────────────────────────
        if "score" in result:
            return max(0.0, min(1.0, float(result["score"])))

        scores = []

        # ── Riemann: ceros en línea crítica ──────────────────────────────────
        if "max_dev" in result:
            # max_dev = desviación máxima de Re(zero) respecto a 0.5
            # 0.0 = perfecto, >1e-5 = dudoso
            dev = float(result["max_dev"])
            scores.append(1.0 if dev < 1e-10 else max(0.0, 1.0 - dev * 1e6))
        if "all_on_line" in result:
            scores.append(1.0 if result["all_on_line"] else 0.2)
        if "n_checked" in result:
            # más ceros verificados = más confianza
            scores.append(min(1.0, float(result["n_checked"]) / 100.0))

        # ── Riemann: GUE / Wigner-Dyson ──────────────────────────────────────
        if "gue_mae" in result:
            mae = float(result["gue_mae"])
            scores.append(max(0.0, 1.0 - mae / 0.3))   # mae=0 → 1.0, mae=0.3 → 0.0
        if "gue_consistent" in result:
            scores.append(0.85 if result["gue_consistent"] else 0.25)
        if "ks_stat" in result:
            ks = float(result["ks_stat"])
            scores.append(max(0.0, 1.0 - ks * 3))
        if "wigner_ks_distance" in result:
            ks = float(result["wigner_ks_distance"])
            scores.append(max(0.0, 1.0 - ks * 2))

        # ── Riemann: densidad de primos ───────────────────────────────────────
        if "mean_riemann_ratio" in result:
            ratio = float(result["mean_riemann_ratio"])
            scores.append(max(0.0, 1.0 - ratio))
        if "max_rel_error" in result:
            err = float(result["max_rel_error"])
            scores.append(max(0.0, 1.0 - err * 5))
        if "consistent_rh" in result:
            scores.append(0.85 if result["consistent_rh"] else 0.20)
        if "prime_results" in result:
            pr = result["prime_results"]
            if isinstance(pr, list) and pr:
                avg = sum(r.get("ratio", 1.0) for r in pr) / len(pr)
                scores.append(max(0.0, 1.0 - avg))

        # ── Riemann: simetría ξ(s) = ξ(1-s) ─────────────────────────────────
        if "symmetry_verified" in result:
            scores.append(0.90 if result["symmetry_verified"] else 0.15)
        if "xi_errors" in result:
            errs = result["xi_errors"]
            if isinstance(errs, list) and errs:
                max_e = max(float(e) for e in errs)
                scores.append(1.0 if max_e < 1e-10 else max(0.0, 1.0 - max_e * 1e8))

        # ── PNP: GCT permanente ───────────────────────────────────────────────
        if "always_differ" in result:
            scores.append(0.80 if result["always_differ"] else 0.20)
        if "matrix_results" in result:
            mr = result["matrix_results"]
            if isinstance(mr, list) and mr:
                frac = sum(1 for r in mr if r.get("differ")) / len(mr)
                scores.append(frac)

        # ── PNP: SAT ─────────────────────────────────────────────────────────
        if "sat_results" in result:
            sr = result["sat_results"]
            if isinstance(sr, list) and sr:
                # ratio de exploración promedio (qué fracción del espacio recorrió)
                ratios = [r.get("ratio", 0) for r in sr if "ratio" in r]
                if ratios:
                    scores.append(min(1.0, sum(ratios) / len(ratios) * 2))

        # ── PNP: Deutsch-Jozsa real ───────────────────────────────────────────
        if "speedup" in result and "exponential_separation" in result:
            if result.get("separation_genuine"):  # DJ real, no harcoded
                speedup = float(result["speedup"])
                n_bits  = int(result.get("n_bits", 8))
                # genuino si speedup ≥ 2^(n/2)
                expected = 2 ** (n_bits // 2)
                scores.append(min(1.0, speedup / expected) if speedup >= expected else 0.3)
            else:
                scores.append(0.3)  # hardcoded no vale

        if "classical_accuracy" in result:
            scores.append(float(result["classical_accuracy"]))

        # ── Fallback por palabras clave solo si no hay métricas numéricas ─────
        if not scores:
            positives = ["separation", "convergence", "verified", "consistent",
                         "exponential", "differs", "gue"]
            text = str(result).lower()
            hits = sum(1 for p in positives if p in text)
            return min(1.0, 0.2 + hits * 0.12)

        return round(sum(scores) / len(scores), 4)

    # ── Recolección de resultados ─────────────────────────────────────────────

    def collect_completed(self) -> list[SubagentResult]:
        """Recoge los subagentes que terminaron sin bloquear."""
        completed = []
        with self._lock:
            done_ids = [tid for tid, f in self._pending.items() if f.done()]
            for tid in done_ids:
                future = self._pending.pop(tid)
                try:
                    result = future.result(timeout=0)
                    self._results.append(result)
                    completed.append(result)
                    status = "✓" if result.ok else "✗"
                    print(f"  [SubagentPool] {tid} {status} "
                          f"score={result.score:.2f} "
                          f"({result.elapsed}s) — {result.hypothesis[:50]}")
                except Exception as e:
                    print(f"  [SubagentPool] {tid} excepción: {e}")
        return completed

    def wait_all(self, timeout: float = 120.0) -> list[SubagentResult]:
        """Espera a todos los subagentes pendientes."""
        with self._lock:
            futures = list(self._pending.values())
        for f in as_completed(futures, timeout=timeout):
            pass
        return self.collect_completed()

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def n_pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def summary(self) -> dict:
        return {
            "tasks_submitted": self._task_count,
            "pending":         self.n_pending,
            "completed":       len(self._results),
            "avg_score":       round(
                sum(r.score for r in self._results) / max(1, len(self._results)), 3
            ),
        }

    def shutdown(self):
        self._executor.shutdown(wait=False, cancel_futures=True)
