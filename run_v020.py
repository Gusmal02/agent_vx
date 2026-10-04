"""
run_v020.py — agente vX v0.2.0  Multi-Agent Coordinator (versión B: swarm independiente)
═══════════════════════════════════════════════════════════════════════════════════════════

Arquitectura:
  N AgentWorkers corren en paralelo, cada uno con su propio estado epistémico.
  Al terminar, el MultiAgentCoordinator merge los resultados por quórum.

  Hipótesis robusta  = aparece en ≥ ceil(N/2) agentes
  Hipótesis candidata = aparece en < ceil(N/2) agentes (vale explorar más)
  Cross robusta       = mismo criterio de quórum sobre pares de acciones

Uso:
  uv run python run_v020.py --problem continual --n-agents 4 --max-hours 1
  uv run python run_v020.py --problem causal    --n-agents 6 --max-hours 2 --quorum 3
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


VERSION = "v0.2.0"

# ── Worker ────────────────────────────────────────────────────────────────────

def run_worker(agent_id: str, problem: str, max_hours: float,
               seed: int) -> dict:
    """
    Lanza run_v012.py como subproceso con --agent-id único.
    Retorna métricas extraídas del stdout + ruta al epistemic_state.
    """
    cmd = [
        sys.executable, "run_v012.py",
        "--problem",   problem,
        "--max-hours", str(max_hours),
        "--seed",      str(seed),
        "--agent-id",  agent_id,
        # sin --cold-start: agent_id único garantiza estado propio; sí guarda al final
    ]
    print(f"  [Worker {agent_id}] iniciando  seed={seed}")
    t0 = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent,
    )
    elapsed = time.time() - t0
    out = result.stdout + result.stderr

    def _ex(pattern):
        import re
        m = re.search(pattern, out)
        return m.group(1) if m else None

    cycles    = _ex(r"\[Fin\] ciclos=(\d+)")
    resultado = _ex(r"resultado=(\w+)")
    richness  = _ex(r"richness=([\d.]+)")
    cross     = _ex(r"cross_hyps=(\d+)")

    state_path = Path("results") / f"epistemic_state_{problem}_{agent_id}.json"

    print(f"  [Worker {agent_id}] fin  ciclos={cycles}  "
          f"richness={richness}  t={elapsed:.0f}s")

    return {
        "agent_id":   agent_id,
        "cycles":     int(cycles)     if cycles   else None,
        "resultado":  resultado,
        "richness":   float(richness) if richness else None,
        "cross_hyps": int(cross)      if cross    else None,
        "elapsed_s":  round(elapsed, 1),
        "state_path": str(state_path),
        "stdout_tail": out[-2000:],
    }


# ── Coordinator ───────────────────────────────────────────────────────────────

class MultiAgentCoordinator:

    def __init__(self, problem: str, n_agents: int, quorum: int | None = None):
        self.problem  = problem
        self.n_agents = n_agents
        self.quorum   = quorum if quorum else (n_agents // 2 + 1)

    def run(self, max_hours: float, base_seed: int = 42) -> dict:
        print(f"\n{'═'*65}")
        print(f"  agente vX {VERSION} — Multi-Agent Coordinator")
        print(f"  problema={self.problem}  agentes={self.n_agents}  "
              f"quórum={self.quorum}  max_hours={max_hours}")
        print(f"{'═'*65}\n")

        t0 = time.time()
        worker_results = []

        with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
            futures = {
                pool.submit(
                    run_worker,
                    f"a{i+1}",
                    self.problem,
                    max_hours,
                    base_seed + i * 7,
                ): i
                for i in range(self.n_agents)
            }
            for fut in as_completed(futures):
                try:
                    worker_results.append(fut.result())
                except Exception as e:
                    i = futures[fut]
                    print(f"  [Worker a{i+1}] ERROR: {e}")

        elapsed_total = time.time() - t0

        # Cargar estados epistémicos de cada worker
        states = []
        for wr in worker_results:
            sp = Path(wr["state_path"])
            if sp.exists():
                with open(sp, encoding="utf-8") as f:
                    states.append(json.load(f))
            else:
                print(f"  [Coordinator] advertencia: no encontré {sp}")

        merged = self._merge(states)
        report = self._build_report(worker_results, merged, elapsed_total)
        self._print_report(report)
        self._save(report)
        return report

    def _merge(self, states: list[dict]) -> dict:
        """
        Consolida N epistemic states por quórum.
        Hipótesis robusta = aparece en >= quorum agentes.
        """
        _empty = {"robust_supported": {}, "candidate_supported": {},
                  "robust_cross": [], "candidate_cross": [],
                  "n_states_merged": 0, "quorum": self.quorum}
        if not states:
            return _empty

        # ── Supported hypotheses ──────────────────────────────────────────
        hyp_votes: dict[str, list[dict]] = {}
        for s in states:
            for action, hyp in s.get("supported_hypotheses", {}).items():
                hyp_votes.setdefault(action, []).append(hyp)

        robust_supported   = {}
        candidate_supported = {}
        for action, votes in hyp_votes.items():
            avg_score = sum(v.get("score_mean", 0) for v in votes) / len(votes)
            summary = {
                "votes":      len(votes),
                "n_agents":   len(states),
                "score_mean": round(avg_score, 4),
                "statement":  votes[0].get("statement", ""),
                "agents":     [v.get("agent_id", "?") for v in votes
                               if "agent_id" in v],
            }
            if len(votes) >= self.quorum:
                robust_supported[action] = summary
            else:
                candidate_supported[action] = summary

        # ── Cross hypotheses ──────────────────────────────────────────────
        cross_votes: dict[tuple, list[dict]] = {}
        for s in states:
            for ch in s.get("cross_hypotheses", []):
                pair = (ch.get("action1",""), ch.get("action2",""))
                cross_votes.setdefault(pair, []).append(ch)

        robust_cross    = []
        candidate_cross = []
        for pair, votes in cross_votes.items():
            avg_r = sum(abs(v.get("correlation", v.get("r", 0))) for v in votes) / len(votes)
            entry = {
                "action1":  pair[0],
                "action2":  pair[1],
                "votes":    len(votes),
                "avg_abs_r": round(avg_r, 4),
                "sample":   votes[0],
            }
            if len(votes) >= self.quorum:
                robust_cross.append(entry)
            else:
                candidate_cross.append(entry)

        return {
            "robust_supported":    robust_supported,
            "candidate_supported": candidate_supported,
            "robust_cross":        robust_cross,
            "candidate_cross":     candidate_cross,
            "n_states_merged":     len(states),
            "quorum":              self.quorum,
        }

    def _build_report(self, worker_results: list[dict],
                      merged: dict, elapsed_total: float) -> dict:
        cycles = [r["cycles"] for r in worker_results if r["cycles"]]
        return {
            "version":       VERSION,
            "problem":       self.problem,
            "n_agents":      self.n_agents,
            "quorum":        self.quorum,
            "timestamp":     datetime.now().isoformat(),
            "elapsed_total": round(elapsed_total, 1),
            "workers":       worker_results,
            "worker_summary": {
                "cycles_mean": round(sum(cycles)/len(cycles), 1) if cycles else None,
                "cycles_min":  min(cycles) if cycles else None,
                "cycles_max":  max(cycles) if cycles else None,
                "converged":   sum(1 for r in worker_results
                                   if r.get("resultado") == "convergent"),
            },
            "merged": merged,
        }

    def _print_report(self, report: dict) -> None:
        ws  = report["worker_summary"]
        mrg = report["merged"]

        print(f"\n{'═'*65}")
        print(f"  REPORTE MULTI-AGENTE — {report['problem'].upper()}  {VERSION}")
        print(f"{'─'*65}")
        print(f"  Agentes: {report['n_agents']}  |  "
              f"Convergieron: {ws['converged']}/{report['n_agents']}  |  "
              f"Ciclos: {ws['cycles_min']}–{ws['cycles_max']} "
              f"(μ={ws['cycles_mean']})")
        print(f"  Tiempo total: {report['elapsed_total']:.0f}s")

        print(f"\n  HALLAZGOS ROBUSTOS (votos ≥ {report['quorum']}):")
        if mrg["robust_supported"]:
            for action, h in mrg["robust_supported"].items():
                print(f"    ✓ [{h['votes']}/{report['n_agents']}]  "
                      f"{action}  score={h['score_mean']:.3f}")
        else:
            print("    (ninguno alcanzó quórum)")

        if mrg["robust_cross"]:
            print(f"\n  CORRELACIONES CRUZADAS ROBUSTAS:")
            for ch in mrg["robust_cross"]:
                print(f"    ✓ [{ch['votes']}/{report['n_agents']}]  "
                      f"{ch['action1']} ↔ {ch['action2']}  "
                      f"|r|={ch['avg_abs_r']:.3f}")

        if mrg["candidate_supported"]:
            print(f"\n  CANDIDATOS (< quórum, explorar más):")
            for action, h in mrg["candidate_supported"].items():
                print(f"    ? [{h['votes']}/{report['n_agents']}]  {action}")

        print(f"{'═'*65}\n")

    def _save(self, report: dict) -> None:
        ts  = datetime.now().strftime("%Y%m%d_%H%M")
        out = Path("results") / f"multiagente_{self.problem}_{ts}.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[Guardado] → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="agente vX v0.2 — loop multi-agente (swarm independiente + coordinator)")
    parser.add_argument("--problem",   choices=["causal","continual"], default="causal")
    parser.add_argument("--n-agents",  type=int,   default=4)
    parser.add_argument("--max-hours", type=float, default=1.0)
    parser.add_argument("--quorum",    type=int,   default=None,
                        help="Votos mínimos para hipótesis robusta (default: ceil(N/2))")
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    coordinator = MultiAgentCoordinator(
        problem  = args.problem,
        n_agents = args.n_agents,
        quorum   = args.quorum,
    )
    coordinator.run(max_hours=args.max_hours, base_seed=args.seed)
