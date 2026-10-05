"""
run_v020.py — agente vX v0.3.0  Multi-Agent Coordinator
════════════════════════════════════════════════════════

Arquitectura:
  Ronda 1:  N workers con SURVEY randomizado por seed  →  divergencia real
  Monitor:  lee epistemic states, detecta frontera, genera misiones
  Ronda 2+: workers arrancan con estado previo + misión asignada
  Disruptor: un agente especial que intenta FALSIFICAR los hallazgos robustos

  Hipótesis robusta  = ≥ ceil(N/2) votos
  Hipótesis frontera = < ceil(N/2) votos  (candidatas a investigar más)

Uso:
  uv run python run_v020.py --problem causal  --n-agents 4 --rounds 2 --max-hours 2
  uv run python run_v020.py --problem causal  --n-agents 4 --rounds 4 --max-hours 2 --disruptor
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


VERSION = "v0.5.0"

# ── Worker ────────────────────────────────────────────────────────────────────

def run_worker(agent_id: str, problem: str, max_hours: float,
               seed: int, cold_start: bool = True,
               corpus_path: str | None = None,
               oracle_model: str | None = None) -> dict:
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
    ] + (["--cold-start"] if cold_start else []) \
      + (["--corpus-path", corpus_path] if corpus_path else []) \
      + (["--oracle-model", oracle_model] if oracle_model else [])
    print(f"  [Worker {agent_id}] iniciando  seed={seed}")
    t0 = time.time()
    # Fix: timeout explícito para que el coordinator nunca se bloquee para siempre.
    # Fix: output a archivo en lugar de capture_output para visibilidad en tiempo real
    #      y para evitar que el pipe llene el buffer en runs largos.
    worker_timeout = max_hours * 3600 + 300   # max_hours + 5 min de gracia
    log_path = Path(__file__).parent / "results" / f"worker_{problem}_{agent_id}.log"
    log_path.parent.mkdir(exist_ok=True)
    try:
        with open(log_path, "wb") as log_fh:
            result = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=log_fh,
                env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"},
                cwd=Path(__file__).parent,
                timeout=worker_timeout,
            )
    except subprocess.TimeoutExpired:
        print(f"  [Worker {agent_id}] TIMEOUT tras {worker_timeout:.0f}s")
        result = None
    except BaseException as _exc:
        print(f"  [Worker {agent_id}] EXCEPCIÓN en subprocess: {type(_exc).__name__}: {_exc}")
        result = None
    elapsed = time.time() - t0
    out = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

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

    def run(self, max_hours: float, base_seed: int = 42,
            n_rounds: int = 1, use_disruptor: bool = False,
            use_researcher: bool = False,
            use_meta_monitor: bool = False,
            oracle_model: str | None = None,
            api_key: str | None = None) -> dict:
        print(f"\n{'═'*65}")
        print(f"  agente vX {VERSION} — Multi-Agent Coordinator")
        print(f"  problema={self.problem}  agentes={self.n_agents}  "
              f"quórum={self.quorum}  max_hours={max_hours}  rondas={n_rounds}")
        if oracle_model:
            print(f"  oracle_model={oracle_model}")
        if use_disruptor:
            print(f"  disruptor=ON")
        if use_researcher:
            print(f"  researcher=ON")
        if use_meta_monitor:
            print(f"  meta_monitor=ON")
        print(f"{'═'*65}\n")

        # Instalar handler de SIGTERM para salida limpia (container recycle)
        _original_sigterm = signal.getsignal(signal.SIGTERM)
        def _handle_sigterm(signum, frame):
            print(f"  [Coordinator] SIGTERM recibido — salida limpia", flush=True)
            signal.signal(signal.SIGTERM, _original_sigterm)
            raise SystemExit(0)
        signal.signal(signal.SIGTERM, _handle_sigterm)

        monitor     = ResearchMonitor(self.problem, self.quorum, self.n_agents)
        agent_ids   = [f"a{i+1}" for i in range(self.n_agents)]
        corpus_path: str | None = None

        # ── Resiliencia: retomar run en curso si existe checkpoint ────────────
        checkpoint_path = Path("results") / f"coordinator_checkpoint_{self.problem}.json"
        all_rounds: list = []
        resume_from_round = 1
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, encoding="utf-8") as _f:
                    _ckpt = json.load(_f)
                all_rounds       = _ckpt.get("all_rounds", [])
                resume_from_round = len(all_rounds) + 1
                corpus_path_saved = _ckpt.get("corpus_path")
                if corpus_path_saved and Path(corpus_path_saved).exists():
                    corpus_path = corpus_path_saved
                if resume_from_round <= n_rounds:
                    print(f"  [Checkpoint] retomando desde ronda {resume_from_round}/{n_rounds} "
                          f"({len(all_rounds)} rondas completadas)")
                else:
                    print(f"  [Checkpoint] run ya completado ({len(all_rounds)} rondas) — iniciando nuevo")
                    all_rounds = []
                    resume_from_round = 1
                    checkpoint_path.unlink(missing_ok=True)
            except Exception as _e:
                print(f"  [Checkpoint] error leyendo checkpoint: {_e} — iniciando desde cero")
                all_rounds = []
                resume_from_round = 1

        # ── Researcher: construye corpus antes de la primera ronda ────────────
        if use_researcher:
            from core.researcher import ResearcherAgent
            researcher = ResearcherAgent(self.problem, api_key)
            corpus = researcher.build_corpus(n_papers=5, max_hyps_per_paper=2)
            _cp = Path("results") / f"corpus_{self.problem}.json"
            corpus_path = str(_cp) if _cp.exists() else None

        # ── MetaMonitor: background thread observador ─────────────────────────
        meta_thread = None
        meta_agent  = None
        if use_meta_monitor and api_key:
            from core.meta_monitor import MetaMonitorAgent
            meta_agent = MetaMonitorAgent(
                problems     = [self.problem],
                api_key      = api_key,
                model        = oracle_model or "claude-sonnet-4-6",
                budget_usd   = 1.0,
                interval_min = max(10.0, max_hours * 60 / 4),  # 4 síntesis por corrida
            )
            meta_thread = meta_agent.start()

        for round_n in range(resume_from_round, n_rounds + 1):
            print(f"\n{'─'*65}")
            print(f"  RONDA {round_n}/{n_rounds}")
            print(f"{'─'*65}")

            t0 = time.time()
            worker_results = []
            # Frío solo si es la primera ronda real del run (no hay estado guardado)
            cold = (round_n == 1 and resume_from_round == 1)

            import threading
            _done_event = threading.Event()

            with ThreadPoolExecutor(max_workers=self.n_agents) as pool:
                futures = {
                    pool.submit(
                        run_worker,
                        agent_ids[i],
                        self.problem,
                        max_hours,
                        base_seed + i * 7,
                        cold,
                        corpus_path,
                        oracle_model,
                    ): i
                    for i in range(self.n_agents)
                }

                def _heartbeat():
                    while not _done_event.wait(30):
                        alive = sum(1 for f in futures if not f.done())
                        print(f"  [Heartbeat] esperando {alive} workers  t={time.time()-t0:.0f}s",
                              flush=True)
                threading.Thread(target=_heartbeat, daemon=True).start()

                for fut in as_completed(futures):
                    try:
                        worker_results.append(fut.result())
                    except BaseException as e:
                        i = futures[fut]
                        print(f"  [Worker {agent_ids[i]}] ERROR ({type(e).__name__}): {e}",
                              flush=True)

            _done_event.set()

            elapsed = time.time() - t0

            # Espera para que el FS flush los archivos (más lento en cloud/Linux)
            time.sleep(5)
            states = []
            for wr in worker_results:
                sp = Path(wr["state_path"])
                # Reintentar hasta 10 veces (10s extra) — cloud puede ser lento
                for attempt in range(10):
                    if sp.exists():
                        break
                    time.sleep(1)
                    if attempt == 4:
                        print(f"  [Coordinator] esperando {sp} ({attempt+1}/10)...")
                if sp.exists():
                    with open(sp, encoding="utf-8") as f:
                        states.append(json.load(f))
                else:
                    print(f"  [Coordinator] advertencia: no encontré {sp}")

            merged   = self._merge(states)
            analysis = monitor.analyze(agent_ids)
            monitor.report(analysis)
            missions = monitor.assign_missions(analysis, agent_ids)
            missions_path = monitor.save_missions(missions, round_n)
            print(f"  [Monitor] misiones → {missions_path}")

            disruptor_summary = None
            if use_disruptor and merged.get("robust_supported"):
                disruptor = DisruptorAgent(self.problem, api_key)
                disruptor_summary = disruptor.run(merged["robust_supported"], round_n)

            round_report = self._build_report(worker_results, merged, elapsed)
            round_report["round"]              = round_n
            round_report["missions"]           = missions
            round_report["disruptor_summary"]  = disruptor_summary
            self._print_report(round_report)
            all_rounds.append(round_report)

            # Guardar checkpoint del coordinator para poder retomar si muere
            try:
                checkpoint_path.parent.mkdir(exist_ok=True)
                with open(checkpoint_path, "w", encoding="utf-8") as _f:
                    json.dump({
                        "problem":      self.problem,
                        "n_agents":     self.n_agents,
                        "n_rounds":     n_rounds,
                        "completed":    round_n,
                        "corpus_path":  corpus_path,
                        "all_rounds":   all_rounds,
                        "saved_at":     datetime.utcnow().isoformat(),
                    }, _f, ensure_ascii=False, indent=2)
                print(f"  [Checkpoint] ronda {round_n} guardada → {checkpoint_path}")
            except Exception as _e:
                print(f"  [Checkpoint] error guardando: {_e}")

        # ── Detener MetaMonitor y registrar su resumen ────────────────────────
        if meta_agent:
            meta_agent.stop()
            ms = meta_agent.summary()
            print(f"  [MetaMonitor] síntesis finales={ms['calls']}  "
                  f"gastado=${ms['spent_usd']:.3f}")

        final = {**all_rounds[-1], "all_rounds": all_rounds}
        if meta_agent:
            final["meta_monitor"] = meta_agent.summary()
        self._save(final)

        # Run completado — borrar checkpoint para que el próximo arranque sea limpio
        checkpoint_path.unlink(missing_ok=True)
        print(f"  [Checkpoint] run completado — checkpoint eliminado")

        return final

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


# ── ResearchMonitor ───────────────────────────────────────────────────────────

class ResearchMonitor:
    """
    Entre rondas: lee epistemic states, detecta frontera, genera misiones.json.
    Una misión = dirección específica que un worker debe explorar en la siguiente ronda.
    """

    def __init__(self, problem: str, quorum: int, n_agents: int):
        self.problem  = problem
        self.quorum   = quorum
        self.n_agents = n_agents

    def analyze(self, agent_ids: list[str]) -> dict:
        """Lee estados de todos los workers y clasifica hipótesis."""
        supported_votes: dict[str, list[str]] = {}
        cross_votes: dict[tuple, list] = {}

        for aid in agent_ids:
            sp = Path("results") / f"epistemic_state_{self.problem}_{aid}.json"
            if not sp.exists():
                continue
            with open(sp, encoding="utf-8") as f:
                state = json.load(f)
            for action in state.get("supported_hypotheses", {}):
                supported_votes.setdefault(action, []).append(aid)
            for ch in state.get("cross_hypotheses", []):
                pair = (ch.get("action1",""), ch.get("action2",""))
                cross_votes.setdefault(pair, []).append(aid)

        robust   = {a: v for a, v in supported_votes.items() if len(v) >= self.quorum}
        frontier = {a: v for a, v in supported_votes.items() if 0 < len(v) < self.quorum}
        unseen   = []  # acciones que ningún agente soportó — posibles puntos ciegos

        return {
            "robust":   robust,
            "frontier": frontier,
            "unseen":   unseen,
            "cross_votes": {f"{p[0]}↔{p[1]}": v for p, v in cross_votes.items()},
        }

    def assign_missions(self, analysis: dict,
                        agent_ids: list[str]) -> dict[str, dict]:
        """
        Prioridad de misiones:
        1. Hipótesis soportadas en frontera (< quórum votos)
        2. Correlaciones cruzadas candidatas (< quórum votos)
        3. Exploración libre si todo es robusto
        """
        missions: dict[str, dict] = {}
        frontier_items  = list(analysis["frontier"].items())
        cross_candidates = [
            (pair, votes) for pair, votes in analysis["cross_votes"].items()
            if len(votes) < self.quorum
        ]

        task_pool = (
            [("explore_frontier", a, f"solo {len(v)}/{self.n_agents} agentes la soportaron")
             for a, v in frontier_items] +
            [("explore_cross", pair, f"correlación candidata con {len(v)}/{self.n_agents} votos")
             for pair, v in cross_candidates]
        )

        for i, aid in enumerate(agent_ids):
            if i < len(task_pool):
                kind, target, reason = task_pool[i]
                missions[aid] = {"type": kind, "target": target, "reason": reason}
            else:
                missions[aid] = {
                    "type":   "free_explore",
                    "target": None,
                    "reason": "todo robusto y sin candidatos — explorar libremente",
                }
        return missions

    def save_missions(self, missions: dict, round_n: int) -> Path:
        out = Path("results") / f"misiones_{self.problem}_r{round_n}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(missions, f, indent=2, ensure_ascii=False)
        return out

    def report(self, analysis: dict) -> None:
        print(f"\n  [Monitor] Robustos ({len(analysis['robust'])}): "
              f"{list(analysis['robust'].keys())}")
        print(f"  [Monitor] Frontera ({len(analysis['frontier'])}): "
              f"{list(analysis['frontier'].keys())}")
        if analysis["unseen"]:
            print(f"  [Monitor] Sin cubrir: {analysis['unseen']}")


# ── DisruptorAgent ────────────────────────────────────────────────────────────

class DisruptorAgent:
    """
    Agente especial: intenta FALSIFICAR los hallazgos robustos del swarm.
    No hace SURVEY normal — recibe hallazgos robustos y genera ataques directos.
    Usa el oracle para proponer código de falsificación, luego lo ejecuta.
    """

    def __init__(self, problem: str, api_key: str | None, budget_usd: float = 0.50):
        self.problem   = problem
        self.api_key   = api_key
        self.budget    = budget_usd
        self._results: list[dict] = []

    def run(self, robust_findings: dict, round_n: int) -> dict:
        if not robust_findings:
            print("  [Disruptor] sin hallazgos robustos que atacar")
            return {"attacks": [], "breaks_found": 0}

        print(f"\n  [Disruptor] atacando {len(robust_findings)} hallazgos robustos...")
        attacks = self._generate_attacks(robust_findings)
        results = self._execute_attacks(attacks)

        breaks = [r for r in results if r.get("breaks")]
        print(f"  [Disruptor] {len(breaks)}/{len(results)} ataques encontraron límites")

        summary = {
            "round":        round_n,
            "n_targeted":   len(robust_findings),
            "n_attacked":   len(results),
            "breaks_found": len(breaks),
            "attacks":      results,
        }
        self._save(summary, round_n)
        return summary

    def _generate_attacks(self, robust_findings: dict) -> list[dict]:
        if not self.api_key:
            return self._synthetic_attacks(robust_findings)

        try:
            import re
            from core.anthropic_http import Anthropic
            client = Anthropic(api_key=self.api_key)
            findings_text = "\n".join(
                f"  [{i+1}] {action}  score={h.get('score_mean',0):.3f}"
                for i, (action, h) in enumerate(robust_findings.items())
            )
            # Atacar solo el hallazgo más fuerte para evitar truncación JSON
            top = list(robust_findings.items())[0]
            findings_text = f"  {top[0]}  score={top[1].get('score_mean',0):.3f}"
            prompt = f"""Eres un agente crítico. Este hallazgo fue validado por múltiples agentes ({self.problem}):
{findings_text}

Escribe UN experimento Python (solo numpy, <15 líneas) que intente FALSIFICARLO en un caso límite extremo.
Responde SOLO en JSON sin texto extra:
{{"attacks": [{{"target": "{top[0]}", "code": "import numpy as np\\nnp.random.seed(99)\\n# ataque extremo\\n_result = {{\\"breaks\\": False, \\"condition\\": \\"caso extremo\\", \\"score\\": 0.0}}", "expected_break": "cuándo debería fallar"}}]}}"""

            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            text = resp.content[0].text
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                data = json.loads(m.group())
                attacks = data.get("attacks", [])
                if attacks:
                    return attacks
                print(f"  [Disruptor/oracle] JSON ok pero attacks=[]. "
                      f"Claves recibidas: {list(data.keys())}")
            else:
                print(f"  [Disruptor/oracle] respuesta sin JSON válido: {text[:120]}")
        except Exception as e:
            print(f"  [Disruptor/oracle] error: {e}")

        return self._synthetic_attacks(robust_findings)

    def _synthetic_attacks(self, robust_findings: dict) -> list[dict]:
        """Ataques sin oracle: variar parámetros en regímenes extremos."""
        attacks = []
        # Diferentes estrategias de ataque por índice
        strategies = [
            ("n=2 extremo",    2,   "n muy pequeño rompe señal"),
            ("n=1000 grande",  1000,"señal se diluye con n grande"),
            ("high_noise",     50,  "ruido dominante rompe correlación"),
            ("uniform_prior",  20,  "prior uniforme cancela el efecto"),
            ("low_variance",   30,  "varianza mínima colapsa la métrica"),
        ]
        for i, action in enumerate(robust_findings):
            strategy, n, reason = strategies[i % len(strategies)]
            seed_val = 100 + i * 17  # seed único por acción
            code = f"""import numpy as np
np.random.seed({seed_val})
# Ataque sintético: {strategy}
n = {n}
scores = np.random.uniform(0, 1, size=n)
baseline = np.random.uniform(0, 1, size=n)
effect = float(np.mean(scores) - np.mean(baseline))
_result = {{'breaks': abs(effect) < 0.05, 'condition': '{strategy}', 'score': abs(effect), 'target': '{action}'}}"""
            attacks.append({"target": action, "code": code,
                            "expected_break": reason})
        return attacks

    def _execute_attacks(self, attacks: list[dict]) -> list[dict]:
        import subprocess, sys, tempfile, os
        results = []
        for atk in attacks:
            code = atk.get("code", "")
            if not code:
                continue
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                                 delete=False, encoding="utf-8") as f:
                    f.write(code)
                    tmp = f.name
                proc = subprocess.run(
                    [sys.executable, tmp],
                    capture_output=True, text=True, timeout=30
                )
                os.unlink(tmp)
                # Extraer _result del output o ejecutar en-process
                local_ns: dict = {}
                exec(code, {"__builtins__": __builtins__}, local_ns)
                result = local_ns.get("_result", {})
                results.append({
                    "target":   atk["target"],
                    "breaks":   bool(result.get("breaks", False)),
                    "condition": result.get("condition", ""),
                    "score":    result.get("score", None),
                    "expected_break": atk.get("expected_break", ""),
                })
                status = "ROMPE" if result.get("breaks") else "resiste"
                print(f"    [{status}] {atk['target']}  "
                      f"condition='{result.get('condition','?')}'")
            except Exception as e:
                results.append({"target": atk["target"], "error": str(e), "breaks": False})
                print(f"    [error] {atk['target']}: {e}")
        return results

    def _save(self, summary: dict, round_n: int) -> None:
        out = Path("results") / f"disruptor_{self.problem}_r{round_n}.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [Disruptor] guardado → {out}")


# ── MultiAgentCoordinator (actualizado para multi-ronda) ──────────────────────

# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="agente vX v0.4 — loop multi-agente con researcher, swarm y disruptor")
    parser.add_argument("--problem",    choices=["causal","continual","riemann","pnp"], default="causal")
    parser.add_argument("--n-agents",   type=int,   default=4)
    parser.add_argument("--max-hours",  type=float, default=1.0)
    parser.add_argument("--quorum",     type=int,   default=None,
                        help="Votos mínimos para hipótesis robusta (default: ceil(N/2))")
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--rounds",     type=int,   default=1,
                        help="Número de rondas (ronda 2+ reutiliza estado epistémico)")
    parser.add_argument("--disruptor",    action="store_true",
                        help="Activar agente disruptor al final de cada ronda")
    parser.add_argument("--researcher",   action="store_true",
                        help="Activar ResearcherAgent: busca papers en arxiv antes del round 1")
    parser.add_argument("--meta-monitor", action="store_true",
                        help="Activar MetaMonitorAgent: síntesis LLM cross-problema cada N min")
    parser.add_argument("--oracle-model",
                        choices=["sonnet", "opus", "fable",
                                 "claude-sonnet-4-6", "claude-opus-5-5", "claude-fable-5-1"],
                        default=None,
                        help="Modelo del oracle (default: claude-sonnet-4-6)")
    args = parser.parse_args()

    # Normalizar nombre de modelo
    _model_map = {
        "sonnet": "claude-sonnet-4-6",
        "opus":   "claude-opus-5-5",
        "fable":  "claude-fable-5-1",
    }
    oracle_model = _model_map.get(args.oracle_model, args.oracle_model)

    api_key = os.getenv("ANTHROPIC_API_KEY")

    coordinator = MultiAgentCoordinator(
        problem  = args.problem,
        n_agents = args.n_agents,
        quorum   = args.quorum,
    )
    coordinator.run(
        max_hours        = args.max_hours,
        base_seed        = args.seed,
        n_rounds         = args.rounds,
        use_disruptor    = args.disruptor,
        use_researcher   = args.researcher,
        use_meta_monitor = args.meta_monitor,
        oracle_model     = oracle_model,
        api_key          = api_key,
    )
