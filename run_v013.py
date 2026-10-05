"""
run_v013.py — Agente vX v0.6.0 — Exploración autónoma
═══════════════════════════════════════════════════════

Arquitectura:
  CORPUS   → referencia. Lo que ya se sabe. No se re-explora.
  PIZARRA  → estado propio. Lo que el agente computa y observa.
  ORÁCULO  → consultor. Solo responde preguntas conceptuales. Nunca genera código.

El agente decide qué explorar. El oráculo orienta cuando el agente lo pregunta.
El código lo escribe el agente, no el LLM.

Uso:
  python run_v013.py --problem riemann --max-hours 0.5 --seed 42 --agent-id a1
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import mpmath
mpmath.mp.dps = 20

from core.blackboard    import Blackboard
from core.math_explorer import MathExplorer, ExplorationTarget
from core.math_tools    import (winding_number, isolate_zeros, verify_zero,
                                 scan_strip, zero_density, analyze_spacing)

VERSION = "v0.6.0"


# ── Oracle — solo preguntas conceptuales ─────────────────────────────────────

class Oracle:
    """
    El oráculo responde preguntas conceptuales del agente.
    NUNCA genera código. NUNCA dice qué explorar.
    Solo orienta: "esto se llama X", "hay una conjetura sobre esto".
    """

    def __init__(self, api_key: str | None, model: str = "claude-sonnet-4-6"):
        self._key   = api_key
        self._model = model
        self._calls = 0

    def ask(self, question: str) -> str:
        if not self._key:
            return "(sin oráculo — continúa explorando)"
        try:
            from core.anthropic_http import Anthropic
            client = Anthropic(api_key=self._key)
            resp = client.messages.create(
                model=self._model,
                max_tokens=400,
                system=(
                    "Eres un matemático consultor. El agente te hace preguntas sobre "
                    "lo que acaba de observar. Responde en 2-4 oraciones: "
                    "qué área matemática estudia esto, qué conjeturas o teoremas aplican, "
                    "qué dirección podría ser fructífera. "
                    "NO generes código. NO digas al agente qué calcular. "
                    "Solo da contexto matemático."
                ),
                messages=[{"role": "user", "content": question}]
            )
            self._calls += 1
            return resp.content[0].text.strip()
        except Exception as e:
            return f"(oráculo error: {e})"

    @property
    def calls(self):
        return self._calls


# ── Ciclo del agente ──────────────────────────────────────────────────────────

class AgentV013:

    def __init__(self, agent_id: str, problem: str, seed: int,
                 api_key: str | None, corpus_context: str | None = None):
        self.agent_id  = agent_id
        self.problem   = problem
        self.seed      = seed
        self.blackboard = Blackboard(agent_id, problem)
        self.explorer   = MathExplorer(rng_seed=seed)
        self.oracle     = Oracle(api_key)
        self._corpus_context = corpus_context
        self._cycles    = 0
        self._t_start   = 0.0

    def run(self, max_hours: float) -> dict:
        self._t_start = time.time()
        max_secs = max_hours * 3600
        print(f"  [Agente {self.agent_id}] iniciando  seed={self.seed}  "
              f"problema={self.problem}", flush=True)

        # Cargar corpus como contexto inicial de la pizarra
        if self._corpus_context:
            self.blackboard.record("corpus_load", {"context": self._corpus_context[:200]})

        # Punto de partida: frontera del conocimiento conocido
        # (corpus dice que los primeros ~15 ceros hasta t≈50 son conocidos)
        start_t = self._starting_frontier()

        while (time.time() - self._t_start) < max_secs:
            self._cycles += 1
            targets = self.explorer.next_targets(self.blackboard, n=2)

            if not targets:
                # Sin targets → saltar a t alta aleatoria
                ft = self.blackboard.frontier_t(50.0)
                t0 = ft + self.explorer._rng.uniform(10, 200)
                targets = [ExplorationTarget(
                    tool="scan", t_min=t0, t_max=t0 + self.explorer.WINDOW,
                    reason="sin targets — salto aleatorio"
                )]

            for target in targets:
                if (time.time() - self._t_start) >= max_secs:
                    break
                self._execute_target(target)

        return self._finish()

    def _starting_frontier(self) -> float:
        """
        El agente no re-explora lo que el corpus ya sabe.
        Empieza desde más allá del conocimiento conocido.
        """
        # El corpus indica que los primeros ceros hasta ~t=50 están verificados
        # El agente arranca más allá, con variación por seed para divergir
        base = 50.0
        # Offset controlado: diferentes seeds → fronteras distintas pero acotadas
        # Seed 42 → +4, seed 49 → +18, seed 56 → +12 (dentro de [50, 100])
        offset = (self.seed % 25) * 2.0
        return base + offset

    def _execute_target(self, target: ExplorationTarget):
        t0 = target.t_min
        t1 = target.t_max
        tool = target.tool

        print(f"    [{self.agent_id}] {tool}  t∈[{t0:.1f},{t1:.1f}]  {target.reason}",
              flush=True)

        result = None
        observation = None

        try:
            if tool == "scan":
                # Registrar región ANTES de procesar (evita re-scan en próximo ciclo)
                self.blackboard.record("scan", {"status": "in_progress"},
                                       t_min=t0, t_max=t1)
                hot_windows = scan_strip(t0, t1, step=2.0)
                result = {"hot_windows": hot_windows, "n_hot": len(hot_windows)}
                # Actualizar con resultado real
                self.blackboard.record("scan", result, t_min=t0, t_max=t1)

                # Para cada ventana caliente: aislar y verificar ceros (sin duplicados)
                seen_t = set()
                for win in hot_windows:
                    candidates = self._isolate_and_verify(
                        win["t_min"], win["t_max"], win["winding"])
                    for t_val in (candidates or []):
                        bucket = round(t_val, 2)
                        if bucket not in seen_t:
                            seen_t.add(bucket)

            elif tool == "density":
                result = zero_density(t0, t1)
                surprise = (result["ratio"] < 0.5 or result["ratio"] > 2.0)
                self.blackboard.record("density", result, t_min=t0, t_max=t1,
                                       surprise=surprise)
                if surprise:
                    print(f"    [{self.agent_id}] ⚡ densidad inusual "
                          f"ratio={result['ratio']:.3f}", flush=True)

            elif tool == "spacing":
                zeros = self.blackboard.all_zeros()
                if len(zeros) >= 2:
                    result = analyze_spacing(zeros)
                    surprise = (result.get("gue_ratio", 1.0) < 0.5 or
                                result.get("gue_ratio", 1.0) > 2.0)
                    self.blackboard.record("spacing", result, t_min=t0, t_max=t1,
                                           surprise=surprise)
                    if surprise:
                        print(f"    [{self.agent_id}] ⚡ espaciado inusual "
                              f"gue_ratio={result.get('gue_ratio'):.3f}", flush=True)

        except Exception as e:
            self.blackboard.record("error", {"tool": tool, "error": str(e)},
                                   t_min=t0, t_max=t1)
            return

        # ── ¿Preguntar al oráculo? ────────────────────────────────────────────
        if result:
            obs = {"tool": tool, "result": result}
            question = self.explorer.should_ask_oracle(obs, self.blackboard)
            if question:
                print(f"    [{self.agent_id}] → oráculo: {question[:80]}...",
                      flush=True)
                answer = self.oracle.ask(question)
                self.blackboard.record_oracle(question, answer)
                print(f"    [{self.agent_id}] ← oráculo: {answer[:100]}...",
                      flush=True)

                # El agente formula una conjetura basada en la respuesta
                self._maybe_conjecture(question, answer, result)

    def _isolate_and_verify(self, t0: float, t1: float, winding: int) -> list:
        """Aisla ceros en [t0,t1] y verifica σ con findroot. Retorna t-values encontrados."""
        sr = (0.48, 0.52)
        candidates = isolate_zeros(sr, (t0, t1), known_winding=winding)
        found_ts = []
        for sigma_est, t_est in candidates:
            # Evitar llamar findroot si ya tenemos ese cero
            if any(abs(z["t"] - t_est) < 0.1 for z in self.blackboard.all_zeros()):
                continue
            if any(abs(t_est - ft) < 0.1 for ft in found_ts):
                continue
            verified = verify_zero(sigma_est, t_est)
            if "error" not in verified:
                is_new = self.blackboard.add_zero(
                    verified["sigma"], verified["t"],
                    verified["deviation"], verified["on_line"]
                )
                if is_new:
                    marker = "✓" if verified["on_line"] else "⚠ FUERA DE LÍNEA"
                    print(f"    [{self.agent_id}] cero: σ={verified['sigma']:.6f} "
                          f"t={verified['t']:.4f}  desv={verified['deviation']:.2e} {marker}",
                          flush=True)
                    found_ts.append(verified["t"])
        return found_ts

    def _maybe_conjecture(self, question: str, oracle_answer: str, result: dict):
        """
        El agente formula una conjetura propia basada en lo que encontró
        y lo que el oráculo contextualizó. La conjetura es del agente.
        """
        zeros = self.blackboard.all_zeros()
        off_line = [z for z in zeros if not z["on_line"]]

        if off_line:
            z = off_line[-1]
            statement = (f"Cero en σ={z['sigma']:.6f} t={z['t']:.2f} "
                         f"(desv={z['deviation']:.2e}) — posible contraejemplo a RH")
            self.blackboard.add_conjecture(statement, evidence=str(result),
                                           confidence=0.3)
        elif result.get("gue_ratio") and result["gue_ratio"] > 1.5:
            statement = (f"Espaciado de ceros con ratio GUE={result['gue_ratio']:.3f} "
                         f"— desviación de Montgomery-Odlyzko en t~{result.get('t_est','?')}")
            self.blackboard.add_conjecture(statement, evidence=oracle_answer,
                                           confidence=0.4)

    def _finish(self) -> dict:
        elapsed = round(time.time() - self._t_start, 1)
        state   = self.blackboard.to_epistemic_state()
        zeros   = self.blackboard.all_zeros()
        off_line = [z for z in zeros if not z["on_line"]]

        # Métricas para que run_v020.py las parsee
        richness = min(1.0, len(zeros) / max(1, self._cycles * 0.5))
        resultado = "convergent" if len(zeros) > 0 and not off_line else "divergent"

        print(f"  [Fin] ciclos={self._cycles}  "
              f"ceros={len(zeros)}  frontera_t={self.blackboard.frontier_t():.1f}  "
              f"off_line={len(off_line)}  oracle_calls={self.oracle.calls}  "
              f"resultado={resultado}  richness={richness:.3f}  "
              f"t={elapsed}s", flush=True)

        # Guardar epistemic state
        out = Path("results") / f"epistemic_state_{self.problem}_{self.agent_id}.json"
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)

        return state


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="agente vX v0.6 — exploración autónoma")
    parser.add_argument("--problem",      choices=["riemann", "pnp", "causal"], default="riemann")
    parser.add_argument("--max-hours",    type=float, default=0.5)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--agent-id",     default="solo")
    parser.add_argument("--cold-start",   action="store_true")
    parser.add_argument("--corpus-path",  default=None)
    parser.add_argument("--oracle-model", default=None)
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")

    corpus_context = None
    if args.corpus_path and Path(args.corpus_path).exists():
        with open(args.corpus_path, encoding="utf-8") as f:
            corpus_context = f.read(3000)

    agent = AgentV013(
        agent_id=args.agent_id,
        problem=args.problem,
        seed=args.seed,
        api_key=api_key,
        corpus_context=corpus_context,
    )
    agent.run(max_hours=args.max_hours)
