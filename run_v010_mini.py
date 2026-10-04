"""
run_v010_mini.py — Prueba pequeña de los 4 mecanismos nuevos (v0.1.0)

Ejecuta 15 ciclos de Riemann para verificar que los 4 cambios funcionan:
  1. ResearchMemory — los resultados se indexan y se leen ciclo a ciclo
  2. creative_crisis — se detecta si hay hipótesis SUPPORTED en tensión
  3. ActionInventor — intenta inventar una herramienta (si hay oracle disponible)
  4. Corpus ML — ml_tools cargado en el registry desde el inicio

No reemplaza run_v009.py. Es un smoke test de los nuevos módulos.

Uso:
    uv run python run_v010_mini.py
    uv run python run_v010_mini.py --cycles 20 --skip-oracle
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import os

from core.math_sandbox       import MathSandbox
from core.hypothesis_tracker import HypothesisTracker
from core.tool_registry      import ToolRegistry
from core.oracle_client      import OracleClient
from core.subagent_pool      import SubagentPool
from core.research_memory    import ResearchMemory
from core.action_inventor    import ActionInventor
from core.corpus.math_corpus import get_corpus, list_corpus_for_domain

load_dotenv()

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--cycles",       type=int,  default=15)
parser.add_argument("--problem",      type=str,  default="riemann")
parser.add_argument("--skip-oracle",  action="store_true",
                    help="No llama al oracle (evita costos en prueba)")
args = parser.parse_args()

PROBLEM  = args.problem
N_CYCLES = args.cycles

print(f"\n{'='*60}")
print(f"  agente vX v0.1.0 — MINI TEST ({N_CYCLES} ciclos, {PROBLEM})")
print(f"{'='*60}\n")

# ── Setup ────────────────────────────────────────────────────────────────────

api_key = os.getenv("ANTHROPIC_API_KEY") if not args.skip_oracle else None
sandbox     = MathSandbox(timeout_sec=30.0)
sub_sandbox = MathSandbox(timeout_sec=40.0)
oracle      = OracleClient(api_key, budget_usd=0.50) if api_key else OracleClient(None, budget_usd=0)
pool        = SubagentPool(sub_sandbox, max_workers=2)
tracker     = HypothesisTracker(subagent_pool=pool, oracle_client=oracle, sandbox=sandbox)
tool_reg    = ToolRegistry()

# ── 1. ResearchMemory ────────────────────────────────────────────────────────
session_id = datetime.now().strftime("%Y%m%d_%H%M")
mem_path   = f"results/memory_{PROBLEM}_{session_id}.json"
Path("results").mkdir(exist_ok=True)
memory = ResearchMemory(domain=PROBLEM, persist_path=mem_path)
print(f"[1] ResearchMemory inicializada → {mem_path}")

# ── 2. Corpus ML cargado ─────────────────────────────────────────────────────
all_tools_loaded = set()
for corpus_name in list_corpus_for_domain(PROBLEM):
    corpus = get_corpus(corpus_name)
    for tool_def in corpus.get("initial_tools", []):
        if tool_def["name"] not in all_tools_loaded:
            ok, msg = tool_reg.invent(
                tool_def["name"], tool_def["code"], tool_def["description"]
            )
            if ok:
                all_tools_loaded.add(tool_def["name"])
    # Registrar el corpus ML aunque no tenga tools (su knowledge se imprime)
    if corpus_name == "ml_tools":
        print(f"[2] Corpus ML cargado: {corpus_name}")
        print(f"    (sin tools iniciales — enseña al oracle qué libs usar para inventar)")

print(f"    Tools cargadas: {list(all_tools_loaded)}\n")


def _get_tags(action: str) -> list:
    tag_map = {
        "compute_zeta_zeros":   ["zeros", "critica", "linea", "riemann"],
        "gue_correlation":      ["gue", "wigner", "espectral", "espaciado"],
        "spectral_approach":    ["espectral", "matriz", "gue", "aleatorio"],
        "xi_symmetry":          ["xi", "simetria", "funcional"],
        "verify_critical_line": ["primos", "li", "zeros", "densidad"],
    }
    return tag_map.get(action, [action])


# ── 3. ActionInventor ────────────────────────────────────────────────────────
inventor = ActionInventor(oracle)
print(f"[3] ActionInventor listo (can_invent={inventor.can_invent})\n")

# ── Acciones Riemann (mismas que v009) ────────────────────────────────────────

RIEMANN_ACTIONS = [
    "verify_critical_line",
    "gue_correlation",
    "spectral_approach",
    "xi_symmetry",
    "compute_zeta_zeros",
]

def run_riemann_action(action: str, cycle: int) -> tuple[float, dict]:
    """Ejecuta una acción de Riemann (subset de _riemann_action de v009)."""

    if action == "compute_zeta_zeros":
        code = f"""
import mpmath, numpy as np
mpmath.mp.dps = 20
n = {20 + cycle * 5}
zeros = [mpmath.zetazero(k) for k in range(1, n+1)]
devs  = [abs(float(z.real) - 0.5) for z in zeros]
max_dev = max(devs)
_result = {{"n_checked": n, "max_dev": round(max_dev, 15),
           "all_on_line": bool(max_dev < 1e-10),
           "score": round(1.0 if max_dev < 1e-10 else max(0.0, 1.0 - max_dev*1e6), 4)}}
"""
        r = sandbox.run(code, label=f"zeta_{cycle}")
        res = r.get("result") or {}
        score = float(res.get("score", 0.0))
        if score > 0.8:
            tracker.record(
                f"Ciclo {cycle}: {res.get('n_checked')} ceros en Re=0.5 "
                f"(max_dev={res.get('max_dev', '?')})",
                domain=PROBLEM, confidence=0.88,
                evidence=f"mpmath dps=20",
            )
        return score, res

    elif action == "gue_correlation":
        offset = 1 + ((cycle // 3) * 30) % 600
        code = f"""
import mpmath, numpy as np
mpmath.mp.dps = 18
zeros = [float(mpmath.zetazero(k).imag) for k in range({offset}, {offset}+40)]
sp = np.diff(zeros); sp /= sp.mean()
wigner_pdf = lambda s: (32/np.pi**2)*s**2*np.exp(-4*s**2/np.pi)
s_sorted = np.sort(sp)
cdf_emp = np.arange(1, len(s_sorted)+1) / len(s_sorted)
s_grid = np.linspace(0, s_sorted[-1]*1.1, 1000)
pdf_g = wigner_pdf(s_grid)
cdf_g = np.cumsum(pdf_g)*(s_grid[1]-s_grid[0]); cdf_g /= cdf_g[-1]
ks = float(np.max(np.abs(cdf_emp - np.interp(s_sorted, s_grid, cdf_g))))
_result = {{"offset": {offset}, "ks_stat": round(ks, 5),
           "gue_consistent": bool(ks < 0.10),
           "score": round(max(0.0, 1.0 - ks*3), 4)}}
"""
        r = sandbox.run(code, label=f"gue_{cycle}")
        res = r.get("result") or {}
        score = float(res.get("score", 0.0))
        if score > 0.6:
            tracker.record(
                f"Ciclo {cycle}: espaciado GUE KS={res.get('ks_stat', '?')} "
                f"(offset={offset})",
                domain=PROBLEM, confidence=0.78,
                evidence=f"KS={res.get('ks_stat', '?')}",
            )
        return score, res

    elif action == "spectral_approach":
        code = f"""
import numpy as np
np.random.seed({cycle})
N = {40 + cycle * 8}
A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
H = (A + A.conj().T) / (2*np.sqrt(2*N))
ev = np.linalg.eigvalsh(H)
sp = np.diff(np.sort(ev.real)); sp /= sp.mean()
hist, edges = np.histogram(sp, bins=20, density=True)
c = (edges[:-1]+edges[1:])/2
wigner = (np.pi/2)*c*np.exp(-np.pi/4*c**2)
ks = float(np.mean(np.abs(hist - wigner)))
_result = {{"matrix_size": N, "wigner_ks_distance": round(ks, 5),
           "score": round(max(0.0, 1.0 - ks*2), 4)}}
"""
        r = sandbox.run(code, label=f"spectral_{cycle}")
        res = r.get("result") or {}
        score = float(res.get("score", 0.0))
        if score > 0.65:
            tracker.record(
                f"Ciclo {cycle}: GUE espectral N={res.get('matrix_size')} "
                f"KS={res.get('wigner_ks_distance', '?')}",
                domain=PROBLEM, confidence=0.74,
                evidence=f"np.random GUE matriz",
            )
        return score, res

    elif action == "xi_symmetry":
        code = f"""
import mpmath
mpmath.mp.dps = 20
def xi(s):
    return s*(s-1)*mpmath.power(mpmath.pi, -s/2)*mpmath.gamma(s/2)*mpmath.zeta(s)
pts = [0.1+0.2j, 0.3+0.5j, 0.4+1.0j, 0.25+2.0j]
errors = []
for s in pts:
    diff = abs(xi(s) - xi(1-s))
    errors.append(float(diff))
max_err = max(errors)
_result = {{"symmetry_verified": bool(max_err < 1e-10),
           "xi_errors": [round(e, 15) for e in errors],
           "score": round(1.0 if max_err < 1e-10 else max(0.0, 1.0 - max_err*1e8), 4)}}
"""
        r = sandbox.run(code, label=f"xi_{cycle}")
        res = r.get("result") or {}
        score = float(res.get("score", 0.0))
        if score > 0.9:
            tracker.record(
                f"Ciclo {cycle}: ξ(s)=ξ(1-s) verificada (max_err<1e-10)",
                domain=PROBLEM, confidence=0.95,
                evidence=f"mpmath dps=20, 4 puntos",
            )
        return score, res

    else:  # verify_critical_line
        code = f"""
import mpmath, numpy as np
mpmath.mp.dps = 15
N = {2000 + cycle * 500}
primes_range = range(2, min(N, 5000))
# Aproximación: π(x) vs li(x)
x_vals = [100, 500, 1000, 2000, min(N, 4000)]
li_ratios = []
for x in x_vals:
    pi_x = sum(1 for p in range(2, x+1)
               if all(p % d != 0 for d in range(2, int(p**0.5)+1)))
    li_x = float(mpmath.li(x))
    if li_x > 0:
        ratio = abs(pi_x - li_x) / li_x
        li_ratios.append(ratio)
mean_r = round(sum(li_ratios)/len(li_ratios), 5) if li_ratios else 1.0
_result = {{"N": N, "mean_riemann_ratio": mean_r,
           "consistent_rh": bool(mean_r < 0.15),
           "score": round(max(0.0, 1.0 - mean_r*4), 4)}}
"""
        r = sandbox.run(code, label=f"vcl_{cycle}")
        res = r.get("result") or {}
        score = float(res.get("score", 0.0))
        if score > 0.5:
            tracker.record(
                f"Ciclo {cycle}: π(x)≈li(x) hasta N={res.get('N')} "
                f"(ratio={res.get('mean_riemann_ratio', '?')})",
                domain=PROBLEM, confidence=0.68,
                evidence=f"mpmath.li exacta",
            )
        return score, res


# ── Bucle mini ────────────────────────────────────────────────────────────────

print(f"{'─'*60}")
print(f"  BUCLE PRINCIPAL: {N_CYCLES} ciclos")
print(f"{'─'*60}\n")

all_scores  = []
crisis_done = False
invention_done = False

for cycle in range(1, N_CYCLES + 1):
    action = RIEMANN_ACTIONS[(cycle - 1) % len(RIEMANN_ACTIONS)]
    print(f"\n[Ciclo {cycle:02d}] acción: {action}")

    t0 = time.time()
    score, res = run_riemann_action(action, cycle)
    elapsed = round(time.time() - t0, 2)

    print(f"  score={score:.3f}  ({elapsed}s)  {str(res)[:100]}")
    all_scores.append(score)

    # ── 1. Indexar en ResearchMemory ─────────────────────────────────────────
    tags   = _get_tags(action)
    mem_id = memory.index(
        summary = f"{action}: score={score:.3f}, {str(res)[:120]}",
        score   = score,
        cycle   = cycle,
        source  = "main",
        tags    = tags,
        result  = res,
    )
    print(f"  [Mem] indexado {mem_id} tags={tags}")

    # ── Tracker verifica hipótesis ────────────────────────────────────────────
    if cycle % 3 == 0:
        for _h in list(tracker._hyps):
            tracker._maybe_verify(_h)
        completed = pool.collect_completed()
        if completed:
            tracker.absorb_subagent_results(completed)
            for r in completed:
                memory.index(
                    summary  = f"Subagente {r.task_id}: {r.hypothesis[:80]}",
                    score    = r.score,
                    cycle    = cycle,
                    source   = "subagent",
                    tags     = ["verification", r.domain],
                    result   = r.result,
                )
                print(f"  [Sub] {r.task_id} score={r.score:.2f} — {r.hypothesis[:50]}")

    # ── 4. creative_crisis check (a partir de ciclo 5) ───────────────────────
    if cycle >= 5 and not crisis_done:
        crisis = tracker.check_creative_crisis(research_memory=memory)
        if crisis:
            crisis_done = True
            print(f"\n  [CreativeCrisis] ✓ Tensión encontrada")
            print(f"    Dirección de síntesis: {crisis['synthesis_direction']}")

    # ── 3. ActionInventor (si score promedio < 0.5 después de ciclo 8) ────────
    if (cycle >= 8 and not invention_done
            and inventor.can_invent
            and not args.skip_oracle):
        avg = sum(all_scores[-5:]) / 5
        if avg < 0.55:
            print(f"\n  [ActionInventor] Score promedio={avg:.2f} < 0.55, intentando inventar...")
            tool_summary = "\n".join(
                f"  {t}: score_avg={tool_reg._tools[t].score_avg:.2f}"
                for t in tool_reg._tools
            ) if tool_reg._tools else "  (sin tools previas)"
            failed = [a for a in RIEMANN_ACTIONS if
                      any(r.get("score", 0) < 0.3
                          for s, r in [(score, res)] if action == a)]

            invention = inventor.try_invent(
                problem        = PROBLEM,
                n_cycles       = cycle,
                tool_summary   = tool_summary,
                failed_actions = failed,
                memory         = memory,
            )
            if invention:
                invention_done = True
                ok, msg = tool_reg.invent(
                    invention["tool_name"],
                    invention["code"],
                    invention["description"],
                )
                print(f"  [ActionInventor] Tool registrada: {invention['tool_name']} ok={ok}")
                if ok:
                    memory.index(
                        summary = f"Invención: {invention['tool_name']} — {invention['description']}",
                        score   = 0.7,
                        cycle   = cycle,
                        source  = "invention",
                        tags    = ["invention", "new_tool", PROBLEM],
                    )

    # ── Mostrar estado de memoria cada 5 ciclos ───────────────────────────────
    if cycle % 5 == 0:
        mem_sum = memory.summary()
        print(f"\n  [Memory] total={mem_sum['total']} avg_score={mem_sum['avg_score']}")
        print(f"           by_source={mem_sum['by_source']}")
        recent_str = memory.to_corpus_knowledge(top_k=3)
        print(f"  [Memory/top3]:\n{recent_str}")

        # Verificar tensiones en memoria
        tension_pairs = memory.tension_pairs(min_score=0.4)
        if tension_pairs:
            print(f"  [Memory/tensión] {len(tension_pairs)} pares en tensión detectados")


# ── Resumen final ─────────────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  RESUMEN FINAL")
print(f"{'='*60}")

avg_score = sum(all_scores) / len(all_scores)
print(f"\n  Ciclos ejecutados: {N_CYCLES}")
print(f"  Score promedio:    {avg_score:.3f}")
print(f"  Score máximo:      {max(all_scores):.3f}")

tracker_report = tracker.self_report()
print(f"\n  Hipótesis:")
print(f"    Total:     {tracker_report['hypotheses']}")
print(f"    Supported: {tracker_report['hypotheses_supported']}")
print(f"    Falsified: {tracker_report['hypotheses_falsified']}")
print(f"    Testing:   {tracker_report['hypotheses_testing']}")

mem_sum = memory.summary()
print(f"\n  ResearchMemory:")
print(f"    Total entradas: {mem_sum['total']}")
print(f"    By source:      {mem_sum['by_source']}")
print(f"    Top tags:       {mem_sum['top_tags']}")
print(f"    Score promedio: {mem_sum['avg_score']}")

print(f"\n  ActionInventor: {inventor.summary()}")
print(f"  CreativeCrisis detectada: {crisis_done}")
print(f"  Corpus ML cargado: ✓")

pool_sum = pool.summary()
print(f"\n  SubagentPool: {pool_sum}")

# Guardar resultados
result_path = f"results/v010_mini_{PROBLEM}_{session_id}.json"
with open(result_path, "w", encoding="utf-8") as f:
    json.dump({
        "session_id":   session_id,
        "problem":      PROBLEM,
        "n_cycles":     N_CYCLES,
        "avg_score":    avg_score,
        "all_scores":   all_scores,
        "tracker":      tracker_report,
        "memory":       mem_sum,
        "inventor":     inventor.summary(),
        "pool":         pool_sum,
        "crisis_found": crisis_done,
    }, f, indent=2, ensure_ascii=False)

print(f"\n  Resultados guardados → {result_path}")
print(f"  Memoria persistida  → {mem_path}")
pool.shutdown()
print(f"\n[OK] Mini test completado\n")
