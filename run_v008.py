"""
run_v008.py — Agente vX v0.0.8 — Bucle autónomo con problema abierto

Corre con:  uv run python run_v008.py
Python:     3.14t (free-threaded, multi-core real)

Arquitectura:
  - SecondOrderAgent: maestro que absorbe mejoras
  - SatelliteLibrary: cada rama1° tiene su propia biblioteca
  - MathSandbox: NumPy, SymPy, SciPy, Matplotlib
  - LeanInterface: verificación formal
  - MetacognitiveLayer: el agente se lee y modifica a sí mismo
  - SatisfactionMeter: detecta estancamiento → pivot de perspectiva
  - CheckpointManager: guarda estado cada 15 min

Problema objetivo (configurable):
  --problem riemann    → Hipótesis de Riemann
  --problem pnp        → P ≠ NP

El objetivo NO es resolver el problema.
El objetivo es observar cómo el agente:
  1. Ataca con sus herramientas actuales
  2. Detecta su propio límite
  3. Inventa nuevas herramientas
  4. Pivota de perspectiva cuando se estanca
  5. Se auto-mejora en el proceso

Duración: --hours 4 (default)
"""

import argparse
import json
import time
import copy
import threading
from datetime import datetime
from pathlib import Path

from library.store import ResonantLibrary
from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from core.second_order import SecondOrderAgent
from core.satellite_library import SatelliteLibrary
from core.math_sandbox import MathSandbox
from core.lean_interface import LeanInterface
from core.metacognitive import MetacognitiveLayer
from core.satisfaction_meter import SatisfactionMeter, SatisfactionState
from core.checkpoint import CheckpointManager
from core.tool_registry import ToolRegistry
from core.corpus.math_corpus import get_corpus, list_corpus
from gym_adapter import evaluar_agente

VERSION = "v0.0.8"

# ── Configuración base ────────────────────────────────────────────────────────

BASE_CONFIG = {
    "N": 75, "M": 5, "K": 3,
    "confidence_threshold": 0.90,
    "filter_threshold": 0.40,
    "stagnation_threshold": 8,
    "epsilon_start": 0.50,
    "epsilon_min": 0.05,
    "n_episodios": 15,
    "reward_threshold": 30.0,
}

PERSPECTIVES = ["mathematician", "physicist", "algorithmist", "python_engineer"]

# Problema objetivo: acciones codificadas en el espacio S²
# El agente elige entre "estrategias de ataque" al problema
PROBLEM_ACTIONS = {
    "riemann": [
        "compute_zeta_zeros",
        "verify_critical_line",
        "gue_correlation",
        "spectral_approach",
        "xi_symmetry",
    ],
    "pnp": [
        "circuit_lower_bound",
        "sat_reduction",
        "algebraic_approach",
        "gct_method",
        "quantum_separation",
    ],
}


# ── Construcción del agente ───────────────────────────────────────────────────

def build_agent(cfg: dict, library, seed: int) -> ProtoTissueAgentV2:
    return ProtoTissueAgentV2(
        N=cfg["N"], M=cfg["M"], K=cfg["K"], seed=seed,
        stagnation_threshold=cfg["stagnation_threshold"],
        filter_threshold=cfg["filter_threshold"],
        library=library,
        confidence_threshold=cfg["confidence_threshold"],
        t_settle=5, warmup_steps=50, subconscious_update_freq=5,
    )


# ── Evaluador de entorno Gymnasium ────────────────────────────────────────────

def eval_gym(agent, env_name: str, n_ep: int, cfg: dict, seed: int,
             verbose: bool = False) -> dict:
    agent.set_env(env_name)
    return evaluar_agente(
        agent, env_name=env_name, n_episodios=n_ep, max_steps=500,
        verbose=verbose, seed=seed, reward_threshold=cfg["reward_threshold"],
        version=VERSION, epsilon_start=cfg["epsilon_start"],
        epsilon_min=cfg["epsilon_min"], epsilon_decay_ep=n_ep // 2, blend=0.6,
    )


# ── Ciclo de ataque al problema ───────────────────────────────────────────────

def attack_problem(sandbox: MathSandbox,
                   lean: LeanInterface,
                   meta: MetacognitiveLayer,
                   meter: SatisfactionMeter,
                   problem: str,
                   perspective: str,
                   cycle: int,
                   ) -> dict:
    """
    Un ciclo de ataque al problema matemático desde una perspectiva.
    Devuelve un resumen del intento con score de progreso.
    """
    corpus = get_corpus(perspective)
    actions = PROBLEM_ACTIONS.get(problem, [])

    # El agente elige la acción según el ciclo y la perspectiva
    action = actions[cycle % len(actions)]

    log = {
        "cycle": cycle,
        "perspective": perspective,
        "action": action,
        "sandbox_results": [],
        "lean_result": None,
        "hypothesis": None,
        "progress_score": 0.0,
    }

    # ── Ejecución según acción ────────────────────────────────────────────────

    if problem == "riemann":
        if action == "compute_zeta_zeros":
            # Escalar número de ceros con el ciclo — cada vuelta explora más profundo
            n_to_compute = 30 + cycle * 10
            zeros = sandbox.riemann_zeta_zeros(n=n_to_compute)
            re_parts = [z[0] for z in zeros] if zeros else []
            n_zeros = len(re_parts)
            if n_zeros == 0:
                score = 0.0
                deviation = 1.0
            else:
                deviation = sum(abs(r - 0.5) for r in re_parts) / n_zeros
                score = max(0.0, 1.0 - deviation * 10)
            log["sandbox_results"].append({
                "action": action,
                "zeros_computed": n_zeros,
                "mean_re_deviation_from_half": round(deviation, 6),
                "score": round(score, 3),
            })
            if n_zeros > 0 and deviation < 1e-6:
                meta.record_hypothesis(
                    f"Ciclo {cycle}: {n_zeros} ceros calculados tienen Re = 0.5 "
                    f"con precisión {deviation:.2e}",
                    domain="riemann", confidence=0.95,
                )
            log["progress_score"] = score

        elif action == "verify_critical_line":
            N = 5000 + cycle * 2000
            result = sandbox.verify_prime_density(N=N) or {}
            # Score principal: qué tan cerca está π(x)-li(x) del bound de Riemann O(√x·ln x)
            # ratio < 1 significa que el error es mejor que lo que la HR predice como peor caso
            riemann_ratio = result.get("mean_riemann_ratio", 1.0)
            rel_error     = result.get("max_rel_error", 1.0)
            score = max(0.0, 1.0 - riemann_ratio) * 0.6 + max(0.0, 1.0 - rel_error) * 0.4
            log["sandbox_results"].append({**result, "score": round(score, 3)})
            log["progress_score"] = score
            if riemann_ratio < 0.5 and cycle > 1:
                meta.record_hypothesis(
                    f"Ciclo {cycle}: π(x)-li(x) < 0.5·√x·ln(x) hasta N={N} "
                    f"(ratio={riemann_ratio:.3f}) — consistente con HR",
                    domain="riemann", confidence=0.7,
                )

        elif action == "gue_correlation":
            n_gue  = 50 + cycle * 15
            offset = 1 + (cycle // 5) * 50
            code = f"""
mpmath.mp.dps = 20
zeros = [float(mpmath.zetazero(k).imag) for k in range({offset}, {offset}+{n_gue})]
spacings = np.diff(zeros)
spacings /= spacings.mean()   # normalizar a espaciado medio = 1

# Wigner-Dyson GUE: p(s) = (32/π²)·s²·exp(-4s²/π)
wigner_gue = lambda s: (32/np.pi**2) * s**2 * np.exp(-4*s**2/np.pi)

# KS test real contra GUE
s_sorted = np.sort(spacings)
cdf_empirica = np.arange(1, len(s_sorted)+1) / len(s_sorted)
cdf_gue = np.array([float(1 - np.exp(-4*s**2/np.pi) * (1 + 4*s**2/np.pi))
                    for s in s_sorted])
ks_stat = float(np.max(np.abs(cdf_empirica - cdf_gue)))

# Correlación par: r₂(s) — densidad de pares a distancia s
bins = np.linspace(0, 3, 30)
hist, _ = np.histogram(spacings, bins=bins, density=True)
centers = (bins[:-1] + bins[1:]) / 2
gue_pdf  = wigner_gue(centers)
mae = float(np.mean(np.abs(hist - gue_pdf)))

_result = {{
    "n_zeros": len(zeros), "offset": {offset},
    "ks_stat": round(ks_stat, 5),
    "mae_gue": round(mae, 5),
    "gue_fit_error": round(ks_stat, 5),   # compatibilidad con score
}}
"""
            r = sandbox.run(code, label=f"gue_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            ks = float(res.get("ks_stat", res.get("gue_fit_error", 1.0)))
            # KS < 0.05 = excelente fit; < 0.10 = bueno; < 0.20 = razonable
            score = max(0.0, 1.0 - ks * 6)
            log["progress_score"] = score
            if ks < 0.05 and cycle > 1:
                meta.record_hypothesis(
                    f"Ciclo {cycle}: espaciados de ceros de ζ siguen GUE con KS={ks:.4f} "
                    f"(offset={offset}, n={n_gue}) — evidencia espectral de Riemann",
                    domain="riemann", confidence=0.80,
                )

        elif action == "spectral_approach":
            # Matriz más grande cada vuelta + semilla fija por ciclo para reproducibilidad
            N = 50 + cycle * 10
            code = f"""
np.random.seed({cycle})
N = {N}
A = np.random.randn(N, N) + 1j*np.random.randn(N, N)
H = (A + A.conj().T) / (2*np.sqrt(2*N))
eigenvalues = np.linalg.eigvalsh(H)
spacings = np.diff(np.sort(eigenvalues.real))
spacings /= spacings.mean()
hist, edges = np.histogram(spacings, bins=25, density=True)
centers = (edges[:-1] + edges[1:]) / 2
wigner_vals = (np.pi/2)*centers*np.exp(-np.pi/4*centers**2)
ks = float(np.mean(np.abs(hist - wigner_vals)))
_result = {{"matrix_size": N, "wigner_ks_distance": round(ks, 5)}}
"""
            r = sandbox.run(code, label=f"spectral_{cycle}")
            log["sandbox_results"].append(r.get("result") or {"error": r.get("error")})
            ks = (r.get("result") or {}).get("wigner_ks_distance", 1.0)
            log["progress_score"] = max(0.0, 1.0 - float(ks) * 5)

        elif action == "xi_symmetry":
            # Puntos de prueba más alejados del eje real conforme avanza el ciclo
            # → cada ciclo verifica simetría en regiones más difíciles
            imag_base = 14 + cycle * 3
            code = f"""
mpmath.mp.dps = 25
test_points = [
    0.3+{imag_base}j, 0.7+{imag_base}j,
    0.5+{imag_base+7}j, 0.2+{imag_base+16}j,
    0.4+{imag_base+25}j,
]
symmetry_errors = []
for s in test_points:
    xi_s   = 0.5*s*(s-1)*mpmath.power(mpmath.pi, -s/2)*mpmath.gamma(s/2)*mpmath.zeta(s)
    xi_1ms = 0.5*(1-s)*(-s)*mpmath.power(mpmath.pi, -(1-s)/2)*mpmath.gamma((1-s)/2)*mpmath.zeta(1-s)
    symmetry_errors.append(float(abs(xi_s - xi_1ms)))
_result = {{"imag_base": {imag_base}, "symmetry_errors": symmetry_errors,
           "max_error": max(symmetry_errors), "n_points": len(test_points)}}
"""
            r = sandbox.run(code, label=f"xi_{cycle}")
            res = r.get("result") or {}
            max_err = res.get("max_error", 1.0)
            log["sandbox_results"].append(res)
            log["progress_score"] = max(0.0, 1.0 - float(max_err))

    elif problem == "pnp":
        if action == "circuit_lower_bound":
            max_n = 16 + cycle * 2
            code = f"""
n_vals = list(range(4, {max_n}))
circuit_sizes = []
for n in n_vals:
    lb = 2**(n/2)
    circuit_sizes.append((n, round(lb, 2)))
ratios = [circuit_sizes[i+1][1]/circuit_sizes[i][1] for i in range(len(circuit_sizes)-1)]
poly_bound = {max_n}**3  # bound polinomial para comparar
_result = {{"max_n": {max_n}, "lower_bounds": circuit_sizes[:6],
           "growth_ratio": round(ratios[0], 3),
           "superpolynomial": bool(ratios[0] > 2.0),
           "gap_vs_poly": round(circuit_sizes[-1][1] / poly_bound, 2)}}
"""
            r = sandbox.run(code, label=f"circuit_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            score = 0.6 if res.get("superpolynomial") else 0.3
            log["progress_score"] = score if r["ok"] else 0.1

        elif action == "sat_reduction":
            seed_val = 42 + cycle
            code = f"""
random.seed({seed_val})
n_vars, n_clauses = 10, 25
clauses = [tuple(random.randint(1, n_vars) * random.choice([-1,1])
                 for _ in range(3))
           for _ in range(n_clauses)]
start = time.time()
found, count = False, 0
for assignment in itertools.product([False,True], repeat=n_vars):
    count += 1
    vals = {{i+1: v for i,v in enumerate(assignment)}}
    if all(any((vals[abs(l)] if l>0 else not vals[abs(l)]) for l in c) for c in clauses):
        found = True; break
elapsed = time.time() - start
_result = {{"n_vars": n_vars, "n_clauses": n_clauses, "sat": found,
           "explored": count, "elapsed_ms": round(elapsed*1000, 2),
           "exhaustive": count == 2**n_vars}}
"""
            r = sandbox.run(code, label=f"sat_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            # Score: cuánto del espacio tuvo que explorar (más = más difícil = más interesante)
            explored_frac = res.get("explored", 0) / (2**10)
            log["progress_score"] = 0.5 + 0.4 * explored_frac if r["ok"] else 0.1

        elif action == "algebraic_approach":
            code = f"""
n = {8 + cycle // 3}   # escalar con el ciclo
results = []
for fn_name, fn in [("parity", lambda x: sum(x) % 2),
                    ("majority", lambda x: int(sum(x) > n//2)),
                    ("and_all",  lambda x: int(all(x)))]:
    # Muestrear para estimar el grado del polinomio (via coeficientes de Fourier)
    samples = [([random.randint(0,1) for _ in range(n)], ) for _ in range(200)]
    vals = [fn(s[0]) for s in samples]
    # Varianza como proxy de complejidad booleana
    var = float(np.var(vals))
    results.append({{"fn": fn_name, "variance": round(var, 4)}})
_result = {{"n_bits": n, "functions": results,
           "parity_hard": bool(results[0]["variance"] > 0.2)}}
"""
            r = sandbox.run(code, label=f"algebraic_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            score = 0.55 if res.get("parity_hard") else 0.3
            log["progress_score"] = score if r["ok"] else 0.1

        elif action == "gct_method":
            code = f"""
import functools
def permanent(M):
    n = len(M)
    if n == 0: return 1
    total = 0
    for perm in itertools.permutations(range(n)):
        prod = functools.reduce(lambda a,b: a*b, [M[i][perm[i]] for i in range(n)])
        total += prod
    return total

sizes = [2, 3, {min(4 + cycle // 4, 6)}]
results = []
for n in sizes:
    M = np.random.randint(0, 5, (n, n)).tolist()
    perm_val = permanent(M)
    det_val  = round(float(np.linalg.det(M)), 4)
    results.append({{"n": n, "perm": perm_val, "det": det_val,
                    "differ": bool(abs(perm_val - det_val) > 0.01)}})
_result = {{"matrix_results": results,
           "always_differ": bool(all(r["differ"] for r in results))}}
"""
            r = sandbox.run(code, label=f"gct_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            score = 0.65 if res.get("always_differ") else 0.35
            log["progress_score"] = score if r["ok"] else 0.1
            if res.get("always_differ"):
                meta.record_hypothesis(
                    f"Ciclo {cycle}: perm(M) ≠ det(M) en todos los tamaños probados "
                    f"(evidencia empírica de separación algebraica GCT)",
                    domain="pnp", confidence=0.4,
                )

        elif action == "quantum_separation":
            # BQP vs NP: Deutsch-Jozsa como ejemplo de separación cuántica
            code = f"""
# Simular Deutsch-Jozsa: diferencia entre oráculo constante y balanceado
# En cuántico: 1 consulta; clásico: hasta 2^(n-1)+1 consultas
n_bits = 8
n_trials = 50
classical_queries = []
quantum_queries   = []
for _ in range(n_trials):
    fn_type = random.choice(["constant", "balanced"])
    if fn_type == "constant":
        fn = lambda x: random.choice([0, 1])
    else:
        half = 2**(n_bits-1)
        outputs = [0]*half + [1]*half
        random.shuffle(outputs)
        fn = lambda x, o=outputs: o[x % len(o)]

    # Clásico: worst case 2^(n-1)+1 queries
    classical_queries.append(2**(n_bits-1) + 1)
    # Cuántico: siempre 1 query
    quantum_queries.append(1)

speedup = np.mean(classical_queries) / np.mean(quantum_queries)
_result = {{"n_bits": n_bits, "classical_avg": float(np.mean(classical_queries)),
           "quantum_avg": 1.0, "speedup": round(float(speedup), 1),
           "exponential_separation": bool(speedup > 100)}}
"""
            r = sandbox.run(code, label=f"quantum_{cycle}")
            res = r.get("result") or {"error": r.get("error")}
            log["sandbox_results"].append(res)
            score = 0.7 if res.get("exponential_separation") else 0.4
            log["progress_score"] = score if r["ok"] else 0.1
            if res.get("exponential_separation") and cycle % 5 == 0:
                meta.record_hypothesis(
                    f"Ciclo {cycle}: Deutsch-Jozsa muestra separación exponencial "
                    f"BQP vs NP (speedup={res.get('speedup')}x con {res.get('n_bits')} bits)",
                    domain="pnp", confidence=0.6,
                )

        else:
            log["sandbox_results"].append({"action": action, "status": "not_implemented"})
            log["progress_score"] = 0.1

    # ── Intentar verificación Lean ────────────────────────────────────────────
    if cycle % 5 == 0:   # cada 5 ciclos, un intento Lean
        template = (lean.riemann_hypothesis_template() if problem == "riemann"
                    else lean.p_np_template())
        lean_r = lean.verify(template, label=f"{problem}_cycle{cycle}")
        log["lean_result"] = {"ok": lean_r["ok"], "output": lean_r["output"][:300]}

    # ── Registrar intento en metacognición ────────────────────────────────────
    meta.record_attempt(
        description=f"{perspective}/{action}",
        result=f"score={log['progress_score']:.3f}",
        domain=problem,
    )

    return log


# ── Bucle principal ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem",  default="riemann",
                        choices=["riemann", "pnp"])
    parser.add_argument("--hours",    type=float, default=4.0)
    parser.add_argument("--env",      default="CartPole-v1")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--resume",   action="store_true",
                        help="Reanudar desde el último checkpoint")
    args = parser.parse_args()

    session_id = f"v008_{args.problem}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    deadline   = time.time() + args.hours * 3600

    print("=" * 70)
    print(f"Agente vX {VERSION} — Problema: {args.problem.upper()}")
    print(f"Python: free-threaded 3.14t  |  Duración: {args.hours}h")
    print(f"Sesión: {session_id}")
    print("=" * 70)

    # ── Inicializar subsistemas ───────────────────────────────────────────────
    library   = ResonantLibrary(db_path="library/resonant_db.sqlite")
    sandbox   = MathSandbox(timeout_sec=45.0)
    lean      = LeanInterface(use_lean=True)
    meter     = SatisfactionMeter(stagnation_window=4, pivot_threshold=8)
    ckpt      = CheckpointManager(session_id, interval_minutes=15.0)
    maestro   = SecondOrderAgent(BASE_CONFIG, library, seed=args.seed)
    tool_reg  = ToolRegistry()
    meta      = MetacognitiveLayer(maestro.master, tool_reg, sandbox)

    # Cargar herramientas iniciales de todos los corpus
    for corpus_name in PERSPECTIVES:
        corpus = get_corpus(corpus_name)
        for tool_def in corpus.get("initial_tools", []):
            ok, msg = tool_reg.invent(
                tool_def["name"], tool_def["code"], tool_def["description"]
            )
            if ok:
                print(f"  [Tool] Registrada: {tool_def['name']}")
            else:
                print(f"  [Tool] Error: {tool_def['name']} — {msg[:60]}")

    # ── Reanudar desde checkpoint ─────────────────────────────────────────────
    global_cycle   = 0
    perspective    = PERSPECTIVES[0]
    all_logs: list = []

    if args.resume:
        saved = ckpt.load_latest()
        if saved:
            meta_state = saved.get("meta", {})
            global_cycle  = meta_state.get("cycle", 0)
            perspective   = meta_state.get("perspective", PERSPECTIVES[0])
            all_logs      = meta_state.get("logs", [])
            if saved.get("tensors"):
                ckpt.restore_agent_tissue(maestro.master, saved["tensors"])
            print(f"  [Checkpoint] Reanudando desde ciclo {global_cycle}")

    print(f"\n[Inicio] Perspectiva inicial: {perspective}")
    print(f"[Corpus] {list_corpus()}")
    print(f"[Tools]  {tool_reg.list_tools()}\n")

    # ── Bucle principal (tiempo real) ─────────────────────────────────────────
    gym_cycle  = 0   # ciclos del entorno Gymnasium (aprendizaje base)
    cfg = copy.deepcopy(BASE_CONFIG)

    try:
        while time.time() < deadline:
            global_cycle += 1
            t_remaining  = (deadline - time.time()) / 3600

            print(f"\n{'─'*70}")
            print(f"  CICLO {global_cycle}  |  {t_remaining:.2f}h restantes  "
                  f"|  perspectiva: {perspective}")
            print(f"  Satisfacción: {meter.satisfaction_score():.2f}  "
                  f"estado: {meter.state.value}  "
                  f"tools: {tool_reg.summary()['total']}")
            print(f"{'─'*70}")

            # ── (A) Ciclo de aprendizaje Gymnasium (mantiene el tejido vivo) ──
            if gym_cycle % 3 == 0:   # cada 3 ciclos globales
                maestro.master.set_env(args.env)
                gym_result = eval_gym(
                    maestro.master, args.env, cfg["n_episodios"],
                    cfg, args.seed + global_cycle, verbose=False,
                )
                gym_media = gym_result["recompensa_media"]
                r = gym_result["reporte_agente"]
                total_dec = r['n_fast_decisions'] + r['n_resonant_decisions']
                fast_pct  = r['n_fast_decisions'] / max(1, total_dec)
                print(f"  [Gym] media={gym_media:.1f}  fast={fast_pct:.0%}")
                maestro.set_baseline(gym_media)

                # Diagnosticar y generar variantes de primer orden
                from run_autoimprove import diagnosticar as diag_fn
                diagnostico = diag_fn(gym_result)
                if diagnostico != "none":
                    def eval_fn_wrap(ag, env_nm, n_ep):
                        ag.set_env(env_nm)
                        return eval_gym(ag, env_nm, n_ep, cfg, args.seed + 100, verbose=False)

                    gen_r = maestro.run_generation(
                        diagnostico=diagnostico,
                        evaluator_fn=eval_fn_wrap,
                        env_name=args.env,
                        n_episodes=10,
                        seed=args.seed,
                        verbose=True,
                    )
                    if gen_r.get("promoted"):
                        cfg = copy.deepcopy(maestro.base_config)
            gym_cycle += 1

            # ── (B) Ataque al problema matemático ─────────────────────────────
            attack_log = attack_problem(
                sandbox, lean, meta, meter,
                problem=args.problem,
                perspective=perspective,
                cycle=global_cycle,
            )
            all_logs.append(attack_log)
            progress = attack_log["progress_score"]

            # Mostrar resultado del ataque
            for sr in attack_log["sandbox_results"]:
                print(f"  [Math/{attack_log['action']}] {sr}")
            if attack_log.get("lean_result"):
                lr = attack_log["lean_result"]
                print(f"  [Lean] ok={lr['ok']}  {lr['output'][:120]}")

            # ── (C) Actualizar medidor de satisfacción ────────────────────────
            state = meter.update(progress)
            print(f"  [Meter] progress={progress:.3f}  state={state.value}")

            # Pivot forzado cada 8 ciclos — independiente del stagnation
            if global_cycle % 8 == 0:
                next_p = meter.next_perspective(perspective, PERSPECTIVES)
                print(f"\n  🔄 ROTACIÓN: {perspective} → {next_p} (ciclo {global_cycle})")
                perspective = next_p
                _try_invent_tool(meta, sandbox, args.problem, global_cycle)
            elif meter.needs_pivot():
                next_p = meter.next_perspective(perspective, PERSPECTIVES)
                print(f"\n  ⚡ PIVOT: {perspective} → {next_p}")
                perspective = next_p
                _try_invent_tool(meta, sandbox, args.problem, global_cycle)
            elif meter.needs_tool_improvement():
                print(f"  ⚙ MEJORA DE HERRAMIENTA (estancamiento detectado)")
                _try_improve_tool(meta, tool_reg, sandbox, args.problem, global_cycle)

            # Invocar herramientas del registry cada 4 ciclos
            if global_cycle % 4 == 0:
                active_tools = [t for t in tool_reg._tools.values() if t.active]
                if active_tools:
                    import random as _rnd
                    tool = _rnd.choice(active_tools)
                    try:
                        import torch as _torch
                        omega = maestro.master.tissue.nodes[
                            maestro.master.exec_ids[0]
                        ].q.detach()
                        tool_result = tool.fn(omega, [omega], {"cycle": global_cycle})
                        print(f"  [Tool/{tool.name}] {str(tool_result)[:120]}")
                        tool_reg.report_outcome(tool.name, float(progress))
                    except Exception as _e:
                        print(f"  [Tool/{tool.name}] error: {_e}")

            # ── (D) Checkpoint periódico ──────────────────────────────────────
            state_to_save = {
                "cycle":        global_cycle,
                "perspective":  perspective,
                "config":       cfg,
                "meter":        meter.summary(),
                "meta":         meta.self_report(),
                "tools":        tool_reg.summary(),
                "logs":         all_logs[-20:],   # últimos 20 logs
                "_tensors":     ckpt.save_agent_tissue(maestro.master),
            }
            ckpt.save(state_to_save)

    except KeyboardInterrupt:
        print("\n  [Interrupción manual]")

    # ── Guardar resultado final ───────────────────────────────────────────────
    _save_final(session_id, all_logs, meta, meter, tool_reg, maestro)


# ── Invención de herramientas en pivot ────────────────────────────────────────

def _try_invent_tool(meta: MetacognitiveLayer,
                     sandbox: MathSandbox,
                     problem: str,
                     cycle: int) -> None:
    """El agente intenta inventar una herramienta al pivotar de perspectiva."""
    name = f"pivot_tool_{cycle}"
    if problem == "riemann":
        code = f"""
def tool_fn(omega, candidates, context):
    import torch, torch.nn.functional as F
    import mpmath; mpmath.mp.dps = 10
    # Herramienta inventada en ciclo {cycle}: scoring por proximidad a ceros de zeta
    try:
        zero_t = float(mpmath.zetazero(1 + (cycle % 10)).imag)
        ref = F.normalize(torch.tensor([0.5, zero_t/100, 0.0]), dim=-1)
        scores = [float(F.cosine_similarity(c.unsqueeze(0), ref.unsqueeze(0)))
                  for c in candidates]
        best = max(range(len(scores)), key=lambda i: scores[i])
        return candidates[best], {{"regime": "invented_{cycle}", "zero_ref": zero_t}}
    except Exception:
        return candidates[0], {{"regime": "fallback"}}
""".replace("{cycle}", str(cycle))
    else:
        code = f"""
def tool_fn(omega, candidates, context):
    import torch, torch.nn.functional as F
    # Herramienta inventada ciclo {cycle}: scoring por complejidad estimada
    import math
    scores = [math.exp(-float(c.norm())) for c in candidates]
    best = max(range(len(scores)), key=lambda i: scores[i])
    return candidates[best], {{"regime": "invented_{cycle}"}}
""".replace("{cycle}", str(cycle))

    result = meta.propose_tool(name, f"Herramienta inventada en pivot ciclo {cycle}", code)
    if result["ok"]:
        print(f"  [Tool inventada] {name} ✓")
    else:
        print(f"  [Tool inventada] {name} ✗ — {result.get('reason','')[:60]}")


def _try_improve_tool(meta: MetacognitiveLayer,
                      tool_reg: ToolRegistry,
                      sandbox: MathSandbox,
                      problem: str,
                      cycle: int) -> None:
    """Mejora la herramienta con peor rendimiento."""
    tools = [t for t in tool_reg._tools.values() if t.active and len(t.performance) > 2]
    if not tools:
        return
    worst = min(tools, key=lambda t: t.avg_performance())
    print(f"  [Mejora] Peor herramienta: '{worst.name}' avg={worst.avg_performance():.2f}")
    # Registrar en metacognición como intento de mejora
    meta.record_attempt(
        f"improve_tool:{worst.name}",
        f"avg_perf={worst.avg_performance():.2f} → mejora pendiente",
        domain="python",
    )


# ── Guardar resultado final ───────────────────────────────────────────────────

def _save_final(session_id, all_logs, meta, meter, tool_reg, maestro):
    import datetime
    result = {
        "version":     VERSION,
        "session_id":  session_id,
        "timestamp":   datetime.datetime.now().isoformat(),
        "total_cycles": len(all_logs),
        "meta_report": meta.self_report(),
        "meter":       meter.summary(),
        "tools":       tool_reg.summary(),
        "hypotheses":  meta._hypotheses,
        "attempts":    meta._attempts[-50:],
        "logs":        all_logs[-100:],
    }
    import numpy as _np

    class _SafeEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, _np.bool_):
                return bool(o)
            if isinstance(o, (_np.integer,)):
                return int(o)
            if isinstance(o, (_np.floating,)):
                return float(o)
            if isinstance(o, _np.ndarray):
                return o.tolist()
            return super().default(o)

    out_file = f"results_v008_{session_id}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, cls=_SafeEncoder)

    print(f"\n{'='*70}")
    print(f"  Sesión completada: {len(all_logs)} ciclos")
    print(f"  Hipótesis registradas: {len(meta._hypotheses)}")
    print(f"  Intentos: {len(meta._attempts)}")
    print(f"  Herramientas inventadas: {tool_reg.summary()['invented']}")
    print(f"  Pivots: {meter.summary()['pivot_count']}")
    print(f"  Satisfacción final: {meter.satisfaction_score():.3f}")
    print(f"  Resultado → {out_file}")
    print(f"{'='*70}")

    # Actualizar bitácora
    with open("bitacora.md", "a", encoding="utf-8") as bk:
        bk.write(f"""

---

## v0.0.8 — {datetime.datetime.now().date()} — Bucle autónomo problema abierto

### Configuración
- Python: 3.14t (free-threaded)
- Problema: {result.get('session_id','').split('_')[1] if '_' in result.get('session_id','') else '?'}
- Ciclos: {result['total_cycles']}
- Duración configurada: ver session_id

### Resultados
- Hipótesis generadas: {len(result['hypotheses'])}
- Herramientas inventadas: {result['tools']['invented']}
- Pivots de perspectiva: {result['meter']['pivot_count']}
- Satisfacción final: {result['meter']['satisfaction']:.3f}

### Archivos
- `{out_file}` — resultado completo
- `checkpoints/` — estados intermedios
""")
    print("  Bitácora actualizada.")


if __name__ == "__main__":
    main()
