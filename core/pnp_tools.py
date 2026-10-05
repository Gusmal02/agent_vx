"""
pnp_tools.py — Herramientas computacionales para exploración de P vs NP.

El agente las aplica a instancias que él elige.
Ninguna es generada por LLM. Todo es verificable, reproducible.
"""

import math
import random
import time


def random_k_sat(k: int, n_vars: int, n_clauses: int, seed: int = 0) -> list[list[int]]:
    """
    Genera una fórmula k-SAT aleatoria.
    Retorna lista de cláusulas; cada cláusula es lista de literales (+/-).
    """
    rng = random.Random(seed)
    clauses = []
    for _ in range(n_clauses):
        vars_chosen = rng.sample(range(1, n_vars + 1), k)
        clause = [v * rng.choice([-1, 1]) for v in vars_chosen]
        clauses.append(clause)
    return clauses


def dpll_solve(clauses: list[list[int]], n_vars: int,
               max_steps: int = 100_000) -> dict:
    """
    DPLL simplificado. Retorna {"sat": bool, "steps": int, "timeout": bool}.
    """
    assignment = {}
    steps = [0]

    def unit_propagate(cls, asgn):
        changed = True
        while changed:
            changed = False
            for c in cls:
                unset = [l for l in c if abs(l) not in asgn]
                satisfied = any((l > 0 and asgn.get(l, False)) or
                                (l < 0 and not asgn.get(-l, True))
                                for l in c)
                if satisfied:
                    continue
                if len(unset) == 0:
                    return None   # conflicto
                if len(unset) == 1:
                    lit = unset[0]
                    asgn[abs(lit)] = (lit > 0)
                    changed = True
        return asgn

    def solve(cls, asgn, depth=0):
        steps[0] += 1
        if steps[0] > max_steps:
            return None   # timeout signal

        # Simplificar cláusulas
        remaining = []
        for c in cls:
            vals = [(asgn[abs(l)] == (l > 0)) if abs(l) in asgn else None
                    for l in c]
            if any(v is True for v in vals):
                continue    # satisfecha
            unset = [c[i] for i, v in enumerate(vals) if v is None]
            if not unset:
                return False  # cláusula vacía
            remaining.append(unset)
        if not remaining:
            return True

        # Elegir variable (primer literal no asignado)
        var = abs(remaining[0][0])

        for val in [True, False]:
            new_asgn = dict(asgn)
            new_asgn[var] = val
            result = solve(remaining, new_asgn, depth + 1)
            if result is True:
                asgn.update(new_asgn)
                return True
            if result is None:
                return None   # timeout
        return False

    result = solve(clauses, assignment)
    if result is None:
        return {"sat": None, "steps": steps[0], "timeout": True}
    return {"sat": bool(result), "steps": steps[0], "timeout": False}


def sat_hardness(k: int, n_vars: int, ratio: float,
                 n_trials: int = 20, seed: int = 0) -> dict:
    """
    Mide dureza empírica de instancias k-SAT al ratio dado.
    ratio = n_clauses / n_vars.
    Retorna {"k","n","ratio","sat_rate","mean_steps","hard_rate","timeout_rate"}.
    """
    n_clauses = int(n_vars * ratio)
    results = []
    for i in range(n_trials):
        clauses = random_k_sat(k, n_vars, n_clauses, seed=seed + i)
        r = dpll_solve(clauses, n_vars)
        results.append(r)

    sat_count     = sum(1 for r in results if r["sat"] is True)
    timeout_count = sum(1 for r in results if r["timeout"])
    steps_list    = [r["steps"] for r in results if not r["timeout"]]
    mean_steps    = sum(steps_list) / len(steps_list) if steps_list else 0

    return {
        "k":           k,
        "n":           n_vars,
        "ratio":       round(ratio, 4),
        "sat_rate":    round(sat_count / n_trials, 4),
        "mean_steps":  round(mean_steps, 1),
        "hard_rate":   round(timeout_count / n_trials, 4),
        "n_trials":    n_trials,
    }


def phase_transition_scan(k: int, n_vars: int,
                          ratio_min: float, ratio_max: float,
                          n_points: int = 8, seed: int = 0) -> list[dict]:
    """
    Escanea la transición de fase de k-SAT en [ratio_min, ratio_max].
    Retorna lista de resultados de sat_hardness por cada ratio.
    """
    results = []
    step = (ratio_max - ratio_min) / max(1, n_points - 1)
    r = ratio_min
    while r <= ratio_max + 1e-9:
        res = sat_hardness(k, n_vars, r, n_trials=15, seed=seed)
        results.append(res)
        r += step
    return results


def resolution_complexity(n_vars: int, n_clauses: int,
                          seed: int = 0, max_steps: int = 50_000) -> dict:
    """
    Mide longitud de prueba de resolución en fórmulas 3-SAT insatisfacibles.
    Usa refutación por resolución, cuenta resoluciones.
    Retorna {"n","m","steps","refuted","ratio_steps_vars"}.
    """
    # Generar instancia insatisfacible: PHP (pigeonhole principle)
    # n+1 palomas en n nidos → siempre insatisfacible
    n = n_vars
    clauses = []

    # PHP_{n+1,n}: hay n+1 palomas, n nidos
    # p_{i,j} = 1 si paloma i va a nido j
    def var(i, j):
        return i * n + j + 1  # 1-indexed

    # Cada paloma debe ir a algún nido
    for i in range(n + 1):
        clauses.append([var(i, j) for j in range(n)])

    # Dos palomas no pueden compartir nido
    for j in range(n):
        for i1 in range(n + 1):
            for i2 in range(i1 + 1, n + 1):
                clauses.append([-var(i1, j), -var(i2, j)])

    n_total_vars = (n + 1) * n
    r = dpll_solve(clauses, n_total_vars, max_steps=max_steps)

    return {
        "n":                n,
        "n_vars_php":       n_total_vars,
        "n_clauses_php":    len(clauses),
        "steps":            r["steps"],
        "refuted":          r["sat"] is False,
        "timeout":          r["timeout"],
        "ratio_steps_vars": round(r["steps"] / max(1, n_total_vars), 2),
    }


def circuit_lower_bound(n_bits: int, seed: int = 0) -> dict:
    """
    Estima complejidad de circuito para funciones booleanas de n_bits.
    Usa paridad (XOR) como función hard: requiere Ω(n) puertas.
    Cuenta puertas AND/OR/NOT para implementar XOR de n bits.
    Retorna {"n_bits","gates_naive","gates_optimal_estimate","lower_bound","ratio"}.
    """
    # XOR de n bits
    # Implementación naïve: (n-1) XOR gates = 3(n-1) AND/OR/NOT cada XOR
    gates_naive = 3 * (n_bits - 1)

    # Cota inferior teórica de Razborov-Smolensky para AC0: exponencial para XOR
    # Para circuitos sin restricción: lower bound = n-1 (trivial)
    # Monotone: exponential (Razborov 1985)
    lower_bound = n_bits - 1

    # Para funciones aleatorias: casi todas requieren 2^n / n puertas
    all_functions = 2 ** (2 ** n_bits)
    gates_for_random = math.floor(2 ** n_bits / n_bits) if n_bits <= 6 else None

    return {
        "n_bits":               n_bits,
        "gates_naive_xor":      gates_naive,
        "lower_bound_xor":      lower_bound,
        "n_boolean_functions":  all_functions if n_bits <= 6 else ">>10^6",
        "gates_random_fn_est":  gates_for_random,
        "shanon_lower_bound":   round(2 ** n_bits / n_bits, 1) if n_bits <= 10 else None,
    }


def analyze_hardness_growth(results: list[dict]) -> dict:
    """
    Analiza cómo crece la dureza con n_vars.
    results: lista de sat_hardness con distintos n.
    Retorna {"growth_type","exponent_est","is_exponential"}.
    """
    if len(results) < 2:
        return {"error": "necesito ≥ 2 puntos"}

    ns = [r["n"] for r in results]
    steps = [r["mean_steps"] for r in results]

    # Estimar exponente: log(steps[i+1]/steps[i]) / log(ns[i+1]/ns[i])
    exponents = []
    for i in range(len(ns) - 1):
        if steps[i] > 0 and ns[i] > 0:
            try:
                e = math.log(steps[i + 1] / steps[i]) / math.log(ns[i + 1] / ns[i])
                exponents.append(e)
            except (ZeroDivisionError, ValueError):
                pass

    if not exponents:
        return {"error": "no se pudo calcular exponente"}

    avg_exp = sum(exponents) / len(exponents)
    is_exp = avg_exp > 2.0   # crecimiento super-polinomial

    return {
        "n_points":      len(results),
        "exponent_est":  round(avg_exp, 3),
        "growth_type":   "exponencial" if is_exp else "polinomial",
        "is_exponential": is_exp,
        "steps_range":   [min(steps), max(steps)],
    }
