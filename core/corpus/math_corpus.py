"""
core/corpus/math_corpus.py — Corpus matemático especializado (v0.0.9)

Corpus domain-específico: Riemann tools solo para Riemann, PNP tools solo para PNP.
Sin dependencias torch en las funciones tool_fn — solo numpy + stdlib.
"""

from typing import Dict, List


# ── Corpus Riemann ────────────────────────────────────────────────────────────

CORPUS_RIEMANN = {
    "name": "riemann_specialist",
    "domain": "riemann",
    "knowledge": """
Hipótesis de Riemann:
  ζ(s) = Σ n^(-s) converge para Re(s) > 1
  Continuación analítica a C sin {1}
  Ceros triviales: s = -2, -4, -6, ...
  Hipótesis: todos los ceros no triviales tienen Re(s) = 1/2 (franja crítica)

Estrategias de ataque conocidas:
  1. Teoría espectral: operadores con valores propios reales ↔ ceros en Re=1/2
  2. Función ξ(s) = s(s-1)π^{-s/2}Γ(s/2)ζ(s) — simétrica en s=1/2
  3. GUE conjecture: espaciado de ceros ~ Wigner-Dyson p(s) = (32/π²)s²exp(-4s²/π)
  4. Cero-libre: no hay ceros con Re(s) cerca de 1 (región libre conocida)
  5. Teorema de Montgomery-Odlyzko: correlación de pares de ceros sigue GUE

Resultados conocidos (2025):
  - Verificados computacionalmente los primeros 10^13 ceros están en Re=1/2
  - Región libre de ceros: σ > 1 - c/log(t) para algún c > 0
""",
    "initial_tools": [
        {
            "name": "zeta_zeros_check",
            "description": "Verifica que los primeros N ceros de ζ están en Re=0.5",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    import mpmath
    mpmath.mp.dps = 15
    # omega es un array numpy — extraer seed como float seguro
    seed = int(abs(float(omega.flat[0])) * 1000) % 1000
    n_zeros = 20 + seed % 30  # entre 20 y 50 ceros
    zeros = [mpmath.zetazero(k) for k in range(1, n_zeros + 1)]
    deviations = [abs(float(z.real) - 0.5) for z in zeros]
    max_dev = max(deviations)
    all_on = max_dev < 1e-10
    score = 1.0 if all_on else max(0.0, 1.0 - max_dev * 1e6)
    best = max(range(len(candidates)), key=lambda i: score * float(abs(float(candidates[i].flat[0]))))
    return candidates[best], {
        "regime": "zeta_zeros",
        "n_checked": n_zeros,
        "max_dev": round(max_dev, 15),
        "all_on_line": all_on,
        "score": round(score, 4),
    }
""",
        },
        {
            "name": "gue_wigner_dyson",
            "description": "Compara espaciado de ceros de ζ con distribución Wigner-Dyson GUE",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    import mpmath
    mpmath.mp.dps = 12
    # Primeros 80 ceros
    zeros = np.array([float(mpmath.zetazero(k).imag) for k in range(1, 81)])
    s = np.diff(zeros)
    s = s / s.mean()  # normalizar espaciado medio a 1
    # Wigner-Dyson GUE: p(s) = (32/π²)s²exp(-4s²/π)
    wigner = lambda x: (32 / np.pi**2) * x**2 * np.exp(-4 * x**2 / np.pi)
    bins = np.linspace(0, 3, 20)
    hist, _ = np.histogram(s, bins=bins, density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    expected = wigner(centers)
    mae = float(np.mean(np.abs(hist - expected)))
    ks_stat = float(np.max(np.abs(np.cumsum(hist) / hist.sum() - np.cumsum(expected) / expected.sum())))
    score = max(0.0, 1.0 - ks_stat * 3)
    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime": "gue_wigner_dyson",
        "gue_mae": round(mae, 4),
        "ks_stat": round(ks_stat, 4),
        "gue_consistent": mae < 0.15,
        "score": round(score, 4),
    }
""",
        },
    ],
}


# ── Corpus P≠NP ───────────────────────────────────────────────────────────────

CORPUS_PNP = {
    "name": "pnp_specialist",
    "domain": "pnp",
    "knowledge": """
P≠NP — Problema del Milenio:
  P = problemas decidibles en tiempo polinomial determinístico
  NP = verificables en tiempo polinomial (o decidibles en NDTM polinomial)
  P ⊆ NP; se conjetura P ≠ NP (ningún NP-completo en P)

Estrategias conocidas:
  1. Circuitos booleanos (Razborov-Rudich): lower bounds para clases AC⁰, TC⁰
     - Håstad: parity ∉ AC⁰, circuitos de profundidad d necesitan exp(n^{1/(d-1)}) puertas
     - Natural proofs: barrera — cualquier prueba que naturalice implica consecuencias
       para pseudoaleatoriedad
  2. GCT (Geometric Complexity Theory, Mulmuley):
     - perm(M) vs det(M): perm es #P-difícil, det está en P
     - Si perm no es proyección de det(M') para M' de tamaño polinomial → P≠NP
     - Separación algebraica: perm y det tienen órbitas distintas bajo GL_n
  3. Communication complexity:
     - rank(M) ≤ circuitos determinísticos para el problema representado por M
     - parity: M tiene rango máximo sobre GF(2)
  4. Deutsch-Jozsa: separación P vs BQP (no directamente P≠NP pero ilustra separaciones)

Barreras conocidas:
  - Relativización (Baker-Gill-Solovay 1975): diagonalización no funciona
  - Natural proofs (Razborov-Rudich 1994): requiere funciones one-way
  - Algebrización (Aaronson-Wigderson 2009): extensión de relativización

Estado actual (2025):
  - Ningún avance fundamental desde las barreras
  - GCT es la aproximación algebraica más prometedora
""",
    "initial_tools": [
        {
            "name": "gct_perm_det_separation",
            "description": "Compara permanent vs determinante en matrices aleatorias (separación GCT)",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    import itertools
    # omega es array numpy
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000

    def permanent(M):
        n = len(M)
        total = 0
        for perm in itertools.permutations(range(n)):
            prod = 1
            for i in range(n):
                prod *= M[i][perm[i]]
            total += prod
        return total

    np.random.seed(seed)
    sizes = [3, 4, 5]
    results = []
    for n in sizes:
        M = np.random.randint(1, 6, (n, n)).tolist()
        perm_val = permanent(M)
        det_val = round(float(np.linalg.det(M)), 4)
        differ = abs(perm_val - det_val) > 0.01
        results.append({"n": n, "perm": perm_val, "det": det_val, "differ": differ})

    always_differ = all(r["differ"] for r in results)
    score = 0.80 if always_differ else 0.25
    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime": "gct_perm_det",
        "matrix_results": results,
        "always_differ": always_differ,
        "score": round(score, 4),
    }
""",
        },
        {
            "name": "circuit_complexity_lower_bound",
            "description": "Håstad lower bound: AC⁰ no puede computar parity, lower bound exponencial",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    # omega es array numpy
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000
    np.random.seed(seed)

    # Håstad: circuito AC⁰ de profundidad d necesita >= exp(n^{1/(d-1)}) puertas
    # para computar parity en n bits
    results = []
    for n in [8, 12, 16]:
        for depth in [2, 3, 4]:
            # Lower bound de Håstad
            hastads_lb = float(np.exp(n ** (1.0 / (depth - 1))))
            # Upper bound polinomial (si estuviera en P)
            poly_ub = n ** depth
            # Separación: lb >> poly_ub para n grande
            separation = round(hastads_lb / poly_ub, 6)
            superpolynomial = hastads_lb > poly_ub * 2
            results.append({
                "n": n, "depth": depth,
                "hastad_lb": round(hastads_lb, 2),
                "poly_ub": poly_ub,
                "separation": separation,
                "superpolynomial": superpolynomial,
            })

    n_super = sum(1 for r in results if r["superpolynomial"])
    score = min(1.0, n_super / len(results) * 1.5)
    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime": "hastad_ac0",
        "results": results,
        "n_superpolynomial": n_super,
        "score": round(score, 4),
    }
""",
        },
    ],
}


# ── Corpus Python (auto-edición) — sin domain específico ─────────────────────

CORPUS_PYTHON = {
    "name": "python_engineer",
    "domain": "code_optimization",
    "knowledge": """
Auto-edición de código Python:
  El agente puede proponer modificaciones a sus propios métodos.
  Restricciones:
    1. Solo modificar métodos en core/ — no library/, no run_*.py
    2. Cada modificación se prueba en un clon antes de aplicarse
    3. El historial de ediciones se guarda en metacognitive._self_edits

Invariantes que no deben romperse:
  - Los tensores del tejido siempre deben estar normalizados (F.normalize)
  - La biblioteca no se escribe en clones (_clone_mode)
  - El contexto del episodio se cierra con close_episode()
""",
    "initial_tools": [],
}


# ── Corpus ML (herramientas de machine learning para exploración matemática) ────

CORPUS_ML = {
    "name": "ml_tools",
    "domain": "ml_exploration",
    "knowledge": """
Herramientas de machine learning disponibles para exploración matemática:

numpy.fft:
  np.fft.fft(x)      — Transformada de Fourier discreta: detecta frecuencias en secuencias
  np.fft.rfft(x)     — FFT para señales reales (más eficiente)
  np.abs(np.fft.fft(x)) — Espectro de amplitud
  Útil para: detectar periodicidad en los ceros de ζ, patrones en gaps entre primos

scipy.optimize:
  from scipy.optimize import fsolve, minimize, root
  fsolve(f, x0)          — Encuentra raíces de f(x)=0 cerca de x0
  minimize(f, x0)        — Minimiza f(x) desde x0
  Útil para: encontrar ceros aproximados, optimizar parámetros de ajuste espectral

scipy.linalg:
  from scipy.linalg import eigvals, svd, norm
  eigvals(A)     — valores propios (más robusto que np.linalg.eigvalsh para complejos)
  svd(A)         — descomposición en valores singulares
  Útil para: análisis espectral de operadores, rango de matrices de comunicación

numpy álgebra lineal avanzada:
  np.linalg.matrix_rank(A)  — rango de la matriz
  np.linalg.cond(A)         — número de condición
  np.linalg.svd(A)          — SVD completo
  Útil para: separación GCT (detectar diferencias de rango entre perm y det)

Patrón general de uso en tool_fn:
  seed = int(abs(float(omega.flat[0])) * 1000) % 10000
  np.random.seed(seed)
  # ... análisis con numpy/scipy ...
  _result = {"metric": valor, "score": round(score, 4)}
""",
    "initial_tools": [],
}


ALL_CORPUS = {
    "riemann_specialist": CORPUS_RIEMANN,
    "pnp_specialist":     CORPUS_PNP,
    "python_engineer":    CORPUS_PYTHON,
    "ml_tools":           CORPUS_ML,
}

# Corpus domain-específico
DOMAIN_CORPUS = {
    "riemann": ["riemann_specialist", "python_engineer", "ml_tools"],
    "pnp":     ["pnp_specialist",     "python_engineer", "ml_tools"],
}


def get_corpus(name: str) -> Dict:
    return ALL_CORPUS.get(name, {})


def list_corpus() -> List[str]:
    return list(ALL_CORPUS.keys())


def list_corpus_for_domain(domain: str) -> List[str]:
    """Devuelve los corpus relevantes para un dominio específico."""
    return DOMAIN_CORPUS.get(domain, list_corpus())
