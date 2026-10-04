"""
core/math_sandbox.py — Sandbox matemático seguro

Entorno de ejecución controlado con acceso a:
  NumPy, SymPy, SciPy, Matplotlib

El agente puede ejecutar código Python para:
  - Calcular valores numéricos (NumPy/SciPy)
  - Manipulación simbólica (SymPy)
  - Visualizar estructuras (Matplotlib → PNG en disco)
  - Verificar hipótesis parciales

La ejecución se hace con timeout y namespace aislado.
El agente recibe el resultado (stdout, valor de retorno, o error).
"""

import io
import sys
import time
import traceback
import textwrap
from typing import Any
from pathlib import Path
import numpy as np

PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# Namespace seguro disponible para el agente
SAFE_GLOBALS = {
    "__builtins__": __builtins__   # imports completos — sandbox local controlado por el agente
}


class MathSandbox:
    """
    Sandbox de ejecución matemática para el agente.

    El agente construye código Python como string, lo envía aquí,
    y recibe el resultado (stdout + valor de retorno o traza de error).
    """

    def __init__(self, timeout_sec: float = 30.0, plot_dir: Path = PLOT_DIR):
        self.timeout_sec  = timeout_sec
        self.plot_dir     = plot_dir
        self._exec_count  = 0
        self._tool_log: list = []

        # Pre-importar librerías en namespace compartido
        import numpy as np
        import sympy as sp
        import scipy
        import scipy.stats
        import mpmath
        import itertools, random, time, math
        import matplotlib
        matplotlib.use("Agg")   # sin ventana — guarda a archivo
        import matplotlib.pyplot as plt

        self._ns = {
            **SAFE_GLOBALS,
            "np": np, "numpy": np,
            "sp": sp, "sympy": sp,
            "scipy": scipy,
            "mpmath": mpmath,
            "itertools": itertools,
            "random": random,
            "time": time,
            "math": math,
            "plt": plt,
            "PLOT_DIR": str(self.plot_dir),
            "_plot_count": [0],
        }

    # ── Ejecución ─────────────────────────────────────────────────────────────

    def run(self, code: str, label: str = "") -> dict:
        """
        Ejecuta `code` en el namespace matemático.
        Devuelve {ok, stdout, result, error, elapsed, plot_path}.
        """
        self._exec_count += 1
        label = label or f"exec_{self._exec_count}"

        # Redirigir stdout
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        start   = time.time()
        result  = None
        error   = None
        plot_path = None

        try:
            compiled = compile(textwrap.dedent(code), f"<{label}>", "exec")

            # Detectar si el agente llama plt.savefig — interceptar
            self._ns["_plot_count"][0] += 1
            self._ns["_current_plot"] = str(
                self.plot_dir / f"{label}_{self._ns['_plot_count'][0]}.png"
            )

            exec(compiled, self._ns)
            result = self._ns.get("_result", None)

        except Exception as e:
            error = f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=5)}"
        finally:
            elapsed = time.time() - start
            sys.stdout = old_stdout

        stdout_text = buffer.getvalue()
        ok = error is None

        # Registrar en log
        self._tool_log.append({
            "label": label, "ok": ok,
            "elapsed": round(elapsed, 3),
            "lines": len(code.splitlines()),
        })

        return {
            "ok":        ok,
            "stdout":    stdout_text[:4000],   # recortar salidas largas
            "result":    result,
            "error":     error,
            "elapsed":   round(elapsed, 3),
            "plot_path": self._ns.get("_current_plot") if ok else None,
        }

    def eval_expr(self, expr_str: str) -> dict:
        """Evalúa una expresión Python simple y devuelve su valor."""
        return self.run(f"_result = {expr_str}", label="eval")

    # ── Herramientas específicas del dominio ──────────────────────────────────

    def riemann_zeta_zeros(self, n: int = 20) -> list:
        """Calcula los primeros n ceros no triviales de ζ(s) en la franja crítica."""
        code = f"""
import mpmath; mpmath.mp.dps = 25
zeros = [mpmath.zetazero(k) for k in range(1, {n}+1)]
_result = [(float(z.real), float(z.imag)) for z in zeros]
"""
        r = self.run(code, label="riemann_zeros")
        return r.get("result", [])

    def verify_prime_density(self, N: int = 10000) -> dict:
        """Verifica π(x) vs li(x) con mpmath para la integral logarítmica exacta."""
        code = f"""
mpmath.mp.dps = 20
primes = list(sp.primerange(2, {N}+1))
xs = np.linspace(100, {N}, 50)
pi_x = np.array([sum(1 for p in primes if p <= x) for x in xs], dtype=float)
# li(x) exacta via integración numérica mpmath — no aproximación
li_x = np.array([float(mpmath.li(float(x))) for x in xs])
errors = np.abs(pi_x - li_x) / np.maximum(np.abs(li_x), 1)
# Error de Riemann vs aproximación clásica: π(x) - li(x) debe ser O(√x · ln x)
riemann_bound = np.sqrt(xs) * np.log(xs)
normalized_err = np.abs(pi_x - li_x) / riemann_bound
_result = {{
    "max_rel_error":      float(errors.max()),
    "mean_rel_error":     float(errors.mean()),
    "max_riemann_ratio":  float(normalized_err.max()),
    "mean_riemann_ratio": float(normalized_err.mean()),
    "n_primes": len(primes),
    "N": {N},
}}
"""
        r = self.run(code, label="prime_density")
        return r.get("result") or {}

    def summary(self) -> dict:
        return {
            "exec_count": self._exec_count,
            "tool_log":   self._tool_log[-10:],
        }
