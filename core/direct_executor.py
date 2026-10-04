"""
core/direct_executor.py — Ejecución directa con entorno Python completo (v0.1.0)

Reemplaza MathSandbox para los nuevos problemas ML.
Sin restricciones de imports — torch, sklearn, scipy, networkx disponibles.

Thread-safe: cada llamada usa su propio StringIO buffer via contextlib.redirect_stdout.
NO redirige sys.stdout globalmente (ese era el bug de deadlock en v009).

Timeout via threading.Thread.join() — el worker corre en daemon thread.
"""

from __future__ import annotations

import contextlib
import io
import threading
import time
import traceback
from queue import Queue, Empty
from typing import Any, Optional

# Fix: importar matplotlib una sola vez a nivel de módulo, no en cada llamada.
# En Linux headless, importar pyplot en múltiples threads simultáneos puede
# colgar el proceso al intentar inicializar fontcache o tocar locks del sistema.
_matplotlib = None
_plt = None
try:
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
except (ImportError, Exception):
    pass


def _make_namespace() -> dict:
    """
    Namespace base para exec(). Importa las librerías disponibles.
    Falla silenciosamente si una librería no está instalada.
    """
    import numpy as np
    import scipy
    import networkx as nx

    ns: dict[str, Any] = {
        "np":      np,
        "numpy":   np,
        "scipy":   scipy,
        "nx":      nx,
        "networkx": nx,
    }

    for lib, alias in [
        ("torch",               "torch"),
        ("sklearn",             "sklearn"),
        ("sklearn.datasets",    "datasets"),
        ("sklearn.linear_model","linear_model"),
        ("sklearn.metrics",     "metrics"),
        ("sklearn.neural_network", "nn_sklearn"),
        ("sklearn.preprocessing", "preprocessing"),
        ("scipy.optimize",      "optimize"),
        ("scipy.linalg",        "linalg"),
        ("scipy.stats",         "stats"),
        ("itertools",           "itertools"),
        ("random",              "random"),
        ("math",                "math"),
        ("json",                "json"),
        ("collections",         "collections"),
        ("sympy",               "sympy"),
        ("sympy",               "sp"),
    ]:
        try:
            import importlib
            mod = importlib.import_module(lib)
            ns[alias] = mod
        except ImportError:
            pass

    # Reusar el import de nivel de módulo — ya inicializado una sola vez
    if _matplotlib is not None:
        ns["matplotlib"] = _matplotlib
    if _plt is not None:
        ns["plt"] = _plt

    return ns


class DirectExecutor:
    """
    Ejecuta código Python directamente con acceso al entorno completo.

    Diferencia clave vs MathSandbox:
    - Sin restricciones de imports
    - Thread-safe: sin manipulación de sys.stdout global
    - Timeout vía daemon thread
    """

    def __init__(self, timeout_sec: float = 60.0):
        self.timeout_sec  = timeout_sec
        self._call_count  = 0
        self._count_lock  = threading.Lock()

    def run(self, code: str, label: str = "exec") -> dict:
        """
        Ejecuta `code` en un namespace con las libs ML disponibles.
        Devuelve {ok, result, stdout, error, elapsed}.
        """
        with self._count_lock:
            self._call_count += 1

        result_q: Queue = Queue()

        def _worker():
            buf = io.StringIO()
            ns  = _make_namespace()
            try:
                with contextlib.redirect_stdout(buf):
                    exec(compile(code, f"<{label}>", "exec"), ns)  # noqa: S102
                _result = ns.get("_result", {})
                if not isinstance(_result, dict):
                    _result = {"value": _result}
                result_q.put({
                    "ok":     True,
                    "result": _result,
                    "stdout": buf.getvalue(),
                    "error":  None,
                })
            except Exception:
                result_q.put({
                    "ok":     False,
                    "result": None,
                    "stdout": buf.getvalue(),
                    "error":  traceback.format_exc(limit=5),
                })

        t = threading.Thread(target=_worker, daemon=True)
        t_start = time.time()
        t.start()
        t.join(timeout=self.timeout_sec)
        elapsed = round(time.time() - t_start, 3)

        if t.is_alive():
            return {
                "ok":     False,
                "result": None,
                "stdout": "",
                "error":  f"timeout after {self.timeout_sec}s",
                "elapsed": elapsed,
            }

        try:
            r = result_q.get_nowait()
        except Empty:
            return {
                "ok":     False,
                "result": None,
                "stdout": "",
                "error":  "worker finished without result",
                "elapsed": elapsed,
            }

        r["elapsed"] = elapsed
        return r

    def summary(self) -> dict:
        return {"calls": self._call_count, "timeout_sec": self.timeout_sec}
