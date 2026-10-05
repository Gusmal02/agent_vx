"""
core/sandbox_v2.py — Sandbox subprocess con auto-pip-install

Diferencias vs math_sandbox.py (exec in-process):
  - Corre en subproceso aislado → crash no mata al agente
  - Auto-install: si el código importa una lib no disponible, pip install + reintento
  - Timeout duro por subproceso
  - Resultado siempre en _result (dict)
"""

import subprocess
import sys
import json
import textwrap
import tempfile
import os
import re
import time
from pathlib import Path
from typing import Optional


# Librerías pre-instaladas — no hacer pip en ellas
_PREINSTALLED = {
    "numpy", "np", "sympy", "sp", "mpmath", "scipy",
    "matplotlib", "torch", "sklearn", "pandas",
}

# Mapeo alias → nombre pip
_ALIAS_TO_PIP: dict[str, str] = {
    "np":      "numpy",
    "sp":      "sympy",
    "plt":     "matplotlib",
    "sklearn": "scikit-learn",
    "cv2":     "opencv-python-headless",
}

_MAX_INSTALL_RETRIES = 1   # un solo intento de pip por ejecución
_DEFAULT_TIMEOUT     = 60  # segundos


def _extract_missing_module(stderr: str) -> Optional[str]:
    """Extrae el nombre del módulo que falta del traceback."""
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", stderr)
    if m:
        raw = m.group(1).split(".")[0]   # solo el paquete raíz
        return _ALIAS_TO_PIP.get(raw, raw)
    return None


def _pip_install(package: str) -> bool:
    """Instala un paquete con pip. Devuelve True si tuvo éxito."""
    print(f"  [Sandbox] pip install {package} ...", flush=True)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", package],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode == 0:
            print(f"  [Sandbox] {package} instalado ✓", flush=True)
            return True
        print(f"  [Sandbox] pip FALLÓ: {result.stderr[:200]}", flush=True)
        return False
    except Exception as e:
        print(f"  [Sandbox] pip error: {e}", flush=True)
        return False


def _run_script(script_path: str, timeout: int) -> tuple[int, str, str]:
    """Ejecuta el script en subproceso. Devuelve (returncode, stdout, stderr)."""
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True, text=True, timeout=timeout,
    )
    return result.returncode, result.stdout, result.stderr


def run_code(code: str, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """
    Ejecuta `code` en un subproceso aislado.
    El código debe asignar su resultado a `_result` (dict).

    Devuelve:
        {
          "success":  bool,
          "result":   dict | None,   # valor de _result
          "stdout":   str,
          "stderr":   str,
          "duration": float,
          "installed": list[str],    # paquetes que se tuvieron que instalar
        }
    """
    installed: list[str] = []

    # Envolver el código del usuario para que _result se serialice a stdout
    wrapper = textwrap.dedent(f"""
import json, sys, traceback
try:
{textwrap.indent(code, '    ')}
    print("__RESULT__:" + json.dumps(_result if isinstance(_result, dict) else {{"value": str(_result)}}))
except Exception as _e:
    print("__RESULT__:" + json.dumps({{"error": str(_e), "traceback": traceback.format_exc()}}))
    sys.exit(1)
""")

    # Escribir a archivo temporal
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(wrapper)
        script = f.name

    t0 = time.time()
    try:
        for attempt in range(_MAX_INSTALL_RETRIES + 1):
            try:
                rc, stdout, stderr = _run_script(script, timeout)
            except subprocess.TimeoutExpired:
                return {
                    "success":   False,
                    "result":    None,
                    "stdout":    "",
                    "stderr":    f"TimeoutExpired ({timeout}s)",
                    "duration":  time.time() - t0,
                    "installed": installed,
                }

            # ¿Faltó una lib?
            missing = _extract_missing_module(stderr)
            if missing and missing not in _PREINSTALLED and attempt == 0:
                ok = _pip_install(missing)
                if ok:
                    installed.append(missing)
                    continue   # reintentar

            # Extraer _result del stdout
            result_dict = None
            clean_stdout = []
            for line in stdout.splitlines():
                if line.startswith("__RESULT__:"):
                    try:
                        result_dict = json.loads(line[len("__RESULT__:"):])
                    except Exception:
                        pass
                else:
                    clean_stdout.append(line)

            success = rc == 0 and result_dict is not None and "error" not in result_dict
            return {
                "success":   success,
                "result":    result_dict,
                "stdout":    "\n".join(clean_stdout),
                "stderr":    stderr,
                "duration":  time.time() - t0,
                "installed": installed,
            }

        # Agotamos reintentos
        return {
            "success":   False,
            "result":    None,
            "stdout":    "",
            "stderr":    "Reintentos de instalación agotados",
            "duration":  time.time() - t0,
            "installed": installed,
        }

    finally:
        try:
            os.unlink(script)
        except Exception:
            pass


# ── Interfaz de alto nivel para el agente ────────────────────────────────────

class SandboxV2:
    """Sandbox subprocess para el agente. Misma API que MathSandbox."""

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT):
        self.timeout    = timeout
        self._run_count = 0
        self._total_s   = 0.0

    def execute(self, code: str, context: dict | None = None) -> dict:
        """
        Ejecuta código. context: variables adicionales a inyectar.
        Devuelve el mismo dict que run_code().
        """
        full_code = ""
        if context:
            for k, v in context.items():
                full_code += f"{k} = {json.dumps(v)}\n"
        full_code += code

        r = run_code(full_code, self.timeout)
        self._run_count += 1
        self._total_s   += r["duration"]

        if r["installed"]:
            print(f"  [SandboxV2] auto-instalados: {r['installed']}", flush=True)
        if not r["success"]:
            err = r.get("stderr", "") or (r.get("result") or {}).get("error", "")
            print(f"  [SandboxV2] error: {err[:200]}", flush=True)

        return r

    def stats(self) -> dict:
        return {
            "runs":    self._run_count,
            "total_s": round(self._total_s, 2),
            "avg_s":   round(self._total_s / max(1, self._run_count), 2),
        }
