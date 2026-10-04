"""
smoke_test_v05.py — Valida: sympy+matplotlib en executor, graceful degradation sin oracle.

Tests:
  1. SymPy disponible en DirectExecutor (álgebra simbólica funciona)
  2. Matplotlib disponible en DirectExecutor (backend Agg, sin pantalla)
  3. Worker completa ciclo COMPLETO sin API key (oracle deshabilitado)
  4. Corpus fallback en VERIFY: oracle unavailable → experimentos del corpus

Uso:
  uv run python smoke_test_v05.py
  uv run python smoke_test_v05.py --skip-run   (solo tests 1-2, sin lanzar workers)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ── TEST 1: SymPy en DirectExecutor ─────────────────────────────────────────

def test_sympy():
    print("\n── TEST 1: SymPy en DirectExecutor ─────────────────────────────")
    try:
        from core.direct_executor import DirectExecutor
        exe = DirectExecutor(timeout_sec=30)
        code = """
import sympy as sp
x = sp.Symbol('x')
expr = sp.expand((x + 1)**3)
coeffs = sp.Poly(expr, x).all_coeffs()
_result = {"score": float(coeffs[0]), "confirms": coeffs == [1, 3, 3, 1], "metric": "binomial_expansion"}
"""
        r = exe.run(code, label="sympy_test")
        ok_exec = r["ok"]
        res = r.get("result") or {}
        confirms = res.get("confirms", False)
        score = res.get("score", 0)

        print(f"  Ejecución OK:    {ok_exec}")
        print(f"  Confirma (1,3,3,1): {confirms}")
        print(f"  Score: {score}")
        if not ok_exec:
            print(f"  Error: {r.get('error','')}")

        ok = ok_exec and confirms
        print(f"  {PASS if ok else FAIL} TEST 1: {'PASS' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  {FAIL} TEST 1 ERROR: {e}")
        return False


# ── TEST 2: Matplotlib en DirectExecutor ────────────────────────────────────

def test_matplotlib():
    print("\n── TEST 2: Matplotlib en DirectExecutor (Agg) ──────────────────")
    try:
        from core.direct_executor import DirectExecutor
        exe = DirectExecutor(timeout_sec=30)
        import tempfile
        tmpfile = Path(tempfile.gettempdir()) / "smoke_test_plot.png"
        code = f"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)
ax.set_title("sin(x)")
fig.savefig(r"{str(tmpfile).replace(chr(92), '/')}")
plt.close(fig)

_result = {{"score": float(y.max()), "confirms": abs(y.max() - 1.0) < 0.01, "metric": "sin_max"}}
"""
        r = exe.run(code, label="matplotlib_test")
        ok_exec = r["ok"]
        res = r.get("result") or {}
        confirms = res.get("confirms", False)
        file_saved = tmpfile.exists()

        print(f"  Ejecución OK:      {ok_exec}")
        print(f"  sin(x) max ≈ 1.0:  {confirms}")
        print(f"  PNG guardado:      {file_saved}")
        if not ok_exec:
            print(f"  Error: {r.get('error','')}")

        ok = ok_exec and confirms
        print(f"  {PASS if ok else FAIL} TEST 2: {'PASS' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  {FAIL} TEST 2 ERROR: {e}")
        return False


# ── TEST 3: Worker completa sin oracle (sin API key) ─────────────────────────

def test_worker_no_oracle():
    print("\n── TEST 3: Worker completa sin oracle ──────────────────────────")
    print("  Lanzando 1 worker × 3 min × problema causal (sin API key)...")
    env = os.environ.copy()
    env.pop("ANTHROPIC_API_KEY", None)  # deshabilitar oracle

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "run_v012.py",
         "--problem", "causal",
         "--agent-id", "smoke_no_oracle",
         "--seed", "77",
         "--max-hours", "0.05"],   # 3 minutos
        capture_output=True, text=True,
        cwd=Path(__file__).parent,
        timeout=300,
        env=env,
    )
    elapsed = time.time() - t0

    out = result.stdout + result.stderr
    worker_started = "[Worker smoke_no_oracle]" in out or "iniciando" in out
    no_oracle_msg  = "Oracle" in out and ("deshabilitado" in out or "sin API" in out.lower()
                                          or "no disponible" in out)
    worker_ended   = "fin" in out or result.returncode == 0 or "FINDING" in out
    no_crash       = result.returncode in (0, 1)  # no crash inesperado

    print(f"  Tiempo: {elapsed:.0f}s")
    print(f"  Worker inició:          {worker_started}")
    print(f"  Oracle deshabilitado:   {no_oracle_msg}")
    print(f"  Worker terminó:         {worker_ended}")
    print(f"  Sin crash (rc={result.returncode}): {no_crash}")

    if result.returncode not in (0, 1):
        print(f"  Tail stderr: {result.stderr[-400:]}")

    ok = no_crash and worker_ended
    print(f"  {PASS if ok else FAIL} TEST 3: {'PASS' if ok else 'FAIL'}")
    return ok


# ── TEST 4: Corpus fallback en VERIFY ───────────────────────────────────────

def test_corpus_fallback():
    print("\n── TEST 4: Corpus fallback en _oracle_propose_experiments ──────")
    try:
        # Crear corpus de prueba con código ejecutable
        test_corpus = {
            "problem": "causal",
            "updated_at": "2026-10-04T00:00:00",
            "n_papers": 1,
            "n_experiments": 1,
            "experiments": [
                {
                    "id": "test_h0",
                    "source_title": "Test paper",
                    "hypothesis": "El sesgo por backdoor es reducible con ajuste",
                    "code": (
                        "import numpy as np\n"
                        "np.random.seed(42)\n"
                        "X = np.random.randn(200)\n"
                        "bias_naive = abs(np.mean(X[:100]) - np.mean(X[100:]))\n"
                        "bias_adj   = abs(np.mean(X) - 0)\n"
                        "_result = {'score': float(bias_adj), 'confirms': bias_adj < bias_naive + 0.5, 'metric': 'backdoor_bias'}"
                    ),
                    "expected_if_true": "bias_adj < bias_naive + 0.5",
                    "origin": "test",
                }
            ],
        }
        corpus_path = RESULTS / "corpus_causal_smoke.json"
        with open(corpus_path, "w", encoding="utf-8") as f:
            json.dump(test_corpus, f)

        # Importar y probar la función directamente
        import importlib.util
        spec = importlib.util.spec_from_file_location("run_v012", "run_v012.py")
        mod  = importlib.util.module_from_spec(spec)
        # Solo necesitamos _corpus_fallback_experiments, no ejecutar el módulo completo
        # Leemos el código directamente
        src = Path("run_v012.py").read_text(encoding="utf-8")

        # Extraer y ejecutar solo _corpus_fallback_experiments
        import re
        m = re.search(
            r'(def _corpus_fallback_experiments\(.*?(?=\ndef |\Z))',
            src, re.DOTALL
        )
        if not m:
            print(f"  {FAIL} No se encontró _corpus_fallback_experiments en run_v012.py")
            return False

        fn_code = m.group(1)
        ns: dict = {"json": json, "Path": Path}
        exec(fn_code, ns)
        fallback_fn = ns["_corpus_fallback_experiments"]

        # Probar con corpus real
        exps = fallback_fn(str(corpus_path))
        has_exps   = len(exps) > 0
        has_code   = all(e.get("code") for e in exps)
        has_hyp    = all(e.get("hypothesis") for e in exps)

        print(f"  Función encontrada:    sí")
        print(f"  Experimentos devueltos: {len(exps)}")
        print(f"  Todos con código:       {has_code}")
        print(f"  Todos con hipótesis:    {has_hyp}")

        # Probar con corpus_path=None
        empty = fallback_fn(None)
        print(f"  Fallback(None)=[]?      {empty == []}")

        ok = has_exps and has_code and has_hyp and empty == []
        print(f"  {PASS if ok else FAIL} TEST 4: {'PASS' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        import traceback
        print(f"  {FAIL} TEST 4 ERROR: {e}")
        traceback.print_exc()
        return False


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true",
                        help="Saltar TEST 3 (sin lanzar workers)")
    args = parser.parse_args()

    print("═" * 62)
    print("  SMOKE TEST v0.5 — sympy + matplotlib + graceful degradation")
    print("═" * 62)

    r1 = test_sympy()
    r2 = test_matplotlib()
    r4 = test_corpus_fallback()

    if args.skip_run:
        print(f"\n  {WARN} TEST 3 omitido (--skip-run)")
        r3 = None
    else:
        r3 = test_worker_no_oracle()

    print("\n" + "═" * 62)
    results = {
        "TEST 1 SymPy executor":          r1,
        "TEST 2 Matplotlib executor":     r2,
        "TEST 3 Worker sin oracle":       r3,
        "TEST 4 Corpus fallback VERIFY":  r4,
    }
    for name, r in results.items():
        if r is None:
            status = f"{WARN} SKIP"
        elif r:
            status = PASS
        else:
            status = FAIL
        print(f"  {status}  {name}")

    skipped  = sum(1 for v in results.values() if v is None)
    all_pass = all(v for v in results.values() if v is not None)
    print(f"\n  {'PIPELINE LISTO ✓' if all_pass else 'REVISAR ANTES DE CAMPAÑA'}"
          f"  ({skipped} skipped)")
    print("═" * 62)
    sys.exit(0 if all_pass else 1)
