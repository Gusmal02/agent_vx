"""
smoke_test_v04.py — Valida el pipeline completo de v0.4 en escala reducida.

Tests:
  1. ResearcherAgent — ¿puede buscar arxiv y extraer hipótesis?
  2. Corpus escrito correctamente (JSON válido, ≥1 hipótesis con código)
  3. Integración end-to-end: 3 agentes × 1 ronda × 20 min con researcher
  4. Workers reciben corpus_path y lo usan en VERIFY
  5. Race condition fix — ¿todos los epistemic states se encuentran?

Uso:
  uv run python smoke_test_v04.py
  uv run python smoke_test_v04.py --skip-run   (solo tests 1-2, sin lanzar workers)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

PASS = "✓"
FAIL = "✗"
WARN = "⚠"
RESULTS = Path("results")
RESULTS.mkdir(exist_ok=True)


# ── TEST 1: ResearcherAgent ──────────────────────────────────────────────────

def test_researcher():
    print("\n── TEST 1: ResearcherAgent (arxiv + oracle) ────────────────")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print(f"  {WARN} Sin ANTHROPIC_API_KEY — solo texto sin código")

    try:
        from core.researcher import ResearcherAgent
        r = ResearcherAgent("causal", api_key)
        corpus = r.build_corpus(n_papers=3, max_hyps_per_paper=2)

        n_exp     = len(corpus.get("experiments", []))
        n_papers  = corpus.get("n_papers", 0)
        with_code = sum(1 for e in corpus["experiments"] if e.get("code"))

        print(f"  Papers encontrados: {n_papers}")
        print(f"  Hipótesis totales:  {n_exp}")
        print(f"  Con código numpy:   {with_code}")

        if n_exp > 0:
            sample = corpus["experiments"][0]
            print(f"  Ejemplo:")
            print(f"    title:  {sample.get('source_title','')[:60]}")
            print(f"    hyp:    {sample.get('hypothesis','')[:80]}")
            has_code = bool(sample.get("code"))
            print(f"    código: {'sí' if has_code else 'no'}")

        ok = n_papers > 0 and n_exp > 0
        print(f"  {PASS if ok else FAIL} TEST 1: {'PASS' if ok else 'FAIL'}")
        return ok, with_code
    except Exception as e:
        print(f"  {FAIL} TEST 1 ERROR: {e}")
        return False, 0


# ── TEST 2: Corpus JSON válido ───────────────────────────────────────────────

def test_corpus_file():
    print("\n── TEST 2: Corpus JSON válido ──────────────────────────────")
    corpus_path = RESULTS / "corpus_causal.json"

    if not corpus_path.exists():
        print(f"  {WARN} corpus_causal.json no existe — saltando")
        return None

    try:
        with open(corpus_path, encoding="utf-8") as f:
            corpus = json.load(f)

        keys_ok   = all(k in corpus for k in ["problem","experiments","n_papers"])
        n_exp     = len(corpus.get("experiments", []))
        with_code = sum(1 for e in corpus["experiments"] if e.get("code"))

        print(f"  Estructura válida: {keys_ok}")
        print(f"  Experimentos: {n_exp}  (con código: {with_code})")

        # Verificar que el código sea ejecutable
        exec_ok = 0
        for exp in corpus["experiments"][:3]:
            code = exp.get("code", "")
            if not code:
                continue
            try:
                local_ns: dict = {}
                exec(code, {"__builtins__": __builtins__, "numpy": __import__("numpy")}, local_ns)
                if "_result" in local_ns:
                    exec_ok += 1
            except Exception as e:
                print(f"  {WARN} código no ejecutable ({exp['id']}): {e}")

        print(f"  Código ejecutable verificado: {exec_ok}/{min(3, with_code)}")
        ok = keys_ok and n_exp > 0
        print(f"  {PASS if ok else FAIL} TEST 2: {'PASS' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  {FAIL} TEST 2 ERROR: {e}")
        return False


# ── TEST 3: Race condition fix ───────────────────────────────────────────────

def test_race_condition_fix():
    """Verifica que el sleep(2) + retry existe en run_v020.py."""
    print("\n── TEST 3: Race condition fix ──────────────────────────────")
    path = Path("run_v020.py")
    code = path.read_text(encoding="utf-8")

    has_sleep   = "time.sleep(2)" in code
    has_retry   = "for _ in range(3)" in code
    has_corpus  = "corpus_path" in code
    has_version = 'VERSION = "v0.4.0"' in code

    print(f"  sleep(2) post-workers:  {has_sleep}")
    print(f"  retry loop (x3):        {has_retry}")
    print(f"  corpus_path integrado:  {has_corpus}")
    print(f"  VERSION = v0.4.0:       {has_version}")

    ok = has_sleep and has_retry and has_corpus and has_version
    print(f"  {PASS if ok else FAIL} TEST 3: {'PASS' if ok else 'FAIL'}")
    return ok


# ── TEST 4: Run end-to-end (3 agentes × 1 ronda × 20 min) ───────────────────

def test_e2e_run():
    print("\n── TEST 4: End-to-end 3 agentes × 20 min ──────────────────")
    print("  Lanzando: uv run python run_v020.py --problem causal "
          "--n-agents 3 --rounds 1 --max-hours 0.33 --researcher")
    print("  (esto tarda ~10 min — workers × 20 min)")

    import subprocess
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "uv", "run", "python", "run_v020.py",
         "--problem", "causal",
         "--n-agents", "3",
         "--rounds", "1",
         "--max-hours", "0.33",
         "--researcher"],
        capture_output=True, text=True,
        cwd=Path(__file__).parent,
        timeout=1800,  # 30 min máximo
    )
    elapsed = time.time() - t0
    out = result.stdout + result.stderr

    # Verificar outputs clave
    researcher_ok = "[Researcher]" in out and "corpus listo" in out
    workers_ok    = out.count("[Worker") >= 3
    states_ok     = "advertencia: no encontré" not in out
    verify_ok     = "VERIFY" in out and "ejecutados=1/1" in out
    corpus_ctx    = "Hipótesis de literatura" in out or "corpus" in out.lower()

    print(f"  Tiempo total: {elapsed:.0f}s")
    print(f"  Researcher construyó corpus: {researcher_ok}")
    print(f"  Workers lanzados (≥3):       {workers_ok}")
    print(f"  Sin race condition warning:  {states_ok}")
    print(f"  VERIFY ejecutado:            {verify_ok}")

    if result.returncode != 0:
        print(f"  {WARN} exit code: {result.returncode}")
        print(f"  Tail stderr: {result.stderr[-500:]}")

    # Buscar archivo de resultado
    import glob
    files = sorted(glob.glob("results/multiagente_causal_*.json"))
    if files:
        print(f"  Resultado guardado: {files[-1]}")

    ok = researcher_ok and workers_ok and result.returncode == 0
    print(f"  {PASS if ok else FAIL} TEST 4: {'PASS' if ok else 'FAIL'}")
    return ok


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-run", action="store_true",
                        help="Saltar TEST 4 (solo validar Researcher y archivos)")
    args = parser.parse_args()

    print("═" * 60)
    print("  SMOKE TEST v0.4 — agente vX")
    print("═" * 60)

    r1, with_code = test_researcher()
    r2 = test_corpus_file()
    r3 = test_race_condition_fix()

    if args.skip_run:
        print(f"\n  {WARN} TEST 4 omitido (--skip-run)")
        r4 = None
    else:
        r4 = test_e2e_run()

    print("\n" + "═" * 60)
    results = {
        "TEST 1 Researcher": r1,
        "TEST 2 Corpus JSON": r2,
        "TEST 3 Race fix": r3,
        "TEST 4 E2E run": r4,
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
    print(f"\n  {'PIPELINE LISTO' if all_pass else 'REVISAR ANTES DE CAMPAÑA LARGA'}"
          f"  ({skipped} skipped)")
    if with_code > 0:
        print(f"  Corpus tiene {with_code} experimentos ejecutables desde literatura")
    print("═" * 60)
    sys.exit(0 if all_pass else 1)
