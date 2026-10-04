"""
experimento_reaprendizaje.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mide si el agente reaprendé más rápido después de olvidar.

Diseño:
  S1  cold-start   → aprende desde cero               (línea base de esfuerzo)
  S2  con estado   → reutiliza conocimiento de S1      (esfuerzo mínimo esperado)
  S3  cold-start   → "olvida" todo                     (¿mismo esfuerzo que S1?)
  S4  con estado   → reutiliza conocimiento de S3      (¿mismo esfuerzo que S2?)

Hipótesis:
  H1: ciclos(S2) << ciclos(S1)     ← persistencia funciona
  H2: ciclos(S3) <= ciclos(S1)     ← reaprendizaje más rápido (transferencia implícita)
  H3: ciclos(S4) ≈ ciclos(S2)     ← el estado de S3 es equivalente al de S1

Si H2 se cumple: el agente tiene memoria implícita más allá del epistemic_state.
Si H2 no se cumple: el olvido es catastrófico — igual que lo que estudia con EWC.
"""

import subprocess
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

PROBLEM    = "continual"
MAX_HOURS  = 1.0
PYTHON_CMD = [sys.executable]   # el intérprete activo (uv run ya activo)

SESSIONS = [
    {"id": "S1", "cold_start": True,  "desc": "Aprendizaje inicial (frío)"},
    {"id": "S2", "cold_start": False, "desc": "Reutilización de S1"},
    {"id": "S3", "cold_start": True,  "desc": "Reaprendizaje (olvida S1)"},
    {"id": "S4", "cold_start": False, "desc": "Reutilización de S3"},
]


def run_session(session: dict) -> dict:
    cmd = PYTHON_CMD + [
        "run_v012.py",
        "--problem", PROBLEM,
        "--max-hours", str(MAX_HOURS),
    ]
    if session["cold_start"]:
        cmd.append("--cold-start")

    print(f"\n{'═'*60}")
    print(f"  {session['id']} — {session['desc']}")
    print(f"  Comando: {' '.join(cmd)}")
    print(f"{'═'*60}\n")

    t0     = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=Path(__file__).parent)
    elapsed = time.time() - t0

    stdout = result.stdout + result.stderr
    print(stdout[-3000:])   # últimas líneas visibles

    # Extraer métricas del output
    cycles    = _extract(r"\[Fin\] ciclos=(\d+)", stdout)
    resultado = _extract(r"resultado=(\w+)", stdout)
    richness  = _extract(r"richness=([\d.]+)", stdout)
    cross     = _extract(r"cross_hyps=(\d+)", stdout)

    return {
        "session":    session["id"],
        "desc":       session["desc"],
        "cold_start": session["cold_start"],
        "cycles":     int(cycles)   if cycles   else None,
        "resultado":  resultado,
        "richness":   float(richness) if richness else None,
        "cross_hyps": int(cross)    if cross    else None,
        "elapsed_s":  round(elapsed, 1),
    }


def _extract(pattern: str, text: str):
    m = re.search(pattern, text)
    return m.group(1) if m else None


def print_table(results: list[dict]) -> None:
    print(f"\n{'═'*70}")
    print(f"  RESULTADOS — experimento_reaprendizaje  ({PROBLEM})")
    print(f"{'─'*70}")
    print(f"  {'Sesión':<6}  {'Frío':>5}  {'Ciclos':>7}  {'Richness':>9}  "
          f"{'Cross':>6}  {'Resultado':<12}  {'Tiempo':>8}")
    print(f"  {'─'*6}  {'─'*5}  {'─'*7}  {'─'*9}  {'─'*6}  {'─'*12}  {'─'*8}")
    for r in results:
        print(f"  {r['session']:<6}  "
              f"{'sí' if r['cold_start'] else 'no':>5}  "
              f"{str(r['cycles'] or '?'):>7}  "
              f"{str(r['richness'] or '?'):>9}  "
              f"{str(r['cross_hyps'] or '?'):>6}  "
              f"{str(r['resultado'] or '?'):<12}  "
              f"{r['elapsed_s']:>7.0f}s")
    print(f"{'─'*70}")

    # Interpretación
    if len(results) >= 4:
        c = {r["session"]: r["cycles"] for r in results if r["cycles"]}
        print(f"\n  INTERPRETACIÓN:")
        if "S1" in c and "S2" in c:
            ratio_12 = c["S1"] / c["S2"] if c["S2"] else float("inf")
            h1 = "✓ CONFIRMA" if ratio_12 > 2 else "✗ NO confirma"
            print(f"  H1 (S2 << S1): {c['S1']} → {c['S2']} ciclos  "
                  f"ratio={ratio_12:.1f}×  {h1}")
        if "S1" in c and "S3" in c:
            delta = c["S1"] - c["S3"]
            h2 = "✓ CONFIRMA (transferencia)" if delta > 0 else (
                 "= NEUTRO (olvido catastrófico)" if delta == 0 else "✗ REGRESIÓN")
            print(f"  H2 (S3 <= S1): {c['S1']} → {c['S3']} ciclos  "
                  f"Δ={delta:+d}  {h2}")
        if "S2" in c and "S4" in c:
            delta = abs(c["S4"] - c["S2"])
            h3 = "✓ CONFIRMA" if delta <= 3 else "✗ Diferencia significativa"
            print(f"  H3 (S4 ≈ S2): {c['S2']} vs {c['S4']} ciclos  "
                  f"Δ={delta}  {h3}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    print(f"\n[Experimento] reaprendizaje — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Problema: {PROBLEM}  |  max_hours={MAX_HOURS} por sesión")
    print(f"  4 sesiones: S1(frío) → S2(estado) → S3(frío) → S4(estado)\n")

    results = []
    for session in SESSIONS:
        r = run_session(session)
        results.append(r)
        print_table(results)   # mostrar tabla parcial tras cada sesión

    # Guardar resultados
    out_path = Path("results") / f"reaprendizaje_{PROBLEM}_{datetime.now():%Y%m%d_%H%M}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[Guardado] → {out_path}")
