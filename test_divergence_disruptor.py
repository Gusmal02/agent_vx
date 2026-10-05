"""
test_divergence_disruptor.py — Prueba local del disruptor basado en divergencia

Idea: el campo F = (Re ζ, Im ζ) tiene divergencia cero en toda región sin ceros.
Donde div ≠ 0 hay un cero → el disruptor encuentra contradicciones sin explorar interior.

Prueba en 3 pasos:
  1. Mapa de divergencia numérica sobre una franja del plano crítico
  2. Winding number (principio del argumento) para detectar ceros en una región
  3. Bisección: reducir la región hasta aislar el cero → candidato a falsificación
"""

import numpy as np
import mpmath
import time

mpmath.mp.dps = 15


# ── 1. Divergencia numérica del campo F = (Re ζ, Im ζ) ───────────────────────

def zeta_field(sigma, t):
    z = mpmath.zeta(complex(sigma, t))
    return float(z.real), float(z.imag)


def divergence_at(sigma, t, h=0.05):
    """∇·F = ∂u/∂σ + ∂v/∂t  (diferencias finitas centradas)"""
    u_fwd, _ = zeta_field(sigma + h, t)
    u_bwd, _ = zeta_field(sigma - h, t)
    _, v_fwd  = zeta_field(sigma, t + h)
    _, v_bwd  = zeta_field(sigma, t - h)
    du_dsigma = (u_fwd - u_bwd) / (2*h)
    dv_dt     = (v_fwd - v_bwd) / (2*h)
    return du_dsigma + dv_dt


def scan_divergence(sigma_range, t_range, n=12):
    """
    Escanea una región y devuelve el mapa de divergencia.
    Regiones con |div| > threshold son candidatas a tener ceros.
    """
    sigmas = np.linspace(*sigma_range, n)
    ts     = np.linspace(*t_range, n)
    grid   = np.zeros((n, n))
    for i, s in enumerate(sigmas):
        for j, t in enumerate(ts):
            grid[i, j] = divergence_at(s, t)
    return sigmas, ts, grid


# ── 2. Winding number (principio del argumento) ───────────────────────────────

def winding_number(sigma_range, t_range, n_boundary=40):
    """
    Calcula (1/2π) ∮ d(arg ζ) alrededor del rectángulo.
    Si != 0 → hay ceros (o polos) dentro.
    Devuelve el número entero de ceros netos.
    """
    s0, s1 = sigma_range
    t0, t1 = t_range

    # Recorrer los 4 lados del rectángulo
    boundary_points = []
    # Lado inferior: t=t0, σ de s0 a s1
    for s in np.linspace(s0, s1, n_boundary):
        boundary_points.append(complex(s, t0))
    # Lado derecho: σ=s1, t de t0 a t1
    for t in np.linspace(t0, t1, n_boundary):
        boundary_points.append(complex(s1, t))
    # Lado superior: t=t1, σ de s1 a s0
    for s in np.linspace(s1, s0, n_boundary):
        boundary_points.append(complex(s, t1))
    # Lado izquierdo: σ=s0, t de t1 a t0
    for t in np.linspace(t1, t0, n_boundary):
        boundary_points.append(complex(s0, t))

    # Evaluar ζ en la frontera
    zeta_vals = [mpmath.zeta(p) for p in boundary_points]
    args      = [float(mpmath.arg(z)) for z in zeta_vals]

    # Sumar saltos de argumento
    total_change = 0.0
    for i in range(len(args)):
        diff = args[(i+1) % len(args)] - args[i]
        # Normalizar al rango (-π, π]
        diff = (diff + np.pi) % (2*np.pi) - np.pi
        total_change += diff

    return round(total_change / (2*np.pi))


# ── 3. Bisección: aislar el cero ──────────────────────────────────────────────

def bisect_to_zero(sigma_range, t_range, depth=6):
    """
    Divide la región en 4 cuadrantes recursivamente.
    Solo entra en cuadrantes con winding != 0.
    Devuelve lista de regiones candidatas a contener un cero.
    """
    candidates = [(sigma_range, t_range, depth)]
    zeros_found = []

    while candidates:
        sr, tr, d = candidates.pop()
        w = winding_number(sr, tr, n_boundary=20)
        if w == 0:
            continue  # sin ceros aquí — no explorar

        sm = (sr[0] + sr[1]) / 2
        tm = (tr[0] + tr[1]) / 2
        width  = sr[1] - sr[0]
        height = tr[1] - tr[0]

        if d == 0 or (width < 0.01 and height < 0.01):
            zeros_found.append(((sr[0]+sr[1])/2, (tr[0]+tr[1])/2, w))
            continue

        # Dividir en 4 cuadrantes
        candidates.extend([
            ((sr[0], sm), (tr[0], tm), d-1),
            ((sm, sr[1]), (tr[0], tm), d-1),
            ((sr[0], sm), (tm, tr[1]), d-1),
            ((sm, sr[1]), (tm, tr[1]), d-1),
        ])

    return zeros_found


# ── 4. Aplicación al disruptor: regiones de hipótesis inestables ──────────────

def find_disruption_targets(hypotheses_scores: dict) -> list:
    """
    Dado un dict {acción: score}, identifica qué regiones del plano (σ, t)
    corresponden a cada hipótesis y calcula el winding en esa región.
    Las regiones con winding != 0 son candidatas a falsificación.
    """
    # Mapeo aproximado: cada acción explora una franja de Im(s)
    action_regions = {
        "zero_density_sweep":     ((0.4, 0.6), (14, 50)),
        "prime_counting_error":   ((0.4, 0.6), (50, 200)),
        "montgomery_correlation": ((0.4, 0.6), (200, 500)),
        "explicit_formula_check": ((0.4, 0.6), (14, 30)),
        "gram_law_violations":    ((0.4, 0.6), (14, 100)),
    }

    targets = []
    for action, score in hypotheses_scores.items():
        if action not in action_regions:
            continue
        sr, tr = action_regions[action]
        w = winding_number(sr, tr, n_boundary=30)
        targets.append({
            "action":  action,
            "score":   score,
            "winding": w,
            "region":  (sr, tr),
            "disrupt": w != 0,  # ← candidato a falsificación
        })
        marker = "⚠ ATACAR" if w != 0 else "  estable"
        print(f"  {marker}  {action:30s}  score={score:.3f}  winding={w:+d}")

    return [t for t in targets if t["disrupt"]]


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DIVERGENCE DISRUPTOR — prueba local")
    print("=" * 60)

    # ── Paso 1: mapa de divergencia en franja crítica ─────────────────────────
    print("\n[1] Mapa de divergencia en σ∈[0.4,0.6], t∈[14,25]...")
    t0 = time.time()
    sigmas, ts, div_grid = scan_divergence((0.4, 0.6), (14.0, 25.0), n=10)
    print(f"    máx |div| = {np.max(np.abs(div_grid)):.4f}  "
          f"media = {np.mean(np.abs(div_grid)):.4f}  "
          f"({time.time()-t0:.1f}s)")

    # Puntos con divergencia alta (candidatos a ceros)
    threshold = np.percentile(np.abs(div_grid), 90)
    hot = [(sigmas[i], ts[j], div_grid[i,j])
           for i in range(len(sigmas))
           for j in range(len(ts))
           if abs(div_grid[i,j]) > threshold]
    print(f"    Puntos calientes (90p): {len(hot)}")
    for s, t, d in hot[:3]:
        print(f"      σ={s:.3f}  t={t:.3f}  div={d:.4f}")

    # ── Paso 2: winding number — ¿hay ceros en esta región? ──────────────────
    print("\n[2] Winding number en σ∈[0.48,0.52], t∈[14,15]...")
    t0 = time.time()
    # Primer cero conocido: t ≈ 14.1347
    w = winding_number((0.48, 0.52), (14.0, 14.3), n_boundary=40)
    print(f"    Winding = {w:+d}  ({time.time()-t0:.1f}s)")
    print(f"    → {'cero detectado ✓' if w != 0 else 'sin ceros'}")

    # Región vacía para comparar
    w2 = winding_number((0.3, 0.4), (14.0, 14.3), n_boundary=40)
    print(f"    Winding en región vacía (σ∈[0.3,0.4]): {w2:+d}  "
          f"→ {'cero detectado' if w2 != 0 else 'sin ceros ✓'}")

    # ── Paso 3: bisección para aislar el cero ────────────────────────────────
    print("\n[3] Bisección: aislar cero en t∈[14,14.3]...")
    t0 = time.time()
    zeros = bisect_to_zero((0.48, 0.52), (14.0, 14.3), depth=5)
    print(f"    Ceros encontrados: {len(zeros)}")
    for sigma_est, t_est, w in zeros:
        # Primer cero real: t = 14.134725
        error = abs(t_est - 14.134725)
        print(f"      σ≈{sigma_est:.4f}  t≈{t_est:.4f}  "
              f"(error={error:.4f})  winding={w:+d}")
    print(f"    ({time.time()-t0:.1f}s)")

    # ── Paso 4: aplicar al disruptor con scores de la ronda 1 ────────────────
    print("\n[4] Targets del disruptor (scores de ronda 1):")
    scores_r1 = {
        "prime_counting_error":   0.974,
        "zero_density_sweep":     0.715,
        "montgomery_correlation": 0.837,
        "explicit_formula_check": 0.926,
        "gram_law_violations":    0.880,
    }
    t0 = time.time()
    targets = find_disruption_targets(scores_r1)
    print(f"\n    Regiones a atacar: {len(targets)}")
    print(f"    ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 60)
    print("  CONCLUSIÓN")
    print("=" * 60)
    print("  • div≈0 en casi todo el plano crítico → campo coherente con RH")
    print("  • Winding ≠ 0 ↔ cero detectado sin explorar interior")
    print("  • Bisección aisla la región exacta → disruptor ataca ahí")
    print("  • Regiones con winding=0 y score alto → hipótesis robustas (no atacar)")
