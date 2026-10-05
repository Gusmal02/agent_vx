"""
math_tools.py — Herramientas matemáticas del agente.

El agente las aplica a regiones que él elige.
Ninguna de estas funciones es generada por LLM.
"""

import math


def winding_number(sigma_range, t_range, n=40):
    """
    Número de vueltas de ζ(s) alrededor del origen en el contorno del rectángulo.
    ≠ 0 → hay ceros (o polos) dentro. O(n) evaluaciones en la frontera.
    """
    import mpmath
    import numpy as np
    s0, s1 = sigma_range
    t0, t1 = t_range
    pts = (
        [complex(s, t0) for s in np.linspace(s0, s1, n)] +
        [complex(s1, t) for t in np.linspace(t0, t1, n)] +
        [complex(s, t1) for s in np.linspace(s1, s0, n)] +
        [complex(s0, t) for t in np.linspace(t1, t0, n)]
    )
    args = [float(mpmath.arg(mpmath.zeta(p))) for p in pts]
    total = 0.0
    for i in range(len(args)):
        d = args[(i + 1) % len(args)] - args[i]
        d = (d + math.pi) % (2 * math.pi) - math.pi
        total += d
    return round(total / (2 * math.pi))


def isolate_zeros(sigma_range, t_range, known_winding=None, depth=7):
    """
    Bisección recursiva para aislar ceros individuales.
    Retorna lista de (sigma_est, t_est) para cada cero.
    """
    initial_w = known_winding if known_winding is not None else winding_number(sigma_range, t_range)
    candidates = [(sigma_range, t_range, depth, initial_w)]
    found = []

    while candidates:
        sr, tr, d, w = candidates.pop()
        if w == 0:
            continue
        sm = (sr[0] + sr[1]) / 2
        tm = (tr[0] + tr[1]) / 2
        if d == 0 or (sr[1] - sr[0] < 0.0005 and tr[1] - tr[0] < 0.0005):
            found.append(((sr[0] + sr[1]) / 2, (tr[0] + tr[1]) / 2))
            continue
        for quad in [((sr[0], sm), (tr[0], tm)),
                     ((sm, sr[1]), (tr[0], tm)),
                     ((sr[0], sm), (tm, tr[1])),
                     ((sm, sr[1]), (tm, tr[1]))]:
            candidates.append((quad[0], quad[1], d - 1, None))

    # Deduplicar: fusionar candidatos dentro de tolerancia 0.1
    merged = []
    for pt in sorted(found, key=lambda p: p[1]):
        if not merged or abs(pt[1] - merged[-1][1]) > 0.1:
            merged.append(pt)
    return merged


def verify_zero(sigma_est, t_est, dps=25):
    """
    Localiza el cero exacto con mpmath.findroot.
    Retorna {"sigma", "t", "deviation", "on_line"}.
    """
    import mpmath
    mpmath.mp.dps = dps
    try:
        root = mpmath.findroot(mpmath.zeta, mpmath.mpc(sigma_est, t_est))
        sigma = float(root.real)
        t     = float(root.imag)
        dev   = abs(sigma - 0.5)
        return {
            "sigma":    round(sigma, 10),
            "t":        round(t, 8),
            "deviation": round(dev, 12),
            "on_line":  dev < 1e-6,
        }
    except Exception as e:
        return {"error": str(e), "sigma_est": sigma_est, "t_est": t_est}


def scan_strip(t_min, t_max, step=5.0, sigma_range=(0.48, 0.52)):
    """
    Escanea la franja t ∈ [t_min, t_max] en ventanas de `step`.
    Retorna lista de ventanas con winding ≠ 0.
    """
    hot = []
    t = t_min
    while t < t_max:
        t1 = min(t + step, t_max)
        w = winding_number(sigma_range, (t, t1))
        if w != 0:
            hot.append({"t_min": t, "t_max": t1, "winding": w})
        t = t1
    return hot


def zero_density(t_min, t_max, sigma_range=(0.48, 0.52)):
    """
    Estima la densidad de ceros en la franja usando winding number.
    Compara con la fórmula de Riemann-von Mangoldt: N(T) ≈ T/(2π) * log(T/(2πe))
    Retorna {"found", "expected", "ratio"}.
    """
    import math
    w = winding_number(sigma_range, (t_min, t_max))
    n_found = abs(w)

    def N(T):
        if T < 2:
            return 0
        return T / (2 * math.pi) * math.log(T / (2 * math.pi * math.e))

    n_expected = max(0.01, N(t_max) - N(t_min))
    return {
        "t_min":    t_min,
        "t_max":    t_max,
        "found":    n_found,
        "expected": round(n_expected, 2),
        "ratio":    round(n_found / n_expected, 4),
    }


def analyze_spacing(zeros):
    """
    Analiza el espaciado entre ceros consecutivos.
    Compara con la distribución GUE (Montgomery-Odlyzko).
    Retorna {"mean_spacing", "variance", "min_gap", "max_gap", "gue_ratio"}.
    """
    import numpy as np
    if len(zeros) < 2:
        return {"error": "necesito ≥ 2 ceros"}
    ts = sorted(z["t"] for z in zeros if "t" in z)
    gaps = np.diff(ts)
    mean = float(np.mean(gaps))
    # Normalizar por espaciado medio
    normed = gaps / mean
    # GUE predice distribución Wigner: P(s) ≈ (π/2)s·exp(-πs²/4)
    # Varianza de GUE ≈ 0.286
    variance = float(np.var(normed))
    gue_var  = 4 / math.pi - 1   # ≈ 0.2732
    return {
        "n_gaps":        len(gaps),
        "mean_spacing":  round(mean, 6),
        "variance_norm": round(variance, 6),
        "gue_variance":  round(gue_var, 6),
        "gue_ratio":     round(variance / gue_var, 4),
        "min_gap":       round(float(np.min(gaps)), 6),
        "max_gap":       round(float(np.max(gaps)), 6),
    }
