"""
verify_continual_finding.py  —  v2
───────────────────────────────────
Verificación del hallazgo central de v0.1.2 (continual).

HIPÓTESIS (tal como el agente la reportó):
  EWC muestra mayor mejora relativa cuando la similitud de tareas
  NO predice bien el olvido. Correlación cruzada: r(sim_fgt_corr, ret_impr) < -0.70.

DISEÑO DEL EXPERIMENTO:
  Se crean N_SCENARIOS escenarios de "estructura de tareas". Cada escenario
  tiene un nivel de complejidad (overlap) que controla cuánto comparten
  las tareas el espacio de parámetros.

  Para CADA escenario se miden simultáneamente ambas métricas del agente:
    1. sim_forgetting_corr: 8 pares con similitud aleatoria entre [-1, 1];
       mide qué tan bien predice sim el olvido en ese escenario.
    2. retention_improvement: par fijo con overlap=scenario_overlap;
       mide cuánto ayuda EWC en ese escenario específico.

  Luego se correlaciona sim_fgt_corr vs retention_improvement entre escenarios.

PREDICCIÓN FALSIFICABLE:
  - Escenario overlap alto (tareas pelean):
      sim_forgetting_corr BAJO (sim no predice bien)  +  ret_impr ALTO (EWC ayuda)
  - Escenario overlap bajo (tareas distintas):
      sim_forgetting_corr ALTO (sim predice bien)     +  ret_impr BAJO (EWC no necesario)
  - Correlación cruzada: r < -0.70
"""

import numpy as np

RNG_SEED     = 2026
N_SCENARIOS  = 40   # escenarios de estructura
N_PAIRS      = 10   # pares por escenario para medir sim_fgt_corr
D_IN         = 30
N_SAMPLES    = 500


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def accuracy(W, X, y):
    return float(np.mean((sigmoid(X @ W.T)[:, 0] > 0.5) == y))


def train_sgd(X, y, W_init, lr=0.04, steps=400, ewc_W=None, ewc_F=None, lam=0.0):
    W = W_init.copy()
    for _ in range(steps):
        p = sigmoid(X @ W.T)[:, 0]; e = p - y
        g = np.zeros_like(W)
        g[0] = (e[:, None] * X).mean(0)
        if ewc_W is not None and lam > 0:
            g += lam * ewc_F * (W - ewc_W)
        W -= lr * g
    return W


def fisher_diag(X, y, W):
    p = sigmoid(X @ W.T)[:, 0]; e = p - y
    F = np.zeros_like(W)
    F[0] = ((e[:, None] * X) ** 2).mean(0)
    return F


def make_task_pair(w_A, overlap: float, rng):
    """
    Crea w_B con `overlap` fracción de features compartidas con w_A.
    overlap=1.0 → w_B = -w_A (conflicto total)
    overlap=0.0 → w_B = vector ortogonal a w_A (sin conflicto)
    """
    w_opp  = -w_A
    w_orth = rng.standard_normal(D_IN)
    w_orth -= np.dot(w_orth, w_A) * w_A
    w_orth /= np.linalg.norm(w_orth) + 1e-9

    w_B = overlap * w_opp + (1 - overlap) * w_orth
    w_B /= np.linalg.norm(w_B) + 1e-9
    return w_B


def measure_sim_forgetting_corr(overlap: float, rng):
    """
    Métrica 1 (igual que task_similarity del agente):
    Genera N_PAIRS pares con similitud aleatoria y mide qué tan bien
    predice la similitud el olvido.
    Pero ahora, la similitud real de cada par depende de overlap_base + ruido.
    """
    sims, fgts = [], []
    for _ in range(N_PAIRS):
        # Similitud aleatoria en [-1, 1] para que haya varianza
        sim_target = rng.uniform(-1.0, 1.0)

        w_A = rng.standard_normal(D_IN)
        w_A /= np.linalg.norm(w_A) + 1e-9

        w_orth = rng.standard_normal(D_IN)
        w_orth -= np.dot(w_orth, w_A) * w_A
        w_orth /= np.linalg.norm(w_orth) + 1e-9

        # w_B = sim_target * w_A + sqrt(1-sim²) * w_orth
        t = np.clip(sim_target, -1.0, 1.0)
        w_B = t * w_A + np.sqrt(max(0, 1 - t**2)) * w_orth
        w_B /= np.linalg.norm(w_B) + 1e-9

        # En este ESCENARIO, el olvido también depende del overlap base del escenario
        # → escalamos la "dificultad" del entrenamiento por overlap
        effective_conflict = 0.5 + 0.5 * overlap  # [0.5, 1.0]
        w_B_scaled = effective_conflict * (-w_A) + (1 - effective_conflict) * w_B
        w_B_scaled /= np.linalg.norm(w_B_scaled) + 1e-9

        X = rng.standard_normal((N_SAMPLES, D_IN))
        y_A = (X @ w_A > 0).astype(float)
        y_B = (X @ w_B_scaled > 0).astype(float)

        W_init = rng.standard_normal((1, D_IN)) * 0.05
        W_A    = train_sgd(X, y_A, W_init, steps=350)
        acc_A_before = accuracy(W_A, X, y_A)
        W_AB   = train_sgd(X, y_B, W_A, steps=250)
        acc_A_after  = accuracy(W_AB, X, y_A)
        forgetting   = acc_A_before - acc_A_after

        sims.append(float(sim_target))
        fgts.append(float(forgetting))

    sims_arr = np.array(sims)
    retns_arr = -np.array(fgts)   # retención = -olvido

    if np.std(sims_arr) < 1e-6 or np.std(retns_arr) < 1e-6:
        return 0.0
    return float(np.corrcoef(sims_arr, retns_arr)[0, 1])


def measure_ewc_retention(overlap: float, rng):
    """
    Métrica 2 (igual que ewc_retention del agente):
    Par fijo A→B con conflicto = overlap.
    Mide cuánto mejora EWC sobre fine-tune sin EWC.
    """
    w_A = rng.standard_normal(D_IN)
    w_A /= np.linalg.norm(w_A) + 1e-9
    w_B = make_task_pair(w_A, overlap, rng)

    X = rng.standard_normal((N_SAMPLES, D_IN))
    y_A = (X @ w_A > 0).astype(float)
    y_B = (X @ w_B > 0).astype(float)

    W_init   = rng.standard_normal((1, D_IN)) * 0.05
    W_A      = train_sgd(X, y_A, W_init, steps=500)
    acc_base = accuracy(W_A, X, y_A)
    F_A      = fisher_diag(X, y_A, W_A)

    # Fine-tune sin EWC
    W_ft = train_sgd(X, y_B, W_A, steps=350)
    fgt_noewc = acc_base - accuracy(W_ft, X, y_A)

    # Fine-tune con EWC — buscar mejor λ
    best_fgt_ewc = fgt_noewc
    for lam in [1.0, 5.0, 20.0, 80.0, 200.0]:
        W_ewc    = train_sgd(X, y_B, W_A, ewc_W=W_A, ewc_F=F_A, lam=lam, steps=350)
        fgt_ewc  = acc_base - accuracy(W_ewc, X, y_A)
        if fgt_ewc < best_fgt_ewc:
            best_fgt_ewc = fgt_ewc

    return float(fgt_noewc - best_fgt_ewc)


# ══════════════════════════════════════════════════════════════════════════════
# Experimento principal
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    rng = np.random.default_rng(RNG_SEED)

    overlap_values = np.linspace(0.05, 0.95, N_SCENARIOS)
    scenarios = []

    print(f"\n{'='*72}")
    print(f"  Verificación hallazgo continual v0.1.2")
    print(f"  H: r(sim_fgt_corr, ret_impr) < -0.70")
    print(f"  {N_SCENARIOS} escenarios × {N_PAIRS} pares | d_in={D_IN} n={N_SAMPLES}")
    print(f"{'='*72}")
    print(f"\n  {'overlap':>8}  {'sim_fgt_corr':>14}  {'ret_impr':>10}")
    print(f"  {'─'*8}  {'─'*14}  {'─'*10}")

    for overlap in overlap_values:
        sfcorr = measure_sim_forgetting_corr(overlap, rng)
        retimpr = measure_ewc_retention(overlap, rng)
        scenarios.append({
            "overlap":              float(overlap),
            "sim_forgetting_corr":  sfcorr,
            "retention_improvement": retimpr,
        })
        print(f"  {overlap:8.2f}  {sfcorr:14.4f}  {retimpr:10.4f}")

    # ── Correlación cruzada entre las dos métricas ────────────────────────────
    sfcorr_arr  = np.array([s["sim_forgetting_corr"]   for s in scenarios])
    retimpr_arr = np.array([s["retention_improvement"] for s in scenarios])

    if np.std(sfcorr_arr) < 1e-6 or np.std(retimpr_arr) < 1e-6:
        cross_r = float("nan")
    else:
        cross_r = float(np.corrcoef(sfcorr_arr, retimpr_arr)[0, 1])

    # ── Verificar por régimen ─────────────────────────────────────────────────
    hi_overlap = [s for s in scenarios if s["overlap"] >= 0.7]   # tareas pelean
    lo_overlap = [s for s in scenarios if s["overlap"] <= 0.3]   # tareas distintas

    mean_sfc_hi  = float(np.mean([s["sim_forgetting_corr"]   for s in hi_overlap]))
    mean_ret_hi  = float(np.mean([s["retention_improvement"] for s in hi_overlap]))
    mean_sfc_lo  = float(np.mean([s["sim_forgetting_corr"]   for s in lo_overlap]))
    mean_ret_lo  = float(np.mean([s["retention_improvement"] for s in lo_overlap]))

    print(f"\n{'─'*72}")
    print(f"\n  RESULTADOS")
    print(f"  {'─'*60}")
    print(f"\n  r(sim_fgt_corr ~ ret_improvement): {cross_r:+.3f}")
    pred_r = "✓ CONFIRMA" if not np.isnan(cross_r) and cross_r < -0.70 else "✗ NO confirma"
    print(f"  Predicción r < -0.70:  {pred_r}")

    print(f"\n  Régimen CONFLICTO ALTO (overlap ≥ 0.70, n={len(hi_overlap)}):")
    print(f"    sim_fgt_corr = {mean_sfc_hi:+.3f}   predicción: BAJO  "
          f"{'✓' if mean_sfc_hi < 0.4 else '✗'}")
    print(f"    ret_impr     = {mean_ret_hi:+.4f}  predicción: ALTO  "
          f"{'✓' if mean_ret_hi > 0.05 else '✗'}")

    print(f"\n  Régimen CONFLICTO BAJO (overlap ≤ 0.30, n={len(lo_overlap)}):")
    print(f"    sim_fgt_corr = {mean_sfc_lo:+.3f}   predicción: ALTO  "
          f"{'✓' if mean_sfc_lo > 0.4 else '✗'}")
    print(f"    ret_impr     = {mean_ret_lo:+.4f}  predicción: BAJO  "
          f"{'✓' if mean_ret_lo < 0.10 else '✗'}")

    # ── Veredicto ─────────────────────────────────────────────────────────────
    preds = [
        not np.isnan(cross_r) and cross_r < -0.70,
        mean_sfc_hi < 0.4 and mean_ret_hi > 0.05,
        mean_sfc_lo > 0.4 and mean_ret_lo < 0.10,
    ]
    n_pass = sum(preds)

    print(f"\n{'─'*72}")
    symbols = ["★★★", "★★☆", "★☆☆", "✗✗✗"]
    msgs    = [
        "HALLAZGO VERIFICADO (3/3 predicciones)",
        "PARCIALMENTE VERIFICADO (2/3)",
        "DÉBILMENTE VERIFICADO (1/3)",
        "NO VERIFICADO — probable artefacto de n pequeño",
    ]
    symbol, msg = symbols[3 - n_pass], msgs[3 - n_pass]
    print(f"\n  {symbol}  {msg}")
    print(f"  r_cross={cross_r:+.3f}  "
          f"ret_hi={mean_ret_hi:.4f}  ret_lo={mean_ret_lo:.4f}  "
          f"sfc_hi={mean_sfc_hi:.3f}  sfc_lo={mean_sfc_lo:.3f}")
    print(f"\n{'='*72}\n")
