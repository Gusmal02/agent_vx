"""
run_v010.py — Agente vX v0.1.0

Cambios respecto a v0.0.9:
  - Problemas: causal (razonamiento causal) y continual (olvido catastrófico)
  - Sin sandbox restringido — DirectExecutor con entorno Python completo
  - ResearchMemory: corpus dinámico, subagentes indexan resultados
  - creative_crisis: detecta tensión entre hipótesis SUPPORTED y sintetiza
  - ActionInventor: oracle inventa nuevas herramientas cuando el score es bajo
  - Corpus ML: scipy, sklearn, networkx disponibles desde el inicio

Uso:
    uv run python run_v010.py --problem causal    --hours 4
    uv run python run_v010.py --problem continual --hours 4
    uv run python run_v010.py --problem causal    --hours 8 --resume
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

from library.store              import ResonantLibrary
from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from core.second_order          import SecondOrderAgent
from core.direct_executor       import DirectExecutor
from core.satisfaction_meter    import SatisfactionMeter
from core.checkpoint            import CheckpointManager
from core.tool_registry         import ToolRegistry
from core.math_env              import MathEnv
from core.oracle_client         import OracleClient
from core.subagent_pool         import SubagentPool
from core.hypothesis_tracker    import HypothesisTracker
from core.research_memory       import ResearchMemory
from core.action_inventor       import ActionInventor
from core.corpus.ml_problems_corpus import (
    get_ml_corpus, list_ml_corpus_for_domain
)

load_dotenv()

VERSION = "v0.1.0"

# ── Configuración ─────────────────────────────────────────────────────────────

BASE_CONFIG = {
    "N": 100, "M": 3, "K": 3,
    "stagnation_threshold": 8,
    "filter_threshold":     0.35,
    "confidence_threshold": 0.88,
    "epsilon_start":        0.50,
    "n_episodios":          15,
}

PROBLEM_ACTIONS = {
    "causal": [
        "backdoor_adjustment",
        "do_calculus_test",
        "irm_vs_erm",
        "causal_discovery",
        "counterfactual_bounds",
    ],
    "continual": [
        "gradient_interference",
        "ewc_retention",
        "fisher_geometry",
        "task_similarity",
        "replay_vs_finetune",
    ],
}

PERSPECTIVE_EMPHASIS = {
    "mathematician":   [0, 1, 2, 3, 4],
    "physicist":       [2, 3, 1, 0, 4],
    "algorithmist":    [4, 0, 2, 1, 3],
    "python_engineer": [1, 0, 3, 2, 4],
}

ORACLE_BUDGET = 2.50


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Causalidad
# ══════════════════════════════════════════════════════════════════════════════

def _causal_action(action: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker,
                   log: dict) -> float:

    if action == "backdoor_adjustment":
        code = f"""
import numpy as np
np.random.seed({cycle * 17 + 3})
results = []
for trial in range(5):
    n = 1500
    # SCM: Z → X → Y, Z → Y  (Z = confounder)
    alpha_zx = 0.6 + np.random.uniform(-0.2, 0.2)
    alpha_zy = 0.4 + np.random.uniform(-0.2, 0.2)
    alpha_xy = 0.5 + np.random.uniform(-0.3, 0.3)  # efecto causal verdadero
    Z = np.random.randn(n)
    X = alpha_zx * Z + np.random.randn(n) * 0.4
    Y = alpha_xy * X + alpha_zy * Z + np.random.randn(n) * 0.4
    # Estimación naive (sesgada)
    b_naive = np.cov(X, Y)[0,1] / (np.var(X) + 1e-9)
    # Ajuste backdoor (regresión con Z)
    A = np.column_stack([X, Z, np.ones(n)])
    b_adj = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    bias_n = abs(b_naive - alpha_xy)
    bias_a = abs(b_adj   - alpha_xy)
    results.append({{"true": round(float(alpha_xy),4),
                     "naive": round(float(b_naive),4),
                     "adjusted": round(float(b_adj),4),
                     "bias_naive": round(float(bias_n),4),
                     "bias_adj":   round(float(bias_a),4),
                     "reduction":  round(float(max(0, bias_n-bias_a)/(bias_n+1e-9)),4)}})
mean_red = sum(r["reduction"] for r in results) / len(results)
score = min(1.0, 0.2 + mean_red * 0.8)
_result = {{"action": "backdoor_adjustment", "trials": results,
            "mean_bias_reduction": round(float(mean_red),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"backdoor_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.6:
            tracker.record(
                f"Ciclo {cycle}: ajuste backdoor reduce sesgo causal "
                f"{res.get('mean_bias_reduction',0):.0%} en promedio",
                domain="causal", confidence=0.72,
                evidence=f"5 trials, ajuste por Z",
            )
        return score

    elif action == "do_calculus_test":
        code = f"""
import numpy as np
import networkx as nx
np.random.seed({cycle * 13 + 7})
# Generar DAGs aleatorios y verificar criterio de backdoor
n_dags = 8
results = []
for _ in range(n_dags):
    # DAG de 4 nodos: X=0, Y=1, Z=2 (confounder), W=3 (mediador)
    structure = np.random.choice(["simple", "mediator", "chain"], p=[0.4, 0.3, 0.3])
    n = 800
    Z = np.random.randn(n)
    W = np.random.randn(n)
    if structure == "simple":
        # Z → X, Z → Y, X → Y
        X = 0.7*Z + np.random.randn(n)*0.4
        Y = 0.5*X + 0.4*Z + np.random.randn(n)*0.4
        true_ace = 0.5
    elif structure == "mediator":
        # X → W → Y (W es mediador)
        X = np.random.randn(n)
        W = 0.8*X + np.random.randn(n)*0.3
        Y = 0.6*W + np.random.randn(n)*0.4
        true_ace = 0.8 * 0.6  # efecto total via mediador
    else:
        # Cadena: Z → X → Y → V
        X = 0.7*Z + np.random.randn(n)*0.3
        V = 0.5*X + np.random.randn(n)*0.4
        Y = V.copy()
        true_ace = 0.5
    # Estimación naive
    b_naive = np.cov(X, Y)[0,1] / (np.var(X)+1e-9)
    # Ajuste (si hay Z observable)
    A = np.column_stack([X, Z, np.ones(n)])
    b_adj = np.linalg.lstsq(A, Y, rcond=None)[0][0]
    err_naive = abs(b_naive - true_ace)
    err_adj   = abs(b_adj   - true_ace)
    results.append({{"structure": structure, "true_ace": round(true_ace,3),
                     "naive": round(float(b_naive),3), "adjusted": round(float(b_adj),3),
                     "err_naive": round(float(err_naive),4), "err_adj": round(float(err_adj),4),
                     "adj_better": bool(err_adj < err_naive)}})
frac_better = sum(1 for r in results if r["adj_better"]) / len(results)
score = min(1.0, 0.3 + frac_better * 0.7)
_result = {{"action": "do_calculus_test", "n_dags": n_dags, "results": results,
            "frac_adj_better": round(frac_better,3), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"docalc_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.65:
            tracker.record(
                f"Ciclo {cycle}: do-calculus identifica efecto causal en "
                f"{res.get('frac_adj_better',0):.0%} de DAGs aleatorios",
                domain="causal", confidence=0.70,
                evidence=f"n_dags={res.get('n_dags')}, ajuste backdoor",
            )
        return score

    elif action == "irm_vs_erm":
        code = f"""
import numpy as np
np.random.seed({cycle * 11 + 5})
# IRM: busca representación invariante entre entornos
# Entorno e1: correlación espuria +0.9, entorno e2: -0.9
# Entorno test: sin correlación espuria
n_trials = 6
trial_results = []
for trial in range(n_trials):
    d = 10
    n_env = 400
    true_coef = np.zeros(d); true_coef[0] = 1.5; true_coef[1] = -1.0  # causas reales
    spurious_coef = np.zeros(d); spurious_coef[2] = 2.0  # correlación espuria
    env_errors = []
    for gamma in [0.9, -0.9]:  # entornos de entrenamiento
        X = np.random.randn(n_env, d)
        X[:, 2] = gamma * (true_coef[:2] @ X[:, :2].T) + np.random.randn(n_env)*0.1
        Y = X @ true_coef + np.random.randn(n_env)*0.5
        # ERM
        b_erm = np.linalg.lstsq(np.column_stack([X, np.ones(n_env)]), Y, rcond=None)[0][:-1]
        env_errors.append(float(np.mean((X @ b_erm - Y)**2)))
    # Test (gamma=0): sin correlación espuria
    X_test = np.random.randn(n_env, d)
    Y_test = X_test @ true_coef + np.random.randn(n_env)*0.5
    # ERM naive (usa features espurias aprendidas en training)
    X_all = np.vstack([np.random.randn(n_env, d) for _ in range(2)])
    gammas = [0.9, -0.9]
    for i, g in enumerate(gammas):
        X_all[i*n_env:(i+1)*n_env, 2] = g * (true_coef[:2] @ X_all[i*n_env:(i+1)*n_env, :2].T)
    Y_all = X_all @ true_coef + np.random.randn(2*n_env)*0.5
    b_erm_all = np.linalg.lstsq(np.column_stack([X_all, np.ones(2*n_env)]), Y_all, rcond=None)[0][:-1]
    mse_erm_test = float(np.mean((X_test @ b_erm_all - Y_test)**2))
    # IRM proxy: usar solo features causales (0,1) — invariantes entre entornos
    b_irm = np.linalg.lstsq(np.column_stack([X_all[:,:2], np.ones(2*n_env)]), Y_all, rcond=None)[0]
    mse_irm_test = float(np.mean((X_test[:,:2] @ b_irm[:2] + b_irm[2] - Y_test)**2))
    trial_results.append({{"mse_erm_test": round(mse_erm_test,4),
                           "mse_irm_test": round(mse_irm_test,4),
                           "irm_wins": bool(mse_irm_test < mse_erm_test),
                           "gap": round(float(mse_erm_test - mse_irm_test),4)}})
frac_irm_wins = sum(1 for r in trial_results if r["irm_wins"]) / len(trial_results)
mean_gap = sum(r["gap"] for r in trial_results) / len(trial_results)
score = min(1.0, 0.2 + frac_irm_wins * 0.5 + max(0, mean_gap)*0.3)
_result = {{"action": "irm_vs_erm", "trials": trial_results,
            "frac_irm_wins": round(frac_irm_wins,3),
            "mean_gap": round(float(mean_gap),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"irm_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.6:
            tracker.record(
                f"Ciclo {cycle}: IRM supera ERM en entorno test en "
                f"{res.get('frac_irm_wins',0):.0%} de trials "
                f"(gap medio={res.get('mean_gap',0):.3f})",
                domain="causal", confidence=0.68,
                evidence=f"entornos gamma=±0.9, test gamma=0",
            )
        return score

    elif action == "causal_discovery":
        code = f"""
import numpy as np
np.random.seed({cycle * 19 + 11})
# PC algorithm simplificado: recuperar esqueleto del DAG por tests de independencia
# Medir SHD (Structural Hamming Distance) normalizada
n_graphs = 5
results = []
for _ in range(n_graphs):
    n = 600
    d = 5  # 5 variables
    # DAG verdadero aleatorio (triangular superior)
    true_adj = (np.random.rand(d, d) > 0.6).astype(float)
    true_adj = np.triu(true_adj, k=1)  # solo aristas hacia adelante
    # Generar datos del SCM
    X = np.zeros((n, d))
    for j in range(d):
        parents = np.where(true_adj[:, j] > 0)[0]
        noise = np.random.randn(n) * 0.5
        if len(parents) > 0:
            X[:, j] = X[:, parents] @ true_adj[parents, j] + noise
        else:
            X[:, j] = noise
    # PC skeleton: test de independencia parcial (correlación parcial)
    # Dos variables son independientes dadas Z si corr(X_i, X_j | Z) ≈ 0
    estimated_adj = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            # Test simple: correlación parcial controlando por el resto
            others = [k for k in range(d) if k != i and k != j]
            if others:
                # Residualizar X_i y X_j sobre las demás
                Z = X[:, others]
                Xi_res = X[:, i] - Z @ np.linalg.lstsq(Z, X[:, i], rcond=None)[0]
                Xj_res = X[:, j] - Z @ np.linalg.lstsq(Z, X[:, j], rcond=None)[0]
                corr = np.corrcoef(Xi_res, Xj_res)[0, 1]
            else:
                corr = np.corrcoef(X[:, i], X[:, j])[0, 1]
            if abs(corr) > 0.15:  # umbral de independencia
                estimated_adj[i, j] = 1
                estimated_adj[j, i] = 1
    # SHD: aristas en verdadero pero no estimado + viceversa
    true_skeleton = ((true_adj + true_adj.T) > 0).astype(float)
    np.fill_diagonal(true_skeleton, 0)
    np.fill_diagonal(estimated_adj, 0)
    shd = int(np.sum(np.abs(true_skeleton - estimated_adj)))
    max_edges = d * (d-1)
    shd_norm = shd / (max_edges + 1e-9)
    results.append({{"n_true_edges": int(true_skeleton.sum()//2),
                     "n_estimated": int(estimated_adj.sum()//2),
                     "shd": shd, "shd_norm": round(float(shd_norm),4)}})
mean_shd = sum(r["shd_norm"] for r in results) / len(results)
score = min(1.0, max(0.0, 1.0 - mean_shd * 3))
_result = {{"action": "causal_discovery", "n_graphs": n_graphs, "results": results,
            "mean_shd_norm": round(float(mean_shd),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"discovery_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: PC algorithm recupera esqueleto causal "
                f"SHD={res.get('mean_shd_norm',1):.3f} normalizado",
                domain="causal", confidence=0.65,
                evidence=f"n_graphs={res.get('n_graphs')}, d=5 variables",
            )
        return score

    else:  # counterfactual_bounds
        code = f"""
import numpy as np
np.random.seed({cycle * 23 + 13})
# Cotas de Tian-Pearl para efectos causales con variables binarias
# P(Y=1|do(X=1)) - P(Y=1|do(X=0)) bounded by observational data
results = []
for trial in range(6):
    n = 1000
    # SCM binario: U no observable, X binario, Y binario
    p_u = 0.4
    U = (np.random.rand(n) < p_u).astype(int)
    X = ((np.random.rand(n) < 0.3 + 0.4 * U)).astype(int)  # U confunde X
    Y = ((np.random.rand(n) < 0.2 + 0.5 * X + 0.3 * U)).astype(int)

    # Observacional: P(Y=1|X=0), P(Y=1|X=1)
    p_y_x0 = Y[X==0].mean() if (X==0).sum() > 0 else 0.5
    p_y_x1 = Y[X==1].mean() if (X==1).sum() > 0 else 0.5
    p_x1   = X.mean()

    # Cota inferior y superior de Tian-Pearl para ACE = P(Y|do(X=1)) - P(Y|do(X=0))
    lb = max(p_y_x1 * p_x1 - (1-p_y_x0)*(1-p_x1),
             -(p_y_x0 * (1-p_x1) + (1-p_y_x1) * p_x1))
    ub = min(p_y_x1 * p_x1 + p_y_x0 * (1-p_x1),
             p_x1 + p_y_x0 - 2 * p_y_x0 * p_x1 + (1 - p_y_x1 + p_y_x0) * p_x1)

    # Verdadero ACE (disponible porque conocemos el SCM)
    Y_do1 = (np.random.rand(n) < 0.2 + 0.5 * 1 + 0.3 * U).astype(int)
    Y_do0 = (np.random.rand(n) < 0.2 + 0.5 * 0 + 0.3 * U).astype(int)
    true_ace = float(Y_do1.mean() - Y_do0.mean())

    in_bounds = float(lb) <= true_ace <= float(ub)
    width = float(ub) - float(lb)
    results.append({{"lb": round(float(lb),4), "ub": round(float(ub),4),
                     "true_ace": round(true_ace,4), "in_bounds": in_bounds,
                     "width": round(width,4)}})

frac_valid = sum(1 for r in results if r["in_bounds"]) / len(results)
mean_width = sum(r["width"] for r in results) / len(results)
score = min(1.0, 0.2 + frac_valid * 0.5 + max(0, 0.5 - mean_width) * 0.6)
_result = {{"action": "counterfactual_bounds", "trials": results,
            "frac_bounds_valid": round(frac_valid,3),
            "mean_width": round(float(mean_width),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"ctf_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: cotas de Tian-Pearl válidas en "
                f"{res.get('frac_bounds_valid',0):.0%} de trials "
                f"(ancho medio={res.get('mean_width',1):.3f})",
                domain="causal", confidence=0.67,
                evidence=f"SCM binario con confounder U no observable",
            )
        return score


# ══════════════════════════════════════════════════════════════════════════════
# Acciones — Olvido Catastrófico
# ══════════════════════════════════════════════════════════════════════════════

def _continual_action(action: str, cycle: int,
                      executor: DirectExecutor,
                      tracker: HypothesisTracker,
                      log: dict) -> float:

    if action == "gradient_interference":
        code = f"""
import numpy as np
np.random.seed({cycle * 17 + 3})
# Medir coseno entre gradientes de múltiples pares de tareas
d_in = 20
n    = 300
results = []
for pair_idx in range(8):
    W = np.random.randn(1, d_in) * 0.1
    # Tarea A: features pares
    X_A = np.random.randn(n, d_in); X_A[:, 1::2] *= 0.05
    y_A = (X_A[:, ::2].sum(axis=1) > 0).astype(float)
    # Tarea B: features impares + rotación según ciclo
    angle = ({cycle} * 0.1 + pair_idx * 0.3) % np.pi
    X_B = np.random.randn(n, d_in); X_B[:, ::2] *= 0.05
    y_B = (np.cos(angle)*X_B[:, 1::2].sum(1) + np.sin(angle)*X_B[:, ::2].sum(1) > 0).astype(float)
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def grad(X, y, W):
        p = sigmoid(X @ W.T)[:,0]; e = p - y
        return (e[:,None]*X).mean(axis=0)
    gA = grad(X_A, y_A, W)
    gB = grad(X_B, y_B, W)
    cos = float(np.dot(gA,gB) / (np.linalg.norm(gA)*np.linalg.norm(gB)+1e-9))
    results.append({{"pair": pair_idx, "cosine_sim": round(cos,4),
                     "interfere": bool(cos < 0), "angle_deg": round(float(np.degrees(angle)),1)}})
n_conflict = sum(1 for r in results if r["interfere"])
mean_cos   = sum(r["cosine_sim"] for r in results) / len(results)
score = min(1.0, 0.3 + (0.5 - mean_cos) * 0.7) if mean_cos < 0.5 else 0.35
_result = {{"action": "gradient_interference", "n_pairs": len(results),
            "n_conflict": n_conflict, "mean_cosine": round(float(mean_cos),4),
            "results": results[:4], "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"grad_int_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: {res.get('n_conflict',0)}/{res.get('n_pairs',8)} pares "
                f"muestran interferencia de gradientes (cos={res.get('mean_cosine',0):.3f})",
                domain="continual", confidence=0.70,
                evidence=f"8 pares de tareas, d_in=20",
            )
        return score

    elif action == "ewc_retention":
        lam_base = 1.0 + (cycle % 10) * 2.0   # lambda varía con el ciclo
        code = f"""
import numpy as np
np.random.seed({cycle * 13 + 7})
# Tareas con conflicto real: mismo espacio X, etiquetas parcialmente opuestas
# Tarea A: y = signo(w_A · x),  Tarea B: y = signo(w_B · x) donde w_B ≈ -w_A
# Esto fuerza olvido real al cambiar de A a B
d_in = 30; n = 500
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def acc(W, X, y): return float(np.mean((sigmoid(X@W.T)[:,0]>0.5)==y))
def train(W, X, y, lr=0.05, steps=300, ewc_W=None, ewc_F=None, lam=0.0):
    W = W.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        g = np.zeros_like(W); g[0] = (e[:,None]*X).mean(0)
        if ewc_W is not None and lam > 0: g += lam * ewc_F * (W - ewc_W)
        W -= lr * g
    return W
def fisher(W, X, y):
    p = sigmoid(X@W.T)[:,0]; e = p-y
    g_per = e[:,None]*X
    F = np.zeros_like(W); F[0] = (g_per**2).mean(0)
    return F
W_init = np.random.randn(2, d_in) * 0.05
# w_A y w_B son opuestos en los primeros d_in//2 features (conflicto real)
w_A = np.random.randn(d_in); w_A[d_in//2:] *= 0.1
w_B = w_A.copy(); w_B[:d_in//2] *= -1.0  # signo opuesto en la mitad clave
X_shared = np.random.randn(n, d_in)
y_A = (X_shared @ w_A > 0).astype(float)
y_B = (X_shared @ w_B > 0).astype(float)
conflict_rate = float(np.mean(y_A != y_B))  # fracción de ejemplos en conflicto
W_A = train(W_init, X_shared, y_A, steps=400)
acc_A_base = acc(W_A, X_shared, y_A)
F_A = fisher(W_A, X_shared, y_A)
lambda_results = []
for lam in [0.0, {lam_base:.2f}, {lam_base*3:.2f}, {lam_base*10:.2f}]:
    W_ft = train(W_A, X_shared, y_B, ewc_W=W_A, ewc_F=F_A, lam=lam)
    a_A  = acc(W_ft, X_shared, y_A)
    a_B  = acc(W_ft, X_shared, y_B)
    lambda_results.append({{"lambda": round(lam,3), "acc_A": round(a_A,3),
                            "acc_B": round(a_B,3), "forgetting": round(float(acc_A_base-a_A),4)}})
best_lam = min(lambda_results[1:], key=lambda x: x["forgetting"])
improvement = lambda_results[0]["forgetting"] - best_lam["forgetting"]
score = min(1.0, 0.2 + max(0, improvement) * 2.5)
_result = {{"action": "ewc_retention", "acc_A_base": round(float(acc_A_base),3),
            "conflict_rate": round(float(conflict_rate),3),
            "lambda_sweep": lambda_results, "best_lambda": best_lam,
            "retention_improvement": round(float(improvement),4), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"ewc_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.5:
            best = res.get("best_lambda", {})
            tracker.record(
                f"Ciclo {cycle}: EWC λ={best.get('lambda')} reduce olvido "
                f"{res.get('retention_improvement',0):.3f} "
                f"(acc_A={best.get('acc_A')}, acc_B={best.get('acc_B')})",
                domain="continual", confidence=0.72,
                evidence=f"sweep λ en {lam_base:.2f}–{lam_base*10:.2f}",
            )
        return score

    elif action == "fisher_geometry":
        code = f"""
import numpy as np
np.random.seed({cycle * 11 + 5})
# FIM diagonal: identificar qué parámetros son críticos para cada tarea
# Hipótesis: parámetros con alta FIM en tarea A y baja en tarea B son los más
# relevantes para EWC — una zona "plana" en B pero "empinada" en A
d_in = 20; n = 400
results = []
for scenario in range(5):
    W = np.random.randn(2, d_in) * 0.1
    X_A = np.random.randn(n, d_in)
    mask_A = np.zeros(d_in); mask_A[:d_in//2] = 1.0
    y_A = (X_A @ mask_A > 0).astype(float)
    X_B = np.random.randn(n, d_in)
    mask_B = np.zeros(d_in); mask_B[d_in//2:] = 1.0
    y_B = (X_B @ mask_B > 0).astype(float)
    def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def train_simple(X, y):
        W_ = W.copy()
        for _ in range(150):
            p = sigmoid(X@W_.T)[:,0]; e = p-y
            W_[0] -= 0.07*(e[:,None]*X).mean(0)
        return W_
    def fisher_diag(X, y, W_):
        p = sigmoid(X@W_.T)[:,0]; e = p-y
        return ((e[:,None]*X)**2).mean(0)
    W_A = train_simple(X_A, y_A)
    W_B = train_simple(X_B, y_B)
    F_A = fisher_diag(X_A, y_A, W_A)
    F_B = fisher_diag(X_B, y_B, W_B)
    # Parámetros que son críticos para A pero no para B
    critical_A = F_A > np.percentile(F_A, 70)
    safe_B     = F_B < np.percentile(F_B, 30)
    safe_to_protect = int((critical_A & safe_B).sum())
    overlap = float((critical_A & ~safe_B).sum()) / (d_in + 1e-9)
    results.append({{"scenario": scenario, "safe_to_protect": safe_to_protect,
                     "overlap": round(overlap, 3),
                     "mean_F_A": round(float(F_A.mean()),4),
                     "mean_F_B": round(float(F_B.mean()),4)}})
mean_safe = sum(r["safe_to_protect"] for r in results) / len(results)
mean_overlap = sum(r["overlap"] for r in results) / len(results)
score = min(1.0, 0.3 + (mean_safe/d_in)*0.5 + (1-mean_overlap)*0.2)
_result = {{"action": "fisher_geometry", "n_scenarios": len(results),
            "mean_safe_params": round(float(mean_safe),2),
            "mean_overlap": round(float(mean_overlap),4),
            "results": results, "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"fisher_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.55:
            tracker.record(
                f"Ciclo {cycle}: FIM identifica {res.get('mean_safe_params',0):.1f} "
                f"parámetros seguros (críticos A, planos en B), overlap={res.get('mean_overlap',1):.3f}",
                domain="continual", confidence=0.68,
                evidence=f"d_in=20, 5 escenarios",
            )
        return score

    elif action == "task_similarity":
        code = f"""
import numpy as np
np.random.seed({cycle * 19 + 11})
# Hipótesis: mayor similitud entre tareas → menor olvido catastrófico
# Similitud controlada: w_B = sim*w_A + (1-sim)*w_ortho
# sim=1 → tareas idénticas (0 olvido), sim=0 → tareas ortogonales (máximo olvido)
d_in = 30; n = 600
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def train_sgd(X, y, W_init, lr=0.04, steps=400):
    W = W_init.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        W[0] -= lr*(e[:,None]*X).mean(0)
    return W
task_pairs = []
for pair_idx in range(8):
    sim = np.random.uniform(0.0, 1.0)
    w_A = np.random.randn(d_in); w_A /= np.linalg.norm(w_A)
    w_orth = np.random.randn(d_in)
    w_orth -= np.dot(w_orth, w_A)*w_A; w_orth /= np.linalg.norm(w_orth)+1e-9
    # Para sim < 0.5: w_B tiene componente opuesta (olvido severo esperado)
    sign = 1.0 if sim >= 0.5 else -1.0
    w_B = sim * w_A + (1-sim) * sign * w_orth
    w_B /= np.linalg.norm(w_B)+1e-9
    X = np.random.randn(n, d_in)
    y_A = (X @ w_A > 0).astype(float)
    y_B = (X @ w_B > 0).astype(float)
    W_init = np.random.randn(1, d_in) * 0.05
    # Gradientes iniciales
    g_A = ((sigmoid(X@W_init.T)[:,0]-y_A)[:,None]*X).mean(0)
    g_B = ((sigmoid(X@W_init.T)[:,0]-y_B)[:,None]*X).mean(0)
    grad_cos = float(np.dot(g_A,g_B)/(np.linalg.norm(g_A)*np.linalg.norm(g_B)+1e-9))
    # Entrena A, luego fine-tune en B, mide retención en A
    W_A  = train_sgd(X, y_A, W_init)
    acc_A_before = float(np.mean((sigmoid(X@W_A.T)[:,0]>0.5)==y_A))
    W_AB = train_sgd(X, y_B, W_A, steps=300)
    acc_A_after  = float(np.mean((sigmoid(X@W_AB.T)[:,0]>0.5)==y_A))
    forgetting = acc_A_before - acc_A_after
    task_pairs.append({{"sim": round(sim,3), "grad_cos": round(grad_cos,4),
                        "forgetting": round(forgetting,4),
                        "acc_A_before": round(acc_A_before,3),
                        "acc_A_after": round(acc_A_after,3)}})
sims = [r["sim"] for r in task_pairs]
fgts = [r["forgetting"] for r in task_pairs]
corr = float(np.corrcoef(sims, [-f for f in fgts])[0,1])  # +1 = más sim → menos olvido
score = min(1.0, 0.3 + max(0, corr) * 0.7)
_result = {{"action": "task_similarity", "n_pairs": len(task_pairs),
            "sim_forgetting_corr": round(corr,4),
            "mean_forgetting": round(float(np.mean(fgts)),4),
            "pairs": task_pairs, "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"tasksim_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.5:
            tracker.record(
                f"Ciclo {cycle}: correlación similitud-olvido = {res.get('similarity_forgetting_correlation',0):.3f} "
                f"(mayor similitud → menor olvido)",
                domain="continual", confidence=0.65,
                evidence=f"6 pares de tareas con similitud controlada",
            )
        return score

    else:  # replay_vs_finetune
        replay_frac = 0.1 + (cycle % 8) * 0.05
        code = f"""
import numpy as np
np.random.seed({cycle * 23 + 13})
# Replay buffer vs fine-tuning puro — tareas conflictivas (mismo X, etiquetas opuestas)
d_in = 30; n = 500; n_replay = int(n * {replay_frac:.2f})
def sigmoid(x): return 1/(1+np.exp(-np.clip(x,-20,20)))
def acc(W, X, y): return float(np.mean((sigmoid(X@W.T)[:,0]>0.5)==y))
def train(X, y, W_init, lr=0.04, steps=350):
    W = W_init.copy()
    for _ in range(steps):
        p = sigmoid(X@W.T)[:,0]; e = p-y
        W[0] -= lr*(e[:,None]*X).mean(0)
    return W
W_init = np.random.randn(2, d_in) * 0.05
# Mismo dataset X — etiquetas A y B basadas en features distintos pero solapados
w_A = np.random.randn(d_in); w_A[d_in//2:] *= 0.1
w_B = w_A.copy(); w_B[:d_in//2] *= -1.2  # conflicto: signo opuesto en features A
X_shared = np.random.randn(n, d_in)
y_A = (X_shared @ w_A > 0).astype(float)
y_B = (X_shared @ w_B > 0).astype(float)
W_A = train(X_shared, y_A, W_init, steps=500)
acc_A_base = acc(W_A, X_shared, y_A)
# Fine-tuning puro en B (sin replay)
W_noreplay = train(X_shared, y_B, W_A)
# Fine-tuning en B CON replay de muestras A
replay_idx = np.random.choice(n, n_replay, replace=False)
X_mix = np.vstack([X_shared, X_shared[replay_idx]])
y_mix = np.concatenate([y_B, y_A[replay_idx]])
W_replay = train(X_mix, y_mix, W_A)
acc_A_norepay = acc(W_noreplay, X_shared, y_A)
acc_A_replay  = acc(W_replay,   X_shared, y_A)
acc_B_replay  = acc(W_replay,   X_shared, y_B)
retention_gain = acc_A_replay - acc_A_norepay
score = min(1.0, 0.2 + max(0, retention_gain) * 3.0)
_result = {{"action": "replay_vs_finetune", "replay_frac": round({replay_frac:.2f},3),
            "acc_A_base": round(float(acc_A_base),3),
            "acc_A_norepay": round(float(acc_A_norepay),3),
            "acc_A_replay": round(float(acc_A_replay),3),
            "acc_B_replay": round(float(acc_B_replay),3),
            "retention_gain": round(float(retention_gain),4),
            "replay_helps": bool(retention_gain > 0.02), "score": round(score,4)}}
"""
        r   = executor.run(code, label=f"replay_{cycle}")
        res = r.get("result") or {}
        log["executor_results"].append(res)
        score = float(res.get("score", 0.0))
        if score > 0.45:
            tracker.record(
                f"Ciclo {cycle}: replay {replay_frac:.0%} mejora retención de A "
                f"{res.get('retention_gain',0):.3f} sobre fine-tune puro",
                domain="continual", confidence=0.66,
                evidence=f"replay_frac={replay_frac:.2f}",
            )
        return score


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def execute_action(action: str, problem: str, cycle: int,
                   executor: DirectExecutor,
                   tracker: HypothesisTracker) -> tuple[float, dict]:
    log = {"action": action, "cycle": cycle, "executor_results": []}
    if problem == "causal":
        score = _causal_action(action, cycle, executor, tracker, log)
    else:
        score = _continual_action(action, cycle, executor, tracker, log)
    tracker.record_attempt(description=action, result=f"score={score:.3f}", domain=problem)
    log["progress_score"] = score
    return score, log


# ══════════════════════════════════════════════════════════════════════════════
# Guardado
# ══════════════════════════════════════════════════════════════════════════════

class _Enc(json.JSONEncoder):
    def default(self, o):
        try:
            return float(o)
        except Exception:
            return str(o)

import json as _json

def _save_final(session_id, all_logs, tracker, meter, tool_reg,
                oracle, pool, memory, inventor, verbose=True):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f"results_v010_{session_id}.json"
    try:
        data = {
            "version":   VERSION,
            "session_id":session_id,
            "timestamp": ts,
            "tracker":   tracker.self_report(),
            "meter":     meter.summary(),
            "tools":     tool_reg.summary(),
            "oracle":    oracle.summary(),
            "pool":      pool.summary(),
            "memory":    memory.summary(),
            "inventor":  inventor.summary(),
            "logs":      all_logs[-50:],
        }
        with open(path, "w", encoding="utf-8") as f:
            _json.dump(data, f, cls=_Enc, indent=2, ensure_ascii=False)
        if verbose:
            print(f"\n  [Guardado] → {path}")
    except Exception as e:
        if verbose:
            print(f"\n  [Error guardado] {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", choices=["causal","continual"], default="causal")
    parser.add_argument("--hours",   type=float, default=4.0)
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--resume",  action="store_true")
    args = parser.parse_args()

    api_key = os.getenv("ANTHROPIC_API_KEY")
    deadline = time.time() + args.hours * 3600

    session_id = datetime.now().strftime(f"v010_{args.problem}_%Y%m%d_%H%M")
    Path("results").mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  agente vX {VERSION}  |  problema: {args.problem}  |  {args.hours}h")
    print(f"  sin sandbox — DirectExecutor con entorno Python completo")
    print(f"{'='*70}\n")

    # ── Setup ────────────────────────────────────────────────────────────────
    executor     = DirectExecutor(timeout_sec=60.0)
    sub_executor = DirectExecutor(timeout_sec=50.0)  # exclusivo para subagentes

    oracle  = OracleClient(api_key, budget_usd=ORACLE_BUDGET)
    pool    = SubagentPool(sub_executor, max_workers=3)
    tracker = HypothesisTracker(subagent_pool=pool, oracle_client=oracle,
                                sandbox=executor)
    meter   = SatisfactionMeter(stagnation_window=5, pivot_threshold=10)

    library = ResonantLibrary(f"library_{args.problem}_{args.seed}.pkl")
    maestro = SecondOrderAgent(BASE_CONFIG, library, seed=args.seed)
    tool_reg = ToolRegistry()

    # ── ResearchMemory ───────────────────────────────────────────────────────
    mem_path = f"results/memory_{args.problem}_{session_id}.json"
    memory   = ResearchMemory(domain=args.problem, persist_path=mem_path)

    # ── ActionInventor ───────────────────────────────────────────────────────
    inventor = ActionInventor(oracle)

    # ── Corpus ──────────────────────────────────────────────────────────────
    print(f"[Corpus] Cargando para '{args.problem}'...")
    all_tools = set()
    for corpus_name in list_ml_corpus_for_domain(args.problem):
        corpus = get_ml_corpus(corpus_name)
        for td in corpus.get("initial_tools", []):
            if td["name"] not in all_tools:
                ok, _ = tool_reg.invent(td["name"], td["code"], td["description"])
                if ok:
                    all_tools.add(td["name"])
                    print(f"  [Tool] {td['name']}")
    print(f"  Tools: {list(all_tools)}\n")

    # ── Acciones ─────────────────────────────────────────────────────────────
    actions = PROBLEM_ACTIONS[args.problem]

    # ── Estado ───────────────────────────────────────────────────────────────
    global_cycle  = 0
    perspective   = "mathematician"
    all_logs: list = []
    last_oracle   = 0.0
    crisis_count  = 0

    # Stagnation por acción: cuenta ciclos consecutivos sin cambio (< 0.02)
    _action_stagnation: dict[str, int] = {a: 0 for a in actions}
    # Limbo: acciones dormantes por score fijo; reviven cuando sube el promedio global
    _dormant: set[str] = set()
    _dormant_avg: dict[str, float] = {}  # satisfaction cuando entró en Limbo

    if args.resume:
        import glob as _glob
        # Busca el JSON más reciente del mismo problema
        candidates = sorted(
            _glob.glob(f"results_v010_v010_{args.problem}_*.json"),
            reverse=True,
        )
        if candidates:
            _resume_file = candidates[0]
            try:
                with open(_resume_file, encoding="utf-8") as _rf:
                    _prev = json.load(_rf)
                _prev_logs = _prev.get("logs", [])
                if _prev_logs:
                    global_cycle = max(l.get("cycle", 0) for l in _prev_logs)
                    # Reconstruir historial de scores por acción
                    for _l in _prev_logs:
                        _a = _l.get("action")
                        _s = _l.get("progress_score", 0.0)
                        if _a in _action_score_hist:
                            _action_score_hist[_a].append(_s)
                            if len(_action_score_hist[_a]) > 20:
                                _action_score_hist[_a].pop(0)
                _prev_meter = _prev.get("meter", {})
                print(f"[Resume] archivo={_resume_file}")
                print(f"[Resume] ciclo_anterior={global_cycle}  satisfaccion={_prev_meter.get('satisfaction','?')}")
                hist_loaded = sum(1 for a in actions if _action_score_hist[a])
                print(f"[Resume] historial reconstruido para {hist_loaded}/{len(actions)} acciones")
            except Exception as _e:
                print(f"[Resume] Error cargando {_resume_file}: {_e}")
                global_cycle = 0
        else:
            print(f"[Resume] No se encontró results_v010_v010_{args.problem}_*.json — inicio frío")

    print(f"[Inicio] perspectiva={perspective}  actions={actions}\n")

    # Mapa de tags por acción
    ACTION_TAGS = {
        "backdoor_adjustment": ["causal", "backdoor", "confunder", "ajuste"],
        "do_calculus_test":    ["causal", "do-calculus", "dag", "identificacion"],
        "irm_vs_erm":          ["causal", "irm", "invarianza", "entornos"],
        "causal_discovery":    ["causal", "descubrimiento", "esqueleto", "dag"],
        "counterfactual_bounds":["causal", "contrafactual", "cotas", "tian-pearl"],
        "gradient_interference":["continual", "gradiente", "interferencia", "coseno"],
        "ewc_retention":       ["continual", "ewc", "fisher", "retencion"],
        "fisher_geometry":     ["continual", "fisher", "geometria", "parametros"],
        "task_similarity":     ["continual", "similitud", "tareas", "olvido"],
        "replay_vs_finetune":  ["continual", "replay", "memoria", "retencion"],
    }

    # Mínimo de segundos entre ciclos — evita acumular miles de entradas idénticas
    MIN_CYCLE_SEC = 15.0
    # Umbral de novedad para indexar en memoria — solo si el score difiere > 0.08
    # de la media de las últimas 5 entradas de esa misma acción
    MEM_NOVELTY_THRESHOLD = 0.12
    # Historial de scores recientes por acción (para calcular novedad)
    _action_score_hist: dict[str, list[float]] = {a: [] for a in actions}

    # ── Bucle principal ───────────────────────────────────────────────────────
    try:
        while time.time() < deadline:
            t_cycle_start = time.time()
            global_cycle += 1
            t_rem = (deadline - t_cycle_start) / 3600

            print(f"\n{'─'*70}")
            print(f"  CICLO {global_cycle}  |  {t_rem:.2f}h  |  {perspective}")
            print(f"  Satisfacción: {meter.satisfaction_score():.2f}  "
                  f"mem: {memory.summary()['total']}  "
                  f"crisis: {crisis_count}")
            print(f"{'─'*70}")

            # ── (A) Elegir acción (salta dormantes) ──────────────────────────
            emphasis = PERSPECTIVE_EMPHASIS.get(perspective, list(range(5)))
            active_indices = [i for i in emphasis if actions[i] not in _dormant]
            if not active_indices:
                _dormant.clear()
                _dormant_avg.clear()
                active_indices = list(emphasis)
                print("  [Limbo] Todas dormantes — reviviendo todas")
            action_idx = active_indices[global_cycle % len(active_indices)]
            action     = actions[action_idx]

            # ── (B) Ejecutar acción ───────────────────────────────────────────
            score, attack_log = execute_action(
                action, args.problem, global_cycle, executor, tracker
            )
            all_logs.append(attack_log)
            for res in attack_log["executor_results"]:
                print(f"  [{action}] {str(res)[:140]}")

            # ── (C) Indexar en ResearchMemory solo si hay novedad ────────────────
            tags = ACTION_TAGS.get(action, [action, args.problem])
            hist = _action_score_hist.setdefault(action, [])
            _is_novel = (
                len(hist) < 3
                or abs(score - sum(hist[-5:]) / len(hist[-5:])) > MEM_NOVELTY_THRESHOLD
            )
            if _is_novel:
                mem_id = memory.index(
                    summary = f"{action}: score={score:.3f}, {str(attack_log['executor_results'][:1])[:120]}",
                    score   = score,
                    cycle   = global_cycle,
                    source  = "main",
                    tags    = tags,
                    result  = attack_log["executor_results"][0] if attack_log["executor_results"] else {},
                )
            else:
                mem_id = "(skip-redundante)"
            hist.append(score)
            if len(hist) > 20:
                hist.pop(0)

            # ── Stagnation por acción ─────────────────────────────────────────
            if len(hist) >= 3 and all(abs(hist[-1-i] - hist[-1]) < 0.02 for i in range(1, min(3, len(hist)))):
                _action_stagnation[action] = _action_stagnation.get(action, 0) + 1
            else:
                _action_stagnation[action] = 0

            # Limbo: acción entra si lleva 15 ciclos sin cambio
            if _action_stagnation.get(action, 0) >= 15 and action not in _dormant:
                _dormant.add(action)
                _dormant_avg[action] = meter.satisfaction_score()
                print(f"  [Limbo] '{action}' dormante "
                      f"(stagnation={_action_stagnation[action]}, score≈{score:.3f})")

            # Revival: acción dormante vuelve si el promedio global subió 0.05
            for _da in list(_dormant):
                if _da != action and meter.satisfaction_score() > _dormant_avg.get(_da, 0.0) + 0.05:
                    _dormant.discard(_da)
                    _action_stagnation[_da] = 0
                    print(f"  [Limbo/revival] '{_da}' reactivada")

            # ── (D) Medidor de satisfacción ───────────────────────────────────
            state = meter.update(score)
            print(f"  [Meter] score={score:.3f}  state={state.value}  mem_id={mem_id}")

            # ── (E) Recoger subagentes ────────────────────────────────────────
            completed = pool.collect_completed()
            if completed:
                tracker.absorb_subagent_results(completed)
                for r in completed:
                    memory.index(
                        summary = f"Sub {r.task_id}: {r.hypothesis[:80]}",
                        score   = r.score,
                        cycle   = global_cycle,
                        source  = "subagent",
                        tags    = ["verification", r.domain],
                        result  = r.result,
                    )
                    print(f"  [Sub/{r.task_id}] score={r.score:.2f} — {r.hypothesis[:55]}")

            # ── (F) Verificación activa de hipótesis cada 3 ciclos ────────────
            if global_cycle % 3 == 0:
                for h in list(tracker._hyps):
                    tracker._maybe_verify(h)

            # ── (G) Creative crisis cada 8 ciclos ─────────────────────────────
            if global_cycle % 8 == 0:
                crisis = tracker.check_creative_crisis(research_memory=memory)
                if crisis:
                    crisis_count += 1
                    print(f"\n  [CreativeCrisis #{crisis_count}]")
                    print(f"    H tensión: {crisis['h1'][:60]}")
                    print(f"    Síntesis:  {crisis['synthesis_direction'][:80]}")

            # ── (H) Rotación de perspectiva ───────────────────────────────────
            if global_cycle % 8 == 0:
                opts = [p for p in PERSPECTIVE_EMPHASIS if p != perspective]
                perspective = opts[meter._pivot_count % len(opts)]
                meter._pivot_count += 1
                print(f"\n  [Rotación] → {perspective}")
            elif meter.needs_pivot():
                perspective = meter.next_perspective(perspective, list(PERSPECTIVE_EMPHASIS.keys()))
                print(f"\n  [Pivot] → {perspective}")

            # ── (I) Oráculo estratégico ───────────────────────────────────────
            now = time.time()
            _any_action_stuck = any(v >= 10 for v in _action_stagnation.values())
            if (oracle.available
                    and (meter._stagnation_count >= 5 or _any_action_stuck)
                    and (now - last_oracle) > 900):
                recent_scores = {a: {"avg": round(
                    sum(l["progress_score"] for l in all_logs[-15:] if l["action"]==a)
                    / max(1, sum(1 for l in all_logs[-15:] if l["action"]==a)), 3
                ), "n": sum(1 for l in all_logs[-15:] if l["action"]==a)} for a in actions}
                failed = [l["action"] for l in all_logs[-8:] if l["progress_score"] < 0.25]
                result = tracker.request_oracle_direction(args.problem, recent_scores, failed)
                if result:
                    last_oracle = now
                    print(f"  [Oracle] {result.get('direction','')[:90]}")

            # ── (J) ActionInventor ────────────────────────────────────────────
            if (global_cycle % 15 == 0
                    and inventor.can_invent
                    and sum(all_scores_recent := [l["progress_score"] for l in all_logs[-8:]]) / max(1, len(all_scores_recent)) < 0.50):
                avg_recent = sum(all_scores_recent) / len(all_scores_recent)
                print(f"\n  [ActionInventor] avg_reciente={avg_recent:.2f} < 0.50, inventando...")
                tool_sum = "\n".join(f"  {n}" for n in all_tools) or "(ninguna)"
                invention = inventor.try_invent(
                    problem        = args.problem,
                    n_cycles       = global_cycle,
                    tool_summary   = tool_sum,
                    failed_actions = failed if 'failed' in dir() else [],
                    memory         = memory,
                )
                if invention:
                    ok, _ = tool_reg.invent(invention["tool_name"], invention["code"],
                                            invention["description"])
                    if ok:
                        all_tools.add(invention["tool_name"])
                        memory.index(
                            summary = f"Invención: {invention['tool_name']} — {invention['description']}",
                            score   = 0.7, cycle=global_cycle,
                            source  = "invention",
                            tags    = ["invention", "new_tool", args.problem],
                        )
                        print(f"  [ActionInventor] ✓ {invention['tool_name']} registrada")

            # ── (K) Memory snapshot cada 10 ciclos ────────────────────────────
            if global_cycle % 10 == 0:
                ms = memory.summary()
                print(f"\n  [Memory] total={ms['total']} avg={ms['avg_score']:.3f} "
                      f"by_source={ms['by_source']}")
                top3 = memory.to_corpus_knowledge(top_k=3)
                print(f"  [Memory/top3]:\n{top3}")

            # ── (L) Guardado periódico ────────────────────────────────────────
            if global_cycle % 25 == 0:
                _save_final(session_id, all_logs, tracker, meter, tool_reg,
                            oracle, pool, memory, inventor, verbose=False)

            # ── (M) Pausa mínima por ciclo ────────────────────────────────────
            elapsed_cycle = time.time() - t_cycle_start
            if elapsed_cycle < MIN_CYCLE_SEC:
                time.sleep(MIN_CYCLE_SEC - elapsed_cycle)

    except KeyboardInterrupt:
        print("\n  [Interrupción manual]")
    finally:
        pool.shutdown()
        _save_final(session_id, all_logs, tracker, meter, tool_reg,
                    oracle, pool, memory, inventor)
