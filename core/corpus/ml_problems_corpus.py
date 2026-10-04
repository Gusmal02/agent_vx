"""
core/corpus/ml_problems_corpus.py — Corpus para problemas ML abiertos (v0.1.0)

Problemas:
  causal     — Razonamiento causal vs correlación estadística (Pearl, do-calculus)
  continual  — Olvido catastrófico / aprendizaje continuo (EWC, Fisher, replay)

Cada corpus incluye:
  - knowledge: base teórica para el oracle
  - initial_tools: funciones tool_fn ejecutables con DirectExecutor
"""

from typing import Dict, List


# ── Corpus Causalidad ─────────────────────────────────────────────────────────

CORPUS_CAUSAL = {
    "name": "causal_specialist",
    "domain": "causal",
    "knowledge": """
Razonamiento Causal vs Correlación Estadística (Pearl, 2000-2018)

El Problema Central:
  ML estándar estima P(Y|X) — distribución observacional.
  No puede responder P(Y|do(X)) — intervención directa sobre X.
  Diferencia clave: en P(Y|X) hay backdoor paths (confounders).
                    en P(Y|do(X)) se cortan los backdoor paths.

Formalismo (do-calculus de Pearl):
  DAG: G = (V, E) — grafo acíclico dirigido de mecanismos causales
  SCM: V_i = f_i(PA_i, U_i) — ecuaciones estructurales
  Backdoor criterion: Z bloquea todos los backdoor paths de X → Y
  Ajuste: P(Y|do(X)) = Σ_z P(Y|X,Z) P(Z)

Algoritmos de descubrimiento causal:
  PC algorithm: esqueleto por tests de independencia condicional
  FCI: extensión para variables latentes
  LiNGAM: modelos lineales no gaussianos → identifica dirección causal

Barreras conocidas:
  - Observational equivalence: múltiples DAGs producen misma distribución
  - Causal faithfulness: puede violarse con cancelaciones exactas
  - Infinite data assumption: tests de independencia necesitan n → ∞

Estado 2024:
  - IRM (Invariant Risk Minimization, Arjovsky 2019): busca representaciones
    invariantes entre entornos como proxy de causalidad
  - DiffCausal: aprendizaje diferenciable de estructura causal
  - Clave abierta: ¿cómo hacer causal discovery en alta dimensión con ruido?
""",
    "initial_tools": [
        {
            "name": "backdoor_adjustment_test",
            "description": "Genera DAG aleatorio con confounder, mide sesgo observacional vs ajuste backdoor",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000
    np.random.seed(seed)

    # DAG: Z → X → Y,  Z → Y (Z es confounder)
    n = 2000
    # Parámetros del SCM
    alpha_zx = 0.7 + np.random.uniform(-0.3, 0.3)
    alpha_zy = 0.5 + np.random.uniform(-0.2, 0.2)
    alpha_xy = 0.6 + np.random.uniform(-0.3, 0.3)

    Z = np.random.randn(n)
    X = alpha_zx * Z + np.random.randn(n) * 0.5
    Y = alpha_xy * X + alpha_zy * Z + np.random.randn(n) * 0.5

    # Estimación observacional (sesgada): E[Y|X] sin ajuste
    cov_xy = np.cov(X, Y)[0,1]
    var_x  = np.var(X)
    beta_naive = cov_xy / (var_x + 1e-9)

    # Estimación causal (ajuste backdoor por Z): regresión Y ~ X + Z
    A = np.column_stack([X, Z, np.ones(n)])
    beta_causal, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
    beta_adjusted = beta_causal[0]  # coef de X ajustado por Z

    bias = abs(beta_naive - alpha_xy)
    bias_adj = abs(beta_adjusted - alpha_xy)
    bias_reduction = max(0.0, 1.0 - bias_adj / (bias + 1e-9))
    score = min(1.0, 0.3 + bias_reduction * 0.7)

    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime": "backdoor_adjustment",
        "true_causal_effect": round(float(alpha_xy), 4),
        "naive_estimate":     round(float(beta_naive), 4),
        "adjusted_estimate":  round(float(beta_adjusted), 4),
        "bias_naive":         round(float(bias), 4),
        "bias_adjusted":      round(float(bias_adj), 4),
        "bias_reduction":     round(float(bias_reduction), 4),
        "score":              round(score, 4),
    }
""",
        },
        {
            "name": "irm_vs_erm_test",
            "description": "Compara IRM vs ERM en múltiples entornos con confounders distintos",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000
    np.random.seed(seed)

    # Dos entornos con diferente correlación espuria X2 → Y
    # El mecanismo causal verdadero es X1 → Y
    results = []
    for gamma in [0.9, -0.9, 0.0]:  # entorno de test: sin correlación espuria
        n = 500
        X1 = np.random.randn(n)  # causa real
        noise_env = gamma * np.random.randn(n)  # correlación espuria varía por entorno
        X2 = noise_env + np.random.randn(n) * 0.1
        Y  = 2.0 * X1 + np.random.randn(n) * 0.3

        X = np.column_stack([X1, X2])
        # ERM: OLS sobre ambas features
        A = np.column_stack([X, np.ones(n)])
        beta_erm, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
        y_pred_erm = A @ beta_erm
        mse_erm = float(np.mean((Y - y_pred_erm)**2))

        # IRM proxy: usar solo X1 (el invariante)
        A1 = np.column_stack([X1, np.ones(n)])
        beta_irm, _, _, _ = np.linalg.lstsq(A1, Y, rcond=None)
        y_pred_irm = A1 @ beta_irm
        mse_irm = float(np.mean((Y - y_pred_irm)**2))

        results.append({"gamma": round(gamma, 2),
                        "mse_erm": round(mse_erm, 4),
                        "mse_irm": round(mse_irm, 4),
                        "irm_wins": mse_irm < mse_erm})

    # En entorno test (gamma=0): IRM debería ganar más
    test_result = results[2]
    irm_advantage = results[0]["mse_erm"] - results[0]["mse_irm"]
    score = min(1.0, 0.3 + max(0.0, irm_advantage) * 2)

    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime": "irm_vs_erm",
        "environments": results,
        "irm_advantage_env1": round(float(irm_advantage), 4),
        "score": round(score, 4),
    }
""",
        },
    ],
}


# ── Corpus Olvido Catastrófico ─────────────────────────────────────────────────

CORPUS_CONTINUAL = {
    "name": "continual_specialist",
    "domain": "continual",
    "knowledge": """
Olvido Catastrófico / Aprendizaje Continuo (McCloskey & Cohen 1989, Kirkpatrick 2017)

El Problema Central:
  Al optimizar θ para tarea B: ∇L_B(θ) desplaza θ fuera del mínimo de tarea A.
  Interferencia de gradientes: ∇L_A · ∇L_B < 0 → aprendizaje B destruye A.
  Plasticidad vs Estabilidad: el sistema no puede tener ambas sin mecanismo especial.

Formalismo:
  EWC (Elastic Weight Consolidation, Kirkpatrick 2017):
    L(θ) = L_B(θ) + λ/2 · Σ_i F_i (θ_i - θ*_{A,i})²
    F_i = E[( ∂/∂θ_i log p(y|x,θ) )²]  ← diagonal de Fisher Information Matrix

  Intuición: F_i mide cuánto depende la tarea A del parámetro i.
  Parámetros con F_i alto → "importantes para A" → costoso cambiarlos para B.

GEM (Gradient Episodic Memory, Lopez-Paz 2017):
  Proyectar ∇L_B en el hiperplano ortogonal a los gradientes pasados:
    g̃ = argmin ||g̃ - g||²  s.t. g̃ · ∇L_A ≥ 0

Progressive Neural Networks (Rusu 2016):
  Columnas separadas por tarea + conexiones laterales → sin interferencia posible.
  Costo: O(T²) parámetros para T tareas.

Estado 2024:
  - Packeт Network: sparsificación para aislar subredes por tarea
  - DualPrompt: prompts por tarea en transformers pretrained
  - Clave abierta: encontrar λ óptimo de EWC sin validation set de tarea antigua.
  - Hipótesis no verificada: ¿existe un λ óptimo universal o depende de la geometría
    del loss landscape de cada par de tareas?
""",
    "initial_tools": [
        {
            "name": "gradient_interference_measure",
            "description": "Mide el coseno entre gradientes de dos tareas para cuantificar interferencia",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000
    np.random.seed(seed)

    # Red neuronal lineal simple: W ∈ R^(d_out x d_in)
    d_in, d_out, n = 20, 10, 200
    W = np.random.randn(d_out, d_in) * 0.1

    # Tarea A: clasificación con patrón "horizontal" (primeros d_in//2 features)
    X_A = np.random.randn(n, d_in)
    X_A[:, d_in//2:] *= 0.05  # las segundas features son ruido en tarea A
    y_A = (X_A[:, :d_in//2].sum(axis=1) > 0).astype(float)

    # Tarea B: clasificación con patrón "vertical" (segundas d_in//2 features)
    X_B = np.random.randn(n, d_in)
    X_B[:, :d_in//2] *= 0.05  # las primeras features son ruido en tarea B
    y_B = (X_B[:, d_in//2:].sum(axis=1) > 0).astype(float)

    def compute_grad(X, y, W):
        logits = X @ W.T  # (n, d_out)
        probs  = 1 / (1 + np.exp(-logits[:, 0]))
        err    = probs - y
        grad_W = (err[:, None] * X).mean(axis=0)  # (d_in,) grad del primer output
        return grad_W

    g_A = compute_grad(X_A, y_A, W)
    g_B = compute_grad(X_B, y_B, W)

    cos_sim = float(np.dot(g_A, g_B) / (np.linalg.norm(g_A) * np.linalg.norm(g_B) + 1e-9))
    interference = max(0.0, -cos_sim)   # positivo si los gradientes apuntan en dirección opuesta
    alignment    = max(0.0,  cos_sim)

    # Score: alto si la interferencia es cuantificable y medible (ni 0 ni 1 trivial)
    score = min(1.0, 0.3 + interference * 0.7) if interference > 0.05 else 0.35

    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime":         "gradient_interference",
        "cosine_sim":     round(cos_sim, 4),
        "interference":   round(interference, 4),
        "alignment":      round(alignment, 4),
        "tasks_conflict": bool(cos_sim < 0),
        "score":          round(score, 4),
    }
""",
        },
        {
            "name": "ewc_retention_test",
            "description": "Entrena tarea A, luego B con/sin EWC y mide retención de A",
            "code": """
def tool_fn(omega, candidates, context):
    import numpy as np
    seed = int(abs(float(omega.flat[0])) * 1000) % 10000
    np.random.seed(seed)

    # Red: capa única W ∈ R^(2 x d_in)
    d_in = 15
    n    = 300

    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
    def bce(pred, y): return -np.mean(y*np.log(pred+1e-9) + (1-y)*np.log(1-pred+1e-9))
    def accuracy(W, X, y):
        pred = sigmoid(X @ W.T)[:, 0]
        return float(np.mean((pred > 0.5) == y))

    # Tarea A: primeras d_in//2 features importan
    X_A = np.random.randn(n, d_in)
    y_A = (X_A[:, :d_in//2].mean(axis=1) > 0).astype(float)

    # Tarea B: últimas d_in//2 features importan
    X_B = np.random.randn(n, d_in)
    y_B = (X_B[:, d_in//2:].mean(axis=1) > 0).astype(float)

    def train(W, X, y, lr=0.05, steps=100, ewc_W=None, ewc_F=None, lam=0.0):
        W = W.copy()
        for _ in range(steps):
            pred = sigmoid(X @ W.T)[:, 0]
            err  = pred - y
            grad = (err[:, None] * X).mean(axis=0)  # (d_in,)
            W_grad = np.zeros_like(W)
            W_grad[0] = grad
            if ewc_W is not None and lam > 0:
                W_grad += lam * ewc_F * (W - ewc_W)
            W -= lr * W_grad
        return W

    def fisher_diag(W, X, y):
        pred = sigmoid(X @ W.T)[:, 0]
        err  = pred - y
        grad_per_sample = err[:, None] * X  # (n, d_in)
        F = (grad_per_sample**2).mean(axis=0)
        F_W = np.zeros_like(W)
        F_W[0] = F
        return F_W

    W_init = np.random.randn(2, d_in) * 0.1

    # Entrenar en A
    W_A = train(W_init, X_A, y_A, steps=200)
    acc_A_base = accuracy(W_A, X_A, y_A)

    # Fisher de A
    F_A = fisher_diag(W_A, X_A, y_A)

    # Fine-tune en B sin EWC
    W_noewc = train(W_A, X_B, y_B, steps=150)
    acc_A_noewc = accuracy(W_noewc, X_A, y_A)
    forgetting_noewc = acc_A_base - acc_A_noewc

    # Fine-tune en B con EWC (lambda = función de omega)
    lam_val = 1.0 + abs(float(omega.flat[0])) * 5.0  # lambda dinámico
    W_ewc = train(W_A, X_B, y_B, steps=150,
                  ewc_W=W_A, ewc_F=F_A, lam=lam_val)
    acc_A_ewc  = accuracy(W_ewc,  X_A, y_A)
    acc_B_ewc  = accuracy(W_ewc,  X_B, y_B)
    forgetting_ewc = acc_A_base - acc_A_ewc

    retention_improvement = forgetting_noewc - forgetting_ewc
    score = min(1.0, 0.3 + max(0.0, retention_improvement) * 2.0)

    best = max(range(len(candidates)), key=lambda i: score)
    return candidates[best], {
        "regime":               "ewc_retention",
        "lambda":               round(float(lam_val), 3),
        "acc_A_base":           round(acc_A_base, 3),
        "acc_A_noewc":          round(acc_A_noewc, 3),
        "acc_A_ewc":            round(acc_A_ewc, 3),
        "acc_B_ewc":            round(acc_B_ewc, 3),
        "forgetting_noewc":     round(float(forgetting_noewc), 4),
        "forgetting_ewc":       round(float(forgetting_ewc), 4),
        "retention_improvement":round(float(retention_improvement), 4),
        "ewc_helps":            bool(retention_improvement > 0.02),
        "score":                round(score, 4),
    }
""",
        },
    ],
}


# ── Corpus ML extendido (base compartida) ─────────────────────────────────────

CORPUS_ML_BASE = {
    "name": "ml_tools_extended",
    "domain": "ml_base",
    "knowledge": """
Librerías disponibles en el ejecutor directo (sin restricciones):

numpy:   arrays, álgebra lineal, FFT, estadística
scipy:   optimize (fsolve, minimize), linalg, stats
sklearn: datasets, linear_model, metrics, neural_network, preprocessing
networkx: grafos, DAGs, algoritmos de grafos
torch:   redes neuronales, autograd, optimizadores (si está instalado)

Patrones de código en tool_fn con DirectExecutor:
  seed = int(abs(float(omega.flat[0])) * 1000) % 10000
  np.random.seed(seed)
  # ... análisis ...
  _result = {"metric": valor, "score": round(score, 4)}

Para causalidad:
  - np.linalg.lstsq para regresión lineal
  - networkx.DiGraph para DAGs
  - scipy.stats.pearsonr para tests de independencia

Para olvido catastrófico:
  - sklearn.neural_network.MLPClassifier para redes pequeñas
  - Matrices numpy para redes lineales simples (más rápido)
  - FIM diagonal = media de (∂log p / ∂θ_i)² sobre los datos de entrenamiento
""",
    "initial_tools": [],
}


# ── Registry ──────────────────────────────────────────────────────────────────

ALL_ML_CORPUS = {
    "causal_specialist":   CORPUS_CAUSAL,
    "continual_specialist":CORPUS_CONTINUAL,
    "ml_tools_extended":   CORPUS_ML_BASE,
}

DOMAIN_ML_CORPUS = {
    "causal":   ["causal_specialist",    "ml_tools_extended"],
    "continual":["continual_specialist", "ml_tools_extended"],
}


def get_ml_corpus(name: str) -> dict:
    return ALL_ML_CORPUS.get(name, {})


def list_ml_corpus_for_domain(domain: str) -> list:
    return DOMAIN_ML_CORPUS.get(domain, list(ALL_ML_CORPUS.keys()))
