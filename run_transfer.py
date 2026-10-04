"""
run_transfer.py — Prueba de transferencia v0.0.3

Experimento: ¿qué pasa cuando el agente vX enfrenta un entorno completamente
nuevo, habiendo aprendido CartPole-v1?

Tres condiciones:
  A: Agente fresco (sin biblioteca)         — baseline resonante
  B: Agente con biblioteca de CartPole       — prueba de transferencia
  C: Agente aleatorio                        — baseline mínimo

Entorno objetivo: Acrobot-v1
  - obs_dim=6 (vs CartPole 4) → W(6×3), distinto de CartPole W(4×3)
  - n_actions=3 (vs CartPole 2)
  - Reward: -1 por paso hasta resolver (swing-up del péndulo doble)
  - Terminación: punta del péndulo supera umbral de altura
  - Threshold "bueno": recompensa > -200 (resuelve en < 200 pasos)

Hipótesis a medir:
  H1: ¿Activa la biblioteca de CartPole el camino rápido en Acrobot?
      (¿hay overlap coseno ≥ 0.88 entre obs_omegas de ambos entornos?)
  H2: Si activa: ¿ayuda, perjudica o es neutral?
  H3: ¿El camino resonante converge más rápido sin la interferencia de la biblioteca?
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from copy import deepcopy

from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from gym_adapter import evaluar_agente, ActionEncoder, ObsEncoder
from library.store import ResonantLibrary


VERSION = "v0.0.3"
ENV_NAME = "Acrobot-v1"
MAX_REWARD = 500          # Acrobot: max pasos = 500
REWARD_BEST = -100        # Excelente: resuelve en ~100 pasos
REWARD_THRESHOLD = -200   # Umbral "bueno": resuelve en < 200 pasos
REWARD_WORST = -500       # Pésimo: no resolvió


def agente_aleatorio(env_name: str, n_episodios: int, seed: int = 42) -> dict:
    import gymnasium as gym
    env = gym.make(env_name)
    recompensas = []
    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep)
        total, done, pasos = 0.0, False, 0
        while not done and pasos < 500:
            obs, r, terminated, truncated, _ = env.step(env.action_space.sample())
            done = terminated or truncated
            total += r; pasos += 1
        recompensas.append(total)
    env.close()
    return {
        "episodios": recompensas,
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_max": float(np.max(recompensas)),
    }


def construir_agente(library=None, seed: int = 42) -> ProtoTissueAgentV2:
    return ProtoTissueAgentV2(
        N=50, M=5, K=3, seed=seed,
        stagnation_threshold=8,
        sim_top_k=2,
        filter_threshold=0.45,
        library=library,
        confidence_threshold=0.93,
        t_settle=5,
        warmup_steps=50,
        subconscious_update_freq=5,
        use_orch_or=False,   # desactivado en prueba de transferencia: ε-greedy cubre la exploración
    )


def run_condition(label: str, library, n_episodios: int, seed: int,
                  verbose: bool = True) -> dict:
    print(f"\n{'─'*60}")
    print(f"[{label}]")
    if library is not None:
        s = library.summary()
        print(f"  Biblioteca: {s['total']} registros "
              f"versiones={s['versions']} avg_reward={s['avg_reward']:.3f}")
    else:
        print("  Biblioteca: NINGUNA (agente fresco)")

    agente = construir_agente(library=library, seed=seed)

    resultado = evaluar_agente(
        agente,
        env_name=ENV_NAME,
        n_episodios=n_episodios,
        max_steps=500,
        verbose=verbose,
        seed=seed,
        reward_threshold=REWARD_THRESHOLD,
        version=VERSION,
        epsilon_start=0.6,
        epsilon_min=0.05,
        epsilon_decay_ep=n_episodios // 2,
        blend=0.6,
    )

    r = resultado["reporte_agente"]
    total_dec = r["n_fast_decisions"] + r["n_resonant_decisions"]
    fast_pct = 100 * r["n_fast_decisions"] / max(1, total_dec)

    print(f"\n  Resultado: media={resultado['recompensa_media']:.1f}  "
          f"max={resultado['recompensa_max']:.1f}")
    print(f"  Regímenes: rápido={r['n_fast_decisions']} ({fast_pct:.0f}%)  "
          f"resonante={r['n_resonant_decisions']}")
    print(f"  Orch-OR: {r['n_simulated']}  Crisis: {r['n_crises']}  "
          f"Revivals: {r['n_revivals']}")

    # Curva de aprendizaje por episodio
    eps = resultado["episodios"]
    buenos = sum(1 for r in eps if r > REWARD_THRESHOLD)
    print(f"  Episodios buenos (>{REWARD_THRESHOLD}): {buenos}/{n_episodios}")

    # Análisis de regímenes por episodio
    regimes = resultado["regimes"]
    print(f"  Regímenes por episodio:")
    for i, (rw, reg) in enumerate(zip(eps, regimes), 1):
        tag = "BUENO" if rw > REWARD_THRESHOLD else "malo"
        f, rs, rn = reg["fast"], reg["resonant"], reg["random"]
        print(f"    Ep {i:2d}: {rw:6.0f}  fast={f:3d} res={rs:2d} rnd={rn:2d}  {tag}")

    return {
        "label": label,
        "media": resultado["recompensa_media"],
        "max": resultado["recompensa_max"],
        "episodios": eps,
        "buenos": buenos,
        "fast_pct": fast_pct,
        "n_fast": r["n_fast_decisions"],
        "n_resonant": r["n_resonant_decisions"],
        "n_crises": r["n_crises"],
        "library_post": r["library"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep",   type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print(f"Prueba de transferencia — v0.0.3")
    print(f"CartPole-v1 → {ENV_NAME}")
    print("=" * 60)

    # ── Condición A: agente fresco — usa DB propia para no contaminar CartPole ─
    lib_A = ResonantLibrary(db_path="library/transfer_A_db.sqlite")
    result_A = run_condition(
        "A — Agente fresco (sin biblioteca CartPole)",
        library=lib_A,
        n_episodios=args.ep,
        seed=args.seed,
    )

    # ── Condición B: arranca con biblioteca de CartPole, escribe en DB propia ─
    # Copia los registros de CartPole a una DB separada para no contaminar el original.
    import shutil, os
    src = "library/resonant_db.sqlite"
    dst = "library/transfer_B_db.sqlite"
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"\n[Setup B] Copiada biblioteca CartPole → {dst}")
    lib_B = ResonantLibrary(db_path=dst)
    result_B = run_condition(
        "B — Agente con biblioteca CartPole",
        library=lib_B,
        n_episodios=args.ep,
        seed=args.seed,
    )

    # ── Condición C: aleatorio ────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[C — Aleatorio]")
    result_C = agente_aleatorio(ENV_NAME, args.ep, seed=args.seed)
    print(f"  media={result_C['recompensa_media']:.1f}  "
          f"max={result_C['recompensa_max']:.1f}")

    # ── Comparativa ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("COMPARATIVA FINAL")
    print(f"{'='*60}")
    print(f"  {'Condición':<40} {'Media':>7}  {'Max':>7}  {'Buenos':>7}  {'Fast%':>6}")
    print(f"  {'-'*40} {'-'*7}  {'-'*7}  {'-'*7}  {'-'*6}")

    for r in [result_A, result_B]:
        print(f"  {r['label']:<40} {r['media']:>7.1f}  {r['max']:>7.1f}  "
              f"{r['buenos']:>6}/{args.ep}  {r['fast_pct']:>5.0f}%")
    print(f"  {'C — Aleatorio':<40} {result_C['recompensa_media']:>7.1f}  "
          f"{result_C['recompensa_max']:>7.1f}  {'—':>7}  {'—':>6}")

    delta_A = result_A["media"] - result_C["recompensa_media"]
    delta_B = result_B["media"] - result_C["recompensa_media"]
    delta_AB = result_B["media"] - result_A["media"]
    print(f"\n  Δ A vs aleatorio: {delta_A:+.1f}")
    print(f"  Δ B vs aleatorio: {delta_B:+.1f}")
    print(f"  Δ B vs A (efecto de la biblioteca): {delta_AB:+.1f}")

    # ── Diagnóstico de hipótesis ──────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("DIAGNÓSTICO")
    print(f"  H1 (¿biblioteca activa fast path?): "
          f"{'SÍ' if result_B['fast_pct'] > 10 else 'NO'} "
          f"({result_B['fast_pct']:.0f}% fast en B vs {result_A['fast_pct']:.0f}% en A)")

    if delta_AB > 20:
        verdict = "TRANSFERENCIA POSITIVA — la biblioteca de CartPole ayuda en Acrobot"
    elif delta_AB < -20:
        verdict = "INTERFERENCIA — la biblioteca de CartPole perjudica en Acrobot"
    else:
        verdict = "TRANSFERENCIA NEUTRA — la biblioteca no interfiere ni ayuda"
    print(f"  H2 (efecto de transferencia): {verdict}")

    # ── Guardar resultados ────────────────────────────────────────────────────
    out = {
        "version": VERSION,
        "env": ENV_NAME,
        "n_episodios": args.ep,
        "A_fresco": {
            "media": result_A["media"], "max": result_A["max"],
            "buenos": result_A["buenos"], "fast_pct": result_A["fast_pct"],
            "episodios": result_A["episodios"],
        },
        "B_con_cartpole": {
            "media": result_B["media"], "max": result_B["max"],
            "buenos": result_B["buenos"], "fast_pct": result_B["fast_pct"],
            "episodios": result_B["episodios"],
        },
        "C_aleatorio": {
            "media": result_C["recompensa_media"], "max": result_C["recompensa_max"],
            "episodios": result_C["episodios"],
        },
        "deltas": {
            "A_vs_random": round(delta_A, 2),
            "B_vs_random": round(delta_B, 2),
            "B_vs_A": round(delta_AB, 2),
        },
        "verdict": verdict,
    }
    with open("results_transfer.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResultados → results_transfer.json")


if __name__ == "__main__":
    main()
