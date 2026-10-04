"""
run_gym.py — Evaluación del agente vX en Gymnasium

Uso:
  python run_gym.py                         # CartPole-v1, 10 episodios
  python run_gym.py --env CartPole-v1 --ep 20
  python run_gym.py --baseline              # comparar con agente aleatorio
  python run_gym.py --ep 30 --baseline      # test más largo
"""

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F

from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from gym_adapter import evaluar_agente, ActionEncoder, ObsEncoder
from library.store import ResonantLibrary

VERSION = "v0.0.4"


def agente_aleatorio(env_name: str, n_episodios: int, seed: int = 42) -> dict:
    import gymnasium as gym
    env = gym.make(env_name)
    recompensas = []
    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep)
        total, done, pasos = 0.0, False, 0
        while not done and pasos < 500:
            accion = env.action_space.sample()
            obs, r, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated
            total += r; pasos += 1
        recompensas.append(total)
    env.close()
    return {
        "episodios":        recompensas,
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_max":   float(np.max(recompensas)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",      default="CartPole-v1")
    parser.add_argument("--ep",       type=int, default=10)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--threshold", type=float, default=30.0,
                        help="Recompensa mínima de episodio para considerarlo bueno")
    args = parser.parse_args()

    print("=" * 60)
    print(f"Agente vX {VERSION} — {args.env}")
    print("=" * 60)

    # ── Biblioteca persistente ────────────────────────────────────────────────
    library = ResonantLibrary(db_path="library/resonant_db.sqlite")
    lib_info = library.summary()
    print(f"\n[Biblioteca] {lib_info['total']} registros"
          f"  versiones={lib_info['versions']}"
          f"  max_reward={lib_info['max_reward']:.3f}")

    # ── Agente vX ─────────────────────────────────────────────────────────────
    agente = ProtoTissueAgentV2(
        N=50, M=5, K=3, seed=args.seed,
        stagnation_threshold=8,
        sim_top_k=2,
        filter_threshold=0.45,
        library=library,
        confidence_threshold=0.93,
        t_settle=5,
        warmup_steps=50,
        subconscious_update_freq=5,
    )

    print(f"\n[Agente vX] {args.ep} episodios en {args.env}  "
          f"(threshold_bueno={args.threshold}):\n")

    resultado = evaluar_agente(
        agente,
        env_name=args.env,
        n_episodios=args.ep,
        verbose=True,
        seed=args.seed,
        reward_threshold=args.threshold,
        version=VERSION,
        epsilon_start=0.6,
        epsilon_min=0.05,
        epsilon_decay_ep=args.ep // 2,
        blend=0.6,
    )

    print(f"\n  Media: {resultado['recompensa_media']:.1f}")
    print(f"  Máx:   {resultado['recompensa_max']:.1f}")

    r = resultado["reporte_agente"]
    total_dec = r['n_fast_decisions'] + r['n_resonant_decisions']
    fast_pct  = 100 * r['n_fast_decisions'] / max(1, total_dec)
    print(f"\n  Decisiones: {total_dec}  "
          f"→ rápidas={r['n_fast_decisions']} ({fast_pct:.0f}%)  "
          f"resonantes={r['n_resonant_decisions']} ({100-fast_pct:.0f}%)")
    print(f"  Orch-OR sims: {r['n_simulated']}  "
          f"Crisis: {r['n_crises']}  "
          f"Revivals: {r['n_revivals']}")

    if r["library"]:
        lib_post = r["library"]
        print(f"\n  Biblioteca post-run: {lib_post['total']} registros  "
              f"avg_reward={lib_post['avg_reward']:.3f}  "
              f"max_reward={lib_post['max_reward']:.3f}")
        if lib_post.get("envs"):
            print(f"  Entornos en biblioteca: {list(lib_post['envs'].keys())}")

    # Diagnóstico de fallos (v0.0.4)
    fails = r.get("failures", {})
    if fails and fails.get("total_failures", 0) > 0:
        print(f"\n  Fallos: {fails['total_failures']}  "
              f"→ {fails['by_type']}  (dominante: {fails['dominant']})")

    # ── Baseline ──────────────────────────────────────────────────────────────
    if args.baseline:
        print(f"\n[Aleatorio] {args.ep} episodios en {args.env}:")
        base = agente_aleatorio(args.env, args.ep, seed=args.seed)
        for i, rr in enumerate(base["episodios"], 1):
            print(f"  Ep {i:2d}: recompensa={rr:.0f}")
        print(f"\n  Media: {base['recompensa_media']:.1f}")
        print(f"  Máx:   {base['recompensa_max']:.1f}")

        delta = resultado["recompensa_media"] - base["recompensa_media"]
        print(f"\n  Δ agente vs aleatorio: {delta:+.1f}")

    # ── Guardar resultados ────────────────────────────────────────────────────
    out = {
        "version": VERSION,
        "env":     args.env,
        "agente_vx": {
            "media":     resultado["recompensa_media"],
            "max":       resultado["recompensa_max"],
            "episodios": resultado["episodios"],
            "fast_pct":  round(fast_pct, 1),
            "fallos":    resultado.get("fallos", []),
            "failures":  r.get("failures", {}),
        },
    }
    if args.baseline:
        out["baseline_aleatorio"] = {
            "media":     base["recompensa_media"],
            "max":       base["recompensa_max"],
            "episodios": base["episodios"],
        }
        out["delta"] = round(delta, 2)

    with open("results_gym.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResultados → results_gym.json")


if __name__ == "__main__":
    main()
