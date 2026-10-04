"""
run_autoimprove.py — Bucle de auto-mejora del agente vX (v0.0.7)

Ciclo de segundo orden:
  1. El MAESTRO (segundo orden) corre N episodios → establece línea base
  2. Diagnostica el tipo de fallo
  3. Genera 3 variantes de primer orden
  4. Evalúa cada variante con biblioteca caliente (solo lectura)
  5. La variante ganadora transfiere su subcampo exec al maestro (absorción)
  6. El maestro corre de nuevo con el nuevo exec → nueva línea base
  7. Repite hasta alcanzar objetivo o agotar iteraciones

"No es que cambie tanto el que piensa sino el que actúa."
El exec del maestro mejora; su ciclo de pensamiento permanece.
"""

import argparse
import json
import copy
import numpy as np

from core.proto_tissue_agent_v2 import ProtoTissueAgentV2
from core.second_order import SecondOrderAgent
from gym_adapter import evaluar_agente
from library.store import ResonantLibrary

VERSION = "v0.0.7"

LIBRARY_DOMINANCE_THRESHOLD = 0.82


# ── Configuración base ────────────────────────────────────────────────────────

BASE_CONFIG = {
    "N":                    50,
    "M":                    5,
    "K":                    3,
    "confidence_threshold":  0.93,
    "filter_threshold":     0.45,
    "stagnation_threshold":  8,
    "epsilon_start":        0.6,
    "epsilon_min":          0.05,
    "n_episodios":          20,
    "reward_threshold":     30.0,
}


# ── Evaluador ─────────────────────────────────────────────────────────────────

def evaluar(agente, env_name: str, n_episodios: int,
            cfg: dict, seed: int, verbose: bool = False) -> dict:
    return evaluar_agente(
        agente,
        env_name=env_name,
        n_episodios=n_episodios,
        max_steps=500,
        verbose=verbose,
        seed=seed,
        reward_threshold=cfg["reward_threshold"],
        version=VERSION,
        epsilon_start=cfg["epsilon_start"],
        epsilon_min=cfg["epsilon_min"],
        epsilon_decay_ep=n_episodios // 2,
        blend=0.6,
    )


def diagnosticar(resultado: dict) -> str:
    r = resultado["reporte_agente"]
    total_dec = r['n_fast_decisions'] + r['n_resonant_decisions']
    fast_pct  = r['n_fast_decisions'] / max(1, total_dec)
    if fast_pct > LIBRARY_DOMINANCE_THRESHOLD:
        return "library_dominance"
    fails     = r.get("failures", {})
    dominante = fails.get("dominant", "none")
    return dominante or "none"


# ── Bucle principal ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",      default="CartPole-v1")
    parser.add_argument("--objetivo", type=float, default=50.0)
    parser.add_argument("--max_iter", type=int, default=6)
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument("--verbose",  action="store_true", default=True)
    args = parser.parse_args()

    print("=" * 65)
    print(f"Agente vX {VERSION} — Segundo Orden")
    print(f"Objetivo: media ≥ {args.objetivo}  |  Máx. iteraciones: {args.max_iter}")
    print("=" * 65)

    library  = ResonantLibrary(db_path="library/resonant_db.sqlite")
    lib_info = library.summary()
    print(f"\n[Biblioteca] {lib_info['total']} registros  versiones={lib_info['versions']}")

    maestro = SecondOrderAgent(BASE_CONFIG, library, seed=args.seed)
    cfg     = copy.deepcopy(BASE_CONFIG)
    historial = []
    exito   = False

    for iteracion in range(1, args.max_iter + 1):
        print(f"\n{'─'*65}")
        print(f"  ITERACIÓN {iteracion}/{args.max_iter}  [MAESTRO]")
        print(f"  Config: N={cfg['N']}  conf_thr={cfg['confidence_threshold']:.2f}  "
              f"ε_start={cfg['epsilon_start']:.2f}  "
              f"filter_thr={cfg['filter_threshold']:.2f}  n_ep={cfg['n_episodios']}")
        print(f"{'─'*65}\n")

        # 1. El maestro corre en el entorno
        maestro.master.set_env(args.env)
        resultado = evaluar(
            maestro.master, args.env, cfg["n_episodios"],
            cfg, args.seed, verbose=args.verbose,
        )

        media    = resultado["recompensa_media"]
        maximo   = resultado["recompensa_max"]
        r        = resultado["reporte_agente"]
        total_dec = r['n_fast_decisions'] + r['n_resonant_decisions']
        fast_pct  = r['n_fast_decisions'] / max(1, total_dec)
        fails     = r.get("failures", {})
        diagnostico = diagnosticar(resultado)

        maestro.set_baseline(media)

        print(f"\n  ── Maestro iter {iteracion}: media={media:.1f}  máx={maximo:.1f}")
        print(f"     fast_pct={fast_pct:.0%}  fallos={fails.get('total_failures',0)}"
              f"  → {fails.get('by_type',{})}  (diagnóstico: {diagnostico})")

        registro = {
            "iteracion":   iteracion,
            "config":      copy.deepcopy(cfg),
            "media":       media,
            "max":         maximo,
            "fast_pct":    round(fast_pct, 3),
            "failures":    fails,
            "diagnostico": diagnostico,
            "exito":       media >= args.objetivo,
            "generacion":  None,
        }

        if media >= args.objetivo:
            print(f"\n  ✓ OBJETIVO ALCANZADO (media {media:.1f} ≥ {args.objetivo})")
            exito = True
            historial.append(registro)
            break

        if iteracion == args.max_iter:
            print(f"\n  ✗ Se agotaron {args.max_iter} iteraciones.")
            historial.append(registro)
            break

        # 2. Generar variantes + absorber al ganador
        def _eval_fn(ag, env_name, n_ep):
            ag.set_env(env_name)
            return evaluar(ag, env_name, n_ep, cfg, args.seed + 10, verbose=False)

        gen_result = maestro.run_generation(
            diagnostico=diagnostico,
            evaluator_fn=_eval_fn,
            env_name=args.env,
            n_episodes=min(15, cfg["n_episodios"]),
            seed=args.seed,
            verbose=True,
        )
        registro["generacion"] = gen_result

        # Sincronizar cfg con el estado actual del maestro
        cfg = copy.deepcopy(maestro.base_config)

        historial.append(registro)

    # ── Guardar historial ─────────────────────────────────────────────────────
    salida = {
        "version":         VERSION,
        "env":             args.env,
        "objetivo":        args.objetivo,
        "exito":           exito,
        "total_iter":      len(historial),
        "mejor_media":     max(h["media"] for h in historial),
        "config_final":    cfg,
        "absorption_log":  maestro.absorption_log,
        "historial":       historial,
    }

    with open("results_autoimprove.json", "w") as f:
        json.dump(salida, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  Resultado final: {'ÉXITO' if exito else 'NO alcanzado'}")
    print(f"  Mejor media en {len(historial)} iteraciones: {salida['mejor_media']:.1f}")
    print(f"  Absorciones: {len(maestro.absorption_log)}")
    print(f"  Historial → results_autoimprove.json")
    print(f"{'='*65}")

    _actualizar_bitacora(salida, args.env)


def _actualizar_bitacora(salida: dict, env_name: str) -> None:
    import datetime
    fecha = datetime.date.today().isoformat()

    absorciones = salida.get("absorption_log", [])
    abs_str = ""
    for a in absorciones:
        abs_str += (f"  Gen {a['generation']}: absorbió '{a['winner']}' "
                    f"(base {a['media_antes']:.1f} → ganador {a['media_ganador']:.1f})\n")

    entrada = f"""

---

## v0.0.7 — {fecha} — Segundo Orden ({env_name})

### Arquitectura
- SecondOrderAgent: el maestro no corre variantes — absorbe el exec del ganador
- Agentes de 1° orden: comparten biblioteca (solo lectura), cada uno prueba una variante
- Absorción: exec del ganador se mezcla con exec del maestro (α=0.40)
- El pensamiento del maestro (perc/imag/sub/mem) permanece intacto

### Iteraciones
"""
    for h in salida["historial"]:
        estado = "✓" if h["exito"] else "✗"
        gen = h.get("generacion")
        gen_str = ""
        if gen:
            winner = gen.get("winner", "?")
            variants = gen.get("variants_tried", [])
            gen_str = f"  → gen: variantes={[v[0] for v in variants]} ganador={winner}"
        entrada += (f"- Iter {h['iteracion']}: media={h['media']:.1f}  "
                    f"fast={h['fast_pct']:.0%}  diag={h['diagnostico']}"
                    f"{gen_str}  {estado}\n")

    entrada += f"""
### Absorciones
{abs_str if abs_str else "  ninguna"}

### Resultado
- Éxito: {salida['exito']}
- Mejor media: {salida['mejor_media']:.1f}
- Iteraciones: {salida['total_iter']}
- Absorciones: {len(absorciones)}
"""
    with open("bitacora.md", "a", encoding="utf-8") as f:
        f.write(entrada)
    print("  Bitácora actualizada.")


if __name__ == "__main__":
    main()
