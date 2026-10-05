"""
pnp_explorer.py — Lógica de exploración autónoma para P vs NP.

El explorer decide qué experimentos hacer basándose en la pizarra.
Estrategias: fase de transición, dureza empírica, complejidad de prueba, circuitos.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PNPTarget:
    tool:     str          # "phase_scan", "hardness", "resolution", "circuit", "growth"
    k:        int = 3
    n_vars:   int = 15
    ratio:    float = 4.27  # cerca de umbral 3-SAT
    seed:     int = 0
    priority: float = 0.5
    reason:   str = ""


# Umbral conocido de fase de transición 3-SAT
KNOWN_3SAT_THRESHOLD = 4.267


class PNPExplorer:
    """
    Genera targets de exploración para P vs NP.

    Estrategias:
      1. FASE    — explorar la transición de fase k-SAT a distintos k
      2. ESCALA  — medir crecimiento de dureza al aumentar n
      3. PRUEBA  — medir complejidad de resolución (pigeonhole)
      4. CIRCUITO — estimar cotas inferiores de circuitos
    """

    def __init__(self, rng_seed: int = 42):
        self._rng = random.Random(rng_seed)
        self._tried_k    = set()
        self._tried_n    = set()
        self._phase_done = False

    def next_targets(self, blackboard, n: int = 2) -> list[PNPTarget]:
        targets = []
        obs     = blackboard.recent_observations(50)
        zeros   = blackboard.all_zeros()   # reutilizamos para "hallazgos"

        # 1. FASE — si no hemos explorado la fase de transición de 3-SAT
        if not self._phase_done and not any(o["tool"] == "phase_scan" for o in obs):
            targets.append(PNPTarget(
                tool="phase_scan", k=3, n_vars=20,
                ratio=KNOWN_3SAT_THRESHOLD,
                priority=0.95,
                reason="explorar transición de fase 3-SAT en n=20",
                seed=self._rng.randint(0, 9999),
            ))
            self._phase_done = True

        # 2. ESCALA — probar k distintos (4-SAT, 5-SAT)
        for k in [4, 5, 2]:
            if k not in self._tried_k and len(targets) < n:
                threshold = self._k_sat_threshold(k)
                targets.append(PNPTarget(
                    tool="phase_scan", k=k, n_vars=15,
                    ratio=threshold,
                    priority=0.80,
                    reason=f"transición de fase {k}-SAT (umbral~{threshold:.2f})",
                    seed=self._rng.randint(0, 9999),
                ))
                self._tried_k.add(k)

        # 3. ESCALA DE DUREZA — medir crecimiento con n
        phase_obs = [o for o in obs if o["tool"] == "phase_scan"]
        if phase_obs and len(targets) < n:
            done_ns = {r.get("result", {}).get("n", 0) for r in phase_obs
                       if isinstance(r.get("result"), dict)}
            # Intentar n más grande
            next_n = max(done_ns | {10}, default=10) + 10
            if next_n <= 60 and next_n not in self._tried_n:
                targets.append(PNPTarget(
                    tool="hardness", k=3, n_vars=next_n,
                    ratio=KNOWN_3SAT_THRESHOLD,
                    priority=0.75,
                    reason=f"medir dureza a n={next_n} (¿crece exponencialmente?)",
                    seed=self._rng.randint(0, 9999),
                ))
                self._tried_n.add(next_n)

        # 4. PRUEBA DE RESOLUCIÓN — pigeonhole
        if not any(o["tool"] == "resolution" for o in obs) and len(targets) < n:
            targets.append(PNPTarget(
                tool="resolution", n_vars=5,
                priority=0.70,
                reason="medir complejidad de resolución PHP_6,5",
                seed=0,
            ))

        # 5. CIRCUITO — cotas inferiores
        if not any(o["tool"] == "circuit" for o in obs) and len(targets) < n:
            targets.append(PNPTarget(
                tool="circuit", n_vars=6,
                priority=0.60,
                reason="estimar cota inferior de circuito para XOR de 6 bits",
            ))

        # 6. GROWTH — si tenemos datos de escala, analizar crecimiento
        hardness_obs = [o for o in obs if o["tool"] in ("phase_scan", "hardness")]
        if len(hardness_obs) >= 3 and not any(o["tool"] == "growth" for o in obs):
            targets.append(PNPTarget(
                tool="growth",
                priority=0.65,
                reason="analizar si el crecimiento de dureza es exponencial",
            ))

        targets.sort(key=lambda x: x.priority, reverse=True)
        return targets[:n]

    def _k_sat_threshold(self, k: int) -> float:
        """Umbral empírico de transición de fase para k-SAT."""
        thresholds = {2: 1.0, 3: 4.267, 4: 9.93, 5: 21.12, 6: 43.37}
        return thresholds.get(k, 2.0 * (2 ** k) * math.log(2))

    def should_ask_oracle(self, observation: dict, blackboard) -> Optional[str]:
        """¿Debería el agente preguntar al oráculo sobre esta observación?"""
        tool   = observation.get("tool")
        result = observation.get("result", {})

        # Caso 1: dureza exponencial detectada
        if tool == "growth":
            growth = result.get("growth_type")
            exp    = result.get("exponent_est", 0)
            if growth == "exponencial":
                return (
                    f"Observé que la dureza de 3-SAT crece con exponente ~{exp:.1f} "
                    f"al aumentar n. ¿Esto es consistente con la hipótesis P≠NP? "
                    f"¿Qué barreras (relativización, pruebas naturales, algebrización) "
                    f"impiden que este tipo de evidencia empírica pruebe P≠NP?"
                )

        # Caso 2: transición de fase muy abrupta
        if tool == "phase_scan":
            results = result if isinstance(result, list) else [result]
            hard_rates = [r.get("hard_rate", 0) for r in results
                          if isinstance(r, dict)]
            if hard_rates and max(hard_rates) > 0.5:
                return (
                    f"En el escaneo de fase de {result[0].get('k',3)}-SAT, "
                    f"el {max(hard_rates)*100:.0f}% de instancias son duras "
                    f"cerca del umbral de transición. ¿Qué nos dice la transición "
                    f"de fase sobre la complejidad promedio vs peor caso? "
                    f"¿Hay conexión con la pregunta P vs NP?"
                )

        # Caso 3: pigeonhole requiere muchos pasos
        if tool == "resolution":
            steps = result.get("steps", 0)
            n     = result.get("n", 0)
            if steps > 1000 * n:
                return (
                    f"El principio del palomar PHP_{n+1},{n} requirió {steps} "
                    f"pasos de resolución para n={n}. ¿Por qué PHP es duro para "
                    f"resolución pero fácil para otros sistemas de prueba como "
                    f"Frege? ¿Qué implica esto sobre la separación de clases de complejidad?"
                )

        # Caso 4: circuito con gran gap entre cota superior e inferior
        if tool == "circuit":
            lb = result.get("lower_bound_xor", 0)
            naive = result.get("gates_naive_xor", 0)
            if naive > 2 * lb:
                return (
                    f"Para XOR de {result.get('n_bits',6)} bits: cota inferior = {lb} puertas, "
                    f"implementación naïve = {naive} puertas. "
                    f"¿Cómo se relaciona el problema de cotas inferiores de circuitos "
                    f"con P vs NP? ¿Por qué es tan difícil probar cotas inferiores superlineales?"
                )

        return None
