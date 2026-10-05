"""
math_explorer.py — Lógica de exploración autónoma del agente.

El explorer decide QUÉ explorar y CON QUÉ herramienta basándose en la pizarra.
No recibe instrucciones del oráculo sobre qué hacer.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExplorationTarget:
    tool:         str            # "scan", "isolate", "density", "spacing"
    t_min:        float
    t_max:        float
    sigma_range:  tuple = (0.48, 0.52)
    priority:     float = 0.5
    reason:       str   = ""


class MathExplorer:
    """
    Genera targets de exploración autónomamente.

    Estrategias (en orden de prioridad):
      1. AVANZAR  — continuar desde la frontera actual
      2. ANOMALÍA — re-explorar zonas donde la densidad fue inusual
      3. SALTO    — ir a una t alta no explorada (exploración aleatoria)
      4. ESPACIADO — analizar los ceros ya encontrados
    """

    WINDOW = 8.0      # tamaño de ventana de exploración
    SIGMA  = (0.48, 0.52)

    def __init__(self, rng_seed: int = 42):
        self._rng = random.Random(rng_seed)

    def next_targets(self, blackboard, n: int = 3) -> list[ExplorationTarget]:
        """Genera los próximos N targets a explorar."""
        targets = []

        frontier = blackboard.frontier_t()
        zeros    = blackboard.all_zeros()
        explored = blackboard.explored_ranges()

        # 1. AVANZAR desde la frontera
        t0 = frontier
        t1 = t0 + self.WINDOW
        if not blackboard.already_explored(t0, t1):
            targets.append(ExplorationTarget(
                tool="scan", t_min=t0, t_max=t1,
                priority=0.9,
                reason=f"avanzar desde frontera t={t0:.1f}"
            ))

        # 2. SALTO — región alta no explorada (exploración genuina)
        jump_t = self._pick_unexplored_jump(explored, frontier)
        if jump_t:
            targets.append(ExplorationTarget(
                tool="scan", t_min=jump_t, t_max=jump_t + self.WINDOW,
                priority=0.7,
                reason=f"salto a región no explorada t={jump_t:.1f}"
            ))

        # 3. ANOMALÍA — densidad inusual en observaciones recientes
        anomaly = self._find_density_anomaly(blackboard)
        if anomaly:
            targets.append(ExplorationTarget(
                tool="density",
                t_min=anomaly["t_min"], t_max=anomaly["t_max"],
                priority=0.8,
                reason=f"densidad anómala ratio={anomaly.get('ratio','?')}"
            ))

        # 4. ESPACIADO — si tenemos ≥ 5 ceros, analizamos los gaps
        if len(zeros) >= 5 and not self._already_did_spacing(blackboard):
            # Usar los últimos 10 ceros encontrados
            recent_zeros = sorted(zeros, key=lambda z: z["t"])[-10:]
            targets.append(ExplorationTarget(
                tool="spacing",
                t_min=recent_zeros[0]["t"],
                t_max=recent_zeros[-1]["t"],
                priority=0.6,
                reason=f"analizar espaciado de {len(recent_zeros)} ceros recientes"
            ))

        # Ordenar por prioridad y retornar los primeros n
        targets.sort(key=lambda x: x.priority, reverse=True)
        return targets[:n]

    def _pick_unexplored_jump(self, explored: list[tuple],
                               frontier: float) -> Optional[float]:
        """
        Elige una t alta no explorada para hacer un salto.
        Prioriza regiones que ningún agente ha tocado.
        """
        explored_set = set()
        for t0, t1 in explored:
            # Marcar ventanas de 5 unidades como exploradas
            t = t0
            while t < t1:
                explored_set.add(round(t / 5) * 5)
                t += 5

        # Buscar en rangos más allá de la frontera actual
        # Rango candidato: frontier + 50 hasta frontier + 500
        lo = frontier + 50
        hi = frontier + 500
        candidates = []
        t = lo
        while t < hi:
            bucket = round(t / 5) * 5
            if bucket not in explored_set:
                candidates.append(t)
            t += self.WINDOW

        if not candidates:
            return None
        # Ponderar hacia t más altas (más desconocidas)
        weights = [math.log(1 + c - lo + 1) for c in candidates]
        return self._rng.choices(candidates, weights=weights, k=1)[0]

    def _find_density_anomaly(self, blackboard) -> Optional[dict]:
        """Busca en la pizarra una observación de density con ratio inusual."""
        obs = blackboard.recent_observations(30)
        anomalies = []
        for o in obs:
            if o["tool"] == "density":
                r = o["result"]
                ratio = r.get("ratio", 1.0)
                # Ratio < 0.5 o > 2.0 es inusual
                if ratio is not None and (ratio < 0.5 or ratio > 2.0):
                    anomalies.append({
                        "t_min": r.get("t_min", o["t_min"]),
                        "t_max": r.get("t_max", o["t_max"]),
                        "ratio": ratio,
                    })
        if not anomalies:
            return None
        # La anomalía más extrema
        return max(anomalies, key=lambda a: abs(a["ratio"] - 1.0))

    def _already_did_spacing(self, blackboard) -> bool:
        obs = blackboard.recent_observations(50)
        return any(o["tool"] == "spacing" for o in obs)

    def should_ask_oracle(self, observation: dict, blackboard) -> Optional[str]:
        """
        ¿Debería el agente preguntar al oráculo sobre esta observación?
        Retorna la pregunta si sí, None si no.
        """
        tool   = observation.get("tool")
        result = observation.get("result", {})

        # Caso 1: encontró un cero fuera de la línea crítica
        if tool in ("scan", "isolate"):
            zeros = blackboard.all_zeros()
            off_line = [z for z in zeros if not z["on_line"]]
            if off_line:
                z = off_line[-1]
                return (f"Encontré un cero en σ={z['sigma']:.6f}, t={z['t']:.4f} "
                        f"con desviación={z['deviation']:.2e} de la línea crítica σ=0.5. "
                        f"¿Qué implicaría matemáticamente un cero en esta posición? "
                        f"¿Hay conjeturas que predigan ceros fuera de σ=0.5 en ciertos rangos de t?")

        # Caso 2: densidad muy inusual
        if tool == "density":
            ratio = result.get("ratio", 1.0)
            if ratio is not None and (ratio < 0.3 or ratio > 3.0):
                return (f"La densidad de ceros en t∈[{result.get('t_min'):.0f}, {result.get('t_max'):.0f}] "
                        f"es {ratio:.2f}x la predicción de Riemann-von Mangoldt. "
                        f"¿Qué teoremas estudian fluctuaciones extremas en la densidad de ceros?")

        # Caso 3: espaciado muy diferente de GUE
        if tool == "spacing":
            gue_ratio = result.get("gue_ratio", 1.0)
            if gue_ratio is not None and (gue_ratio < 0.5 or gue_ratio > 2.0):
                return (f"El espaciado normalizado de los ceros tiene varianza={result.get('variance_norm'):.3f} "
                        f"vs varianza GUE={result.get('gue_variance'):.3f} (ratio={gue_ratio:.2f}). "
                        f"¿Qué significa una desviación así de la conjetura de Montgomery?")

        return None
