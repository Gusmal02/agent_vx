# Finding: Overhead Epistémico del Estado Previo

**ID:** agentevx-f002  
**Fecha:** 2026-10-03  
**Problema estudiado:** Aprendizaje continual  
**Experimento:** `experimento_reaprendizaje.py` — 4 sesiones S1→S4  
**Datos:** `results/reaprendizaje_continual_20261003_2339.json`

---

## Enunciado

Cuando el agente vX inicia con estado epistémico previo (`has_prior=True`), converge en **más ciclos** que desde cero, no menos.

## Datos

| Sesión | Modo         | Ciclos | Richness | Cross |
|--------|--------------|--------|----------|-------|
| S1     | frío         | 35     | 1.000    | 6     |
| S2     | con estado   | 42     | 1.000    | 6     |
| S3     | frío         | 35     | 1.000    | 6     |
| S4     | con estado   | 42     | 1.000    | 6     |

- El patrón es **perfectamente determinista**: S1=S3=35, S2=S4=42.
- Overhead = 7 ciclos extra (~20%) cuando existe estado previo.

## Mecanismo causal

La regla `should_transition(has_prior=True)` exige `n_new >= 1`: al menos una hipótesis genuinamente nueva más allá de las que ya estaban en el estado cargado. El agente que encuentra las 3 hipótesis conocidas no puede consolidar — debe seguir explorando hasta encontrar algo nuevo o alcanzar `FOCUS_MAX_CYCLES`.

El estado previo no es un *scaffold* que acelera; es un *filtro de novedad* que eleva la barra de convergencia.

## Interpretación meta

El agente que estudia EWC exhibe el mismo fenómeno que estudia:

- **Sin estado previo**: el olvido es catastrófico y completo. Cada cold-start cuesta exactamente lo mismo (35 ciclos). No hay *savings*.
- **Con estado previo**: el conocimiento almacenado no produce transferencia positiva en velocidad. Produce overhead epistémico porque el agente debe demostrar novedad, no solo confirmación.

Esto invierte el patrón de la memoria biológica, donde el reaprendizaje es más rápido que el aprendizaje original.

## Trade-off identificado

| Modo | Costo | Qué garantiza |
|------|-------|---------------|
| cold-start | 35 ciclos | Reconvergencia en hallazgos conocidos |
| con estado | 42 ciclos | Solo consolida si encuentra algo genuinamente nuevo |

El overhead de 7 ciclos es el precio de evitar convergencia trivial (reconfirmar lo ya sabido sin explorar).

## Implicaciones para el diseño

1. El parámetro clave es la regla `n_new >= 1`. Relajarla a `n_new >= 0` después de N ciclos daría savings a costa de tolerar convergencia sobre conocimiento previo.
2. La perfecta determinismo (35/42 exactos en dos repeticiones) indica que el espacio del problema `continual` es lo suficientemente estable para que el agente siga la misma trayectoria.
3. Si se quiere medir genuina transferencia de aprendizaje habría que usar un problema con mayor varianza estocástica donde los paths de exploración difieran entre runs.

## Relación con finding anterior

**agentevx-f001** (hallazgo sobre EWC): `r(sim_fgt_corr, retention_improvement) = -0.857` — EWC ayuda más cuando la similitud entre tareas no predice el olvido.

El presente finding es el análogo arquitectural: el estado epistémico ayuda menos de lo esperado cuando el problema es estacionario. Ambos hallazgos apuntan a que las métricas de similitud/transferencia no son predictores lineales del beneficio de la memoria.
