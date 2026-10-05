# Agent v013 — Diseño desde cero

## Problema con v012

El oráculo es el motor. El agente ejecuta lo que el LLM genera.
La exploración es la del LLM, no la del agente.

## Principio central

El agente tiene tres capas:

```
CORPUS          → referencia. Lo que ya se sabe. No se re-explora.
PIZARRA         → estado propio. Lo que el agente computa y observa.
ORÁCULO         → consultor. Responde preguntas conceptuales. Nunca genera código.
```

## Ciclo del agente

```
1. Lee pizarra + corpus
2. Decide qué región explorar (él solo, basado en lo que ya sabe)
3. Ejecuta código propio (sus herramientas matemáticas)
4. Escribe el resultado en la pizarra
5. Si el resultado lo sorprende → pregunta al oráculo "¿qué es esto?"
6. Oráculo orienta conceptualmente → agente vuelve al paso 2
7. Si el resultado es nuevo respecto al corpus → marca como hallazgo
```

## Regla dura

**El oráculo nunca genera código.**
Solo responde preguntas del tipo:
- "Encontré este patrón en los espaciados de ceros, ¿qué área estudia esto?"
- "Esta densidad se comporta así, ¿hay alguna conjetura relacionada?"

El agente escribe todo el código basándose en sus herramientas y en lo que observa.

## Herramientas del agente (código que él sabe ejecutar)

No son generadas por LLM. Son funciones matemáticas fijas que el agente aplica
a regiones que él elige:

```python
TOOLS = {
    "winding":   scan_winding(sigma_range, t_range)   # ¿hay ceros aquí?
    "isolate":   isolate_zero(sigma_range, t_range)   # aislar un cero exacto
    "density":   zero_density(t_min, t_max, width)    # densidad de ceros
    "spacing":   analyze_spacing(zeros_list)           # análisis de espaciados
    "deviation": measure_deviation(zeros_list)         # ¿cuánto se aleja de σ=0.5?
}
```

## Explorer — cómo decide qué explorar

```
Estado inicial: t_frontier = max(t conocido en corpus)
                             (por defecto: t=50, después de los primeros ~15 ceros)

Ciclo de exploración:
  → Si winding(t_frontier, t_frontier+10) ≠ 0:
      aislar ceros en esa franja
      medir desviación de σ=0.5
      guardar en pizarra
      avanzar t_frontier
  → Si el espaciado de ceros es inusual:
      preguntar al oráculo "¿qué significa este patrón?"
      anotar contexto en pizarra
      explorar la zona inusual más de cerca
  → Si se atora (misma región, sin progreso):
      saltar a una t aleatoria más alta
      o cambiar herramienta (de winding a density)
```

## Pizarra — estructura

```python
{
  "explored_regions": [(t_min, t_max, n_zeros_found), ...],
  "zeros_found":      [(sigma, t, deviation), ...],
  "frontier_t":       float,         # hasta dónde llegó
  "surprises":        [...],         # resultados inesperados
  "oracle_answers":   [...],         # lo que el oráculo respondió
  "conjectures":      [...],         # hipótesis propias del agente
}
```

## Qué sale del agente

No un "score" sobre una hipótesis fija. Una lista de:
- Ceros que encontró y verificó (con σ, t, desviación)
- Regiones donde la densidad de ceros es inusual
- Patrones en el espaciado que no estaban en el corpus
- Conjeturas propias basadas en sus observaciones

## Archivos

| Archivo | Rol |
|---|---|
| `run_v013.py` | Nuevo agente — ciclo completo |
| `core/blackboard.py` | Pizarra SQLite del agente |
| `core/math_explorer.py` | Lógica de exploración autónoma |
| `core/math_tools.py` | Herramientas matemáticas (winding, isolate, density, spacing) |
| `run_v020.py` | Coordinador — llama v013 en lugar de v012 |
