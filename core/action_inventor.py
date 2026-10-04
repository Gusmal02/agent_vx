"""
core/action_inventor.py — Inventor de acciones via oracle (v0.1.0)

El agente propone nuevas acciones más allá de las 5 hardcodeadas.
Usa una llamada dedicada al oracle (presupuesto separado, máx 1/sesión).

Basado en:
  - E4: el método orgánico gana 80% porque no está forzado a estar entre p1 y p2
  - hipotesis_mente_creativa P6: sin input externo, el campo agota sus combinaciones
  - La invención es el input externo más valioso: nuevas herramientas, no solo nuevos datos
"""

from __future__ import annotations

import json
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.oracle_client import OracleClient
    from core.research_memory import ResearchMemory


ML_LIBRARIES_KNOWN = """
Librerías disponibles en el sandbox (ya instaladas):
  numpy        — arrays, álgebra lineal, FFT (np.fft), estadística
  scipy        — scipy.optimize (fsolve, minimize, root), scipy.linalg, scipy.signal
  mpmath       — aritmética de precisión arbitraria, zetazero, gamma, zeta
  sympy        — álgebra simbólica, factorización, límites, series
  itertools    — permutaciones, combinaciones (para permanent)
  networkx     — grafos (útil para estructuras combinatorias)

Para Riemann: mpmath.zetazero(k), mpmath.zeta(s), mpmath.li(x)
Para P≠NP:   np.linalg.det, itertools.permutations, sympy.factorint
Análisis espectral: np.linalg.eigvalsh, np.fft.fft
Optimización: scipy.optimize.fsolve, scipy.optimize.minimize
"""

INVENTION_PROMPT_TEMPLATE = """
Eres un matemático computacional. El agente autónomo que ataca {problem} ha ejecutado
{n_cycles} ciclos con las siguientes herramientas y resultados:

Herramientas actuales y sus scores:
{tool_summary}

Mejores resultados encontrados:
{memory_summary}

Lo que NO ha funcionado bien (score < 0.4 en últimos ciclos):
{failed_actions}

{libraries}

Tu tarea: INVENTAR UNA NUEVA HERRAMIENTA matemática que el agente NO tiene aún.
La herramienta debe:
1. Ser diferente a las ya existentes
2. Usar métodos que aún no se han probado
3. Ser ejecutable en el sandbox (solo imports de las librerías listadas arriba)
4. Retornar un _result dict con un campo "score" (0.0–1.0)

Responde en JSON exactamente con esta estructura:
{{
  "tool_name": "nombre_snake_case",
  "description": "qué hace en una línea",
  "rationale": "por qué podría avanzar en {problem} (2-3 frases)",
  "code": "def tool_fn(omega, candidates, context):\\n    import numpy as np\\n    # ... código completo\\n    _result = {{...}}\\n"
}}
"""


class ActionInventor:
    """
    Inventa nuevas acciones matemáticas via oracle.
    Budget: máx 1 invención por sesión (costoso pero alto valor).
    """

    def __init__(self, oracle: "OracleClient"):
        self._oracle    = oracle
        self._invented  = 0       # invenciones en esta sesión
        self.MAX_PER_SESSION = 2  # límite por sesión
        self.last_invention: Optional[dict] = None

    @property
    def can_invent(self) -> bool:
        return (
            self._invented < self.MAX_PER_SESSION
            and self._oracle.available
        )

    def try_invent(self,
                   problem:         str,
                   n_cycles:        int,
                   tool_summary:    str,
                   failed_actions:  list[str],
                   memory:          Optional["ResearchMemory"] = None,
                   ) -> Optional[dict]:
        """
        Intenta inventar una nueva herramienta.
        Devuelve dict con tool_name, description, code, rationale.
        O None si no se puede/debe inventar ahora.
        """
        if not self.can_invent:
            return None

        mem_summary = (memory.to_corpus_knowledge(top_k=5)
                       if memory else "(sin memoria indexada)")

        prompt = INVENTION_PROMPT_TEMPLATE.format(
            problem        = problem,
            n_cycles       = n_cycles,
            tool_summary   = tool_summary,
            memory_summary = mem_summary,
            failed_actions = ", ".join(failed_actions) if failed_actions else "ninguna",
            libraries      = ML_LIBRARIES_KNOWN,
        )

        try:
            raw = self._oracle._client.messages.create(
                model      = "claude-sonnet-4-6",
                max_tokens = 1500,
                messages   = [{"role": "user", "content": prompt}],
            )
            text = raw.content[0].text.strip()

            # Extraer JSON
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            invention = json.loads(text)

            required = {"tool_name", "description", "code"}
            if not required.issubset(invention.keys()):
                return None

            self._invented += 1
            self.last_invention = invention
            print(f"  [ActionInventor] Nueva herramienta: {invention['tool_name']}")
            print(f"    Razón: {invention.get('rationale', '')[:120]}")
            return invention

        except Exception as e:
            print(f"  [ActionInventor] Error: {e}")
            return None

    def summary(self) -> dict:
        return {
            "invented_this_session": self._invented,
            "max_per_session":       self.MAX_PER_SESSION,
            "last_tool":             (self.last_invention or {}).get("tool_name"),
        }
