"""
tool_registry.py — Registro de herramientas del agente

El agente tiene un conjunto de herramientas (estrategias de decisión).
Este módulo permite:
  - Registrar herramientas existentes con su rendimiento histórico
  - Inventar nuevas herramientas (funciones Python) en tiempo de ejecución
  - Seleccionar la herramienta más adecuada según el contexto actual
  - Desactivar herramientas que demuestran bajo rendimiento

Una "herramienta" es cualquier función que, dado un omega y contexto,
devuelve un candidato o acción. El tejido resonante es la herramienta base.
"""

import torch
import torch.nn.functional as F
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import textwrap


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable                       # fn(omega, candidates, context) → (winner, info)
    performance: List[float] = field(default_factory=list)
    active: bool = True
    invented: bool = False             # True si fue creada por el propio agente

    def avg_performance(self, last_n: int = 10) -> float:
        if not self.performance:
            return 0.0
        return sum(self.performance[-last_n:]) / min(len(self.performance), last_n)

    def record(self, reward: float) -> None:
        self.performance.append(reward)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "active": self.active,
            "invented": self.invented,
            "uses": len(self.performance),
            "avg_reward": round(self.avg_performance(), 2),
        }


class ToolRegistry:
    """
    Registro y selector de herramientas.

    El agente puede:
      - `register()`: añadir una herramienta al registro
      - `select()`: elegir la mejor herramienta para el contexto actual
      - `invent()`: crear una nueva herramienta a partir de código Python
      - `report_outcome()`: actualizar el rendimiento de la herramienta usada
    """

    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._last_used: Optional[str] = None

    # ── Registro ──────────────────────────────────────────────────────────────

    def register(self,
                 name: str,
                 fn: Callable,
                 description: str = "",
                 invented: bool = False) -> None:
        self._tools[name] = Tool(
            name=name,
            description=description,
            fn=fn,
            invented=invented,
        )

    def deactivate(self, name: str) -> None:
        if name in self._tools:
            self._tools[name].active = False

    # ── Selección ─────────────────────────────────────────────────────────────

    def select(self,
               context_vector: Optional[torch.Tensor] = None,
               memory_status: str = "empty",
               ) -> Optional[Tool]:
        """
        Selecciona la herramienta activa con mejor rendimiento promedio.
        Si el contexto indica memoria escasa, preferir herramientas de exploración.
        """
        active = [t for t in self._tools.values() if t.active]
        if not active:
            return None

        # Con memoria escasa, preferir herramientas inventadas (más especializadas)
        if memory_status in ("empty", "sparse"):
            invented = [t for t in active if t.invented]
            if invented:
                best = max(invented, key=lambda t: t.avg_performance())
                self._last_used = best.name
                return best

        best = max(active, key=lambda t: t.avg_performance())
        self._last_used = best.name
        return best

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    # ── Invención de herramientas ─────────────────────────────────────────────

    def invent(self,
               name: str,
               code_str: str,
               description: str = "herramienta inventada",
               ) -> Tuple[bool, str]:
        """
        Crea una nueva herramienta a partir de código Python en string.

        El código debe definir una función llamada `tool_fn(omega, candidates, context)`.
        Devuelve (éxito, mensaje).

        SEGURIDAD: solo se ejecuta en el sandbox del agente, no en producción.
        """
        namespace: dict = {"torch": torch, "F": F}
        try:
            exec(textwrap.dedent(code_str), namespace)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"
        except Exception as e:
            return False, f"Error al compilar: {e}"

        if "tool_fn" not in namespace:
            return False, "El código no define 'tool_fn(omega, candidates, context)'"

        fn = namespace["tool_fn"]
        self.register(name, fn, description=description, invented=True)
        return True, f"Herramienta '{name}' registrada correctamente"

    # ── Feedback ──────────────────────────────────────────────────────────────

    def report_outcome(self, tool_name: str, reward: float) -> None:
        if tool_name in self._tools:
            self._tools[tool_name].record(reward)
            # Auto-desactivar herramientas con muy bajo rendimiento sostenido
            t = self._tools[tool_name]
            if len(t.performance) >= 5 and t.avg_performance(5) < -10:
                t.active = False

    # ── Resumen ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        return {
            "total": len(self._tools),
            "active": sum(1 for t in self._tools.values() if t.active),
            "invented": sum(1 for t in self._tools.values() if t.invented),
            "tools": [t.summary() for t in self._tools.values()],
        }

    def list_tools(self) -> List[str]:
        return list(self._tools.keys())
