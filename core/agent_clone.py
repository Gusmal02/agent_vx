"""
agent_clone.py — Mecanismo de clonación y prueba de mejoras

El agente puede clonarse a sí mismo para probar cambios sin afectar
su estado real. Si el clon demuestra mejora → se aplican los cambios al original.

Flujo:
  1. agente.clone()                        → clon aislado
  2. apply_change(clon, cambio_propuesto)  → modifica el clon
  3. evaluar(clon, env)                    → medir resultado
  4. si mejora → apply_change(original, cambio_propuesto)

Tipos de cambios que el clon puede probar:
  - Hiperparámetros (confidence_threshold, epsilon, N, etc.)
  - Nueva herramienta (código Python)
  - Ajuste de pesos del tejido (copia de tensores)
"""

import copy
import torch
from typing import Any, Dict, Optional, Tuple


# ── Tipos de cambio ───────────────────────────────────────────────────────────

class HyperparamChange:
    """Cambio de un hiperparámetro."""
    def __init__(self, param: str, value: Any):
        self.param = param
        self.value = value

    def __repr__(self):
        return f"HyperparamChange({self.param}={self.value})"


class ToolChange:
    """Agregar o reemplazar una herramienta (código Python)."""
    def __init__(self, name: str, code: str, description: str = ""):
        self.name = name
        self.code = code
        self.description = description

    def __repr__(self):
        return f"ToolChange(name={self.name})"


class WeightChange:
    """Ajuste de un tensor de pesos del tejido."""
    def __init__(self, attr_path: str, delta: torch.Tensor):
        self.attr_path = attr_path   # ej: "tissue.W_perc"
        self.delta = delta

    def __repr__(self):
        return f"WeightChange({self.attr_path})"


Change = HyperparamChange | ToolChange | WeightChange


# ── Aplicador de cambios ──────────────────────────────────────────────────────

def apply_change(agent, change: Change) -> Tuple[bool, str]:
    """
    Aplica un cambio a un agente (original o clon).
    Devuelve (éxito, mensaje).
    """
    if isinstance(change, HyperparamChange):
        if hasattr(agent, change.param):
            setattr(agent, change.param, change.value)
            return True, f"  param '{change.param}' → {change.value}"
        return False, f"  '{change.param}' no existe en el agente"

    elif isinstance(change, ToolChange):
        if hasattr(agent, 'tool_registry') and agent.tool_registry is not None:
            ok, msg = agent.tool_registry.invent(
                change.name, change.code, change.description
            )
            return ok, msg
        return False, "El agente no tiene tool_registry"

    elif isinstance(change, WeightChange):
        parts = change.attr_path.split(".")
        obj = agent
        try:
            for p in parts[:-1]:
                obj = getattr(obj, p)
            attr = parts[-1]
            tensor = getattr(obj, attr)
            with torch.no_grad():
                tensor.add_(change.delta)
            return True, f"  peso '{change.attr_path}' actualizado"
        except AttributeError as e:
            return False, f"  ruta inválida '{change.attr_path}': {e}"

    return False, f"Tipo de cambio desconocido: {type(change)}"


# ── Clonador ──────────────────────────────────────────────────────────────────

class AgentClone:
    """
    Gestor del ciclo clonar → probar → decidir.

    Uso:
        cloner = AgentClone(agent_original)
        clone  = cloner.make_clone()
        apply_change(clone, HyperparamChange("confidence_threshold", 0.85))
        # evaluar clone en entorno...
        if mejora:
            cloner.promote(cambio)
    """

    def __init__(self, original):
        self._original = original

    def make_clone(self):
        """
        Crea una copia del agente original.
        La biblioteca SQLite no es serializable con deepcopy — se guarda aparte,
        se clona el agente sin ella, y luego se restaura la referencia.
        El clon tiene _clone_mode=True para no escribir en la biblioteca.
        """
        original_lib = self._original.library
        original_mem = self._original.memory_module

        # Retirar temporalmente los objetos no-serializables
        self._original.library = None
        self._original.memory_module = None

        clone = copy.deepcopy(self._original)

        # Restaurar en original y en clon (referencia compartida, solo lectura en clon)
        self._original.library       = original_lib
        self._original.memory_module = original_mem
        clone.library                = original_lib
        clone.memory_module          = original_mem

        clone._clone_mode = True
        return clone

    def promote(self, change: Change) -> Tuple[bool, str]:
        """Aplica el cambio probado en el clon al agente original."""
        return apply_change(self._original, change)

    def evaluate_changes(self,
                         changes: list,
                         evaluator_fn,
                         env_name: str,
                         n_episodes: int = 10,
                         verbose: bool = False,
                         ) -> Dict:
        """
        Construye un clon, aplica todos los cambios de la lista,
        evalúa con evaluator_fn y devuelve el resultado.

        evaluator_fn(agent, env_name, n_episodes) → {"recompensa_media": float, ...}
        """
        clone = self.make_clone()
        applied = []
        failed = []
        for ch in changes:
            ok, msg = apply_change(clone, ch)
            if ok:
                applied.append(str(ch))
            else:
                failed.append(f"{ch}: {msg}")

        if verbose:
            print(f"  [Clon] Cambios aplicados: {applied}")
            if failed:
                print(f"  [Clon] Fallidos: {failed}")

        result = evaluator_fn(clone, env_name, n_episodes)
        result["clone_changes"] = applied
        result["clone_failures"] = failed
        return result
