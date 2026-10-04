"""
core/checkpoint.py — Sistema de checkpoints periódicos

Guarda el estado completo del agente cada N minutos o N iteraciones.
Permite reanudar una sesión larga (v0.0.8 corre 4-5 horas) desde
el último punto de control sin perder progreso.

Estado guardado:
  - Pesos del tejido (exec_ids, perc_ids, etc.)
  - Configuración actual
  - Historial de iteraciones
  - ToolRegistry (herramientas inventadas)
  - HypothesisLog
  - SatisfactionMeter state
"""

import json
import time
import torch
import copy
from pathlib import Path
from typing import Optional


CHECKPOINT_DIR = Path("checkpoints")


class CheckpointManager:

    def __init__(self,
                 session_id: str,
                 interval_minutes: float = 15.0):
        self.session_id      = session_id
        self.interval_sec    = interval_minutes * 60
        self._last_save      = time.time()
        self._save_count     = 0
        CHECKPOINT_DIR.mkdir(exist_ok=True)

    # ── Guardar ───────────────────────────────────────────────────────────────

    def save(self, state: dict, force: bool = False) -> Optional[Path]:
        """
        Guarda state si ha pasado interval_sec desde el último save.
        Con force=True guarda incondicionalmente.
        Devuelve la ruta del archivo o None si no se guardó.
        """
        now = time.time()
        if not force and (now - self._last_save) < self.interval_sec:
            return None

        self._save_count += 1
        filename = CHECKPOINT_DIR / f"{self.session_id}_ck{self._save_count:04d}.pt"

        # Separar tensores del resto (torch.save para tensores, json no puede)
        tensors = state.pop("_tensors", {})
        meta    = _make_serializable(state)

        torch.save({"meta": meta, "tensors": tensors}, filename)
        self._last_save = now
        print(f"  [Checkpoint] Guardado → {filename.name}")
        return filename

    def save_agent_tissue(self, agent) -> dict:
        """Extrae los tensores del tejido del agente como dict serializable."""
        tensors = {}
        for i, node in enumerate(agent.tissue.nodes):
            tensors[f"node_{i}_q"]       = node.q.clone()
            tensors[f"node_{i}_q_local"] = node.q_local.clone()
        for (i, j), edge in agent.tissue.edges.items():
            tensors[f"edge_{i}_{j}_r"] = edge.r.clone()
        return tensors

    # ── Cargar ────────────────────────────────────────────────────────────────

    def load_latest(self) -> Optional[dict]:
        """Carga el checkpoint más reciente de esta sesión."""
        pattern = list(CHECKPOINT_DIR.glob(f"{self.session_id}_ck*.pt"))
        if not pattern:
            return None
        latest = max(pattern, key=lambda p: p.stat().st_mtime)
        data = torch.load(latest, weights_only=False)
        print(f"  [Checkpoint] Cargado ← {latest.name}")
        return data

    def restore_agent_tissue(self, agent, tensors: dict) -> None:
        """Restaura los tensores del tejido en el agente."""
        for i, node in enumerate(agent.tissue.nodes):
            if f"node_{i}_q" in tensors:
                node.q       = tensors[f"node_{i}_q"].clone()
                node.q_local = tensors[f"node_{i}_q_local"].clone()
        for (i, j), edge in agent.tissue.edges.items():
            key = f"edge_{i}_{j}_r"
            if key in tensors:
                edge.r = tensors[key].clone()

    def list_checkpoints(self) -> list:
        return sorted(CHECKPOINT_DIR.glob(f"{self.session_id}_ck*.pt"))

    def due(self) -> bool:
        return (time.time() - self._last_save) >= self.interval_sec


def _make_serializable(obj):
    """Convierte recursivamente tensores y tipos no-JSON a serializables."""
    if isinstance(obj, torch.Tensor):
        return {"__tensor__": True, "data": obj.tolist()}
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        return str(obj)
