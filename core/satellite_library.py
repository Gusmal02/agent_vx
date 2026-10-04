"""
core/satellite_library.py — Biblioteca con procedencia de rama satélite

Cada agente de primer orden (satélite) escribe en su propia biblioteca
temporal. Cuando su resultado supera el umbral, sus entradas se fusionan
en la biblioteca principal con una etiqueta de procedencia.

Esto evita que exploraciones fallidas contaminen la biblioteca del maestro.
"""

import sqlite3
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, List
from library.store import ResonantLibrary


class SatelliteLibrary:
    """
    Biblioteca temporal para un agente de primer orden.

    Escribe en un archivo SQLite separado (satélite).
    Merge condicional: solo si la rama supera el umbral.
    """

    def __init__(self, satellite_id: str, base_dir: str = "library/satellites"):
        Path(base_dir).mkdir(parents=True, exist_ok=True)
        self._db_path = f"{base_dir}/{satellite_id}.sqlite"
        self._lib = ResonantLibrary(db_path=self._db_path)
        self.satellite_id = satellite_id
        self._entry_count = 0

    def store(self, omega, action_idx, reward, regime="resonant",
              version="v0.0.8", env_name="unknown"):
        self._lib.store(omega, action_idx, reward, regime, version, env_name)
        self._entry_count += 1

    def store_episode(self, steps, total_reward, reward_max=500.0,
                      regime="resonant", version="v0.0.8", env_name="unknown"):
        self._lib.store_episode(steps, total_reward, reward_max=reward_max,
                                regime=regime, version=version, env_name=env_name)

    def query_resonance(self, omega, min_samples=10, env_name=None):
        return self._lib.query_resonance(omega, min_samples=min_samples, env_name=env_name)

    def count(self, env_name=None):
        return self._lib.count(env_name=env_name)

    def merge_into(self, main_library: ResonantLibrary,
                   min_reward_percentile: float = 0.6) -> int:
        """
        Fusiona las entradas buenas del satélite en la biblioteca principal.
        Solo transfiere entradas con reward >= percentil de la rama.
        Devuelve cuántas entradas se transfirieron.
        """
        conn = sqlite3.connect(self._db_path)
        rows = conn.execute(
            "SELECT omega_x, omega_y, omega_z, action_idx, reward, regime, version, env_name "
            "FROM resonance_library ORDER BY reward DESC"
        ).fetchall()
        conn.close()

        if not rows:
            return 0

        rewards = [r[4] for r in rows]
        threshold = sorted(rewards)[int(len(rewards) * (1 - min_reward_percentile))]

        transferred = 0
        main_conn = sqlite3.connect(main_library.db_path)
        for row in rows:
            ox, oy, oz, action_idx, reward, regime, version, env_name = row
            if reward < threshold:
                continue
            omega = torch.tensor([ox, oy, oz], dtype=torch.float32)
            omega = F.normalize(omega, dim=-1)
            main_conn.execute(
                "INSERT INTO resonance_library "
                "(omega_x, omega_y, omega_z, action_idx, reward, regime, version, env_name) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (float(omega[0]), float(omega[1]), float(omega[2]),
                 action_idx, reward, f"{regime}:sat_{self.satellite_id}",
                 version, env_name)
            )
            transferred += 1
        main_conn.commit()
        main_conn.close()
        return transferred

    def summary(self) -> dict:
        s = self._lib.summary()
        s["satellite_id"] = self.satellite_id
        return s
