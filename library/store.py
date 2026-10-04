"""
library/store.py — ResonantLibrary (db-rmf Nivel 0)

Almacén persistente de episodios y patrones de decisión.
Sobrevive entre versiones del agente — el conocimiento no se pierde al actualizar.

Consultas soportadas:
  C1 resonancia  : acción más similar al estado actual (filtrada por env)
  C2 novedad     : episodios más diferentes (para exploración)
  C3 temporal    : episodios más recientes
  C4 frontera    : episodios con mayor recompensa acumulada

v0.0.4: env_name en cada registro. Las queries C1/C4 filtran por entorno activo.
        Registros sin env_name heredan 'unknown' para compatibilidad con v0.0.2/v0.0.3.
"""

import sqlite3
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class ResonantLibrary:
    """
    Biblioteca resonante persistente basada en SQLite.

    Cada registro es una tupla (env_name, omega, action_idx, reward, regime, version, timestamp).
    Las consultas C1-C4 filtran por env_name cuando se provee.
    """

    def __init__(self, db_path: str = "library/resonant_db.sqlite"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS episodes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                omega       TEXT    NOT NULL,
                action_idx  INTEGER NOT NULL,
                reward      REAL    NOT NULL,
                regime      TEXT    DEFAULT 'resonant',
                version     TEXT    DEFAULT 'unknown',
                timestamp   REAL    NOT NULL,
                env_name    TEXT    DEFAULT 'unknown'
            )
        """)
        # Migración: añade env_name si la DB ya existe sin esa columna
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(episodes)").fetchall()}
        if "env_name" not in cols:
            self.conn.execute("ALTER TABLE episodes ADD COLUMN env_name TEXT DEFAULT 'unknown'")
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_reward ON episodes(reward DESC)
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_env ON episodes(env_name)
        """)
        self.conn.commit()

    # ── Escritura ─────────────────────────────────────────────────────────────

    def store(self,
              omega: torch.Tensor,
              action_idx: int,
              reward: float,
              regime: str = "resonant",
              version: str = "unknown",
              env_name: str = "unknown") -> None:
        """Guarda un paso de episodio con su recompensa normalizada."""
        self.conn.execute(
            "INSERT INTO episodes (omega, action_idx, reward, regime, version, timestamp, env_name) "
            "VALUES (?,?,?,?,?,?,?)",
            (json.dumps(omega.tolist()), int(action_idx), float(reward),
             regime, version, time.time(), env_name)
        )
        self.conn.commit()

    def store_episode(self,
                      steps: List[Tuple[torch.Tensor, int]],
                      total_reward: float,
                      reward_max: float = 500.0,
                      regime: str = "resonant",
                      version: str = "unknown",
                      env_name: str = "unknown") -> None:
        """Guarda todos los pasos de un episodio con su recompensa normalizada."""
        norm = total_reward / max(1.0, abs(reward_max))
        rows = [
            (json.dumps(omega.tolist()), int(action_idx), norm,
             regime, version, time.time() + i * 1e-6, env_name)
            for i, (omega, action_idx) in enumerate(steps)
        ]
        self.conn.executemany(
            "INSERT INTO episodes (omega, action_idx, reward, regime, version, timestamp, env_name) "
            "VALUES (?,?,?,?,?,?,?)",
            rows
        )
        self.conn.commit()

    # ── Consultas ─────────────────────────────────────────────────────────────

    def count(self, env_name: str = None) -> int:
        if env_name:
            return self.conn.execute(
                "SELECT COUNT(*) FROM episodes WHERE env_name=?", (env_name,)
            ).fetchone()[0]
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def _load_all(self, env_name: str = None) -> List[Tuple[torch.Tensor, int, float]]:
        if env_name:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes WHERE env_name=?",
                (env_name,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes"
            ).fetchall()
        return [(torch.tensor(json.loads(r[0]), dtype=torch.float32), r[1], r[2])
                for r in rows]

    def query_resonance(self,
                        omega: torch.Tensor,
                        min_samples: int = 10,
                        env_name: str = None,
                        ) -> Optional[Tuple[int, float, float]]:
        """
        C1 — Acción más resonante con el estado actual, filtrada por entorno.

        Devuelve (action_idx, cos_similarity, reward_normalizado) o None si
        la biblioteca tiene menos de min_samples registros para ese entorno.
        """
        rows = self._load_all(env_name=env_name)
        if len(rows) < min_samples:
            return None

        best_cos, best_action, best_reward = -2.0, 0, 0.0
        for vec, action, reward in rows:
            cos = float(F.cosine_similarity(omega.unsqueeze(0), vec.unsqueeze(0)))
            if cos > best_cos:
                best_cos, best_action, best_reward = cos, action, reward

        return (best_action, best_cos, best_reward)

    def query_novelty(self,
                      omega: torch.Tensor,
                      n: int = 3,
                      env_name: str = None,
                      ) -> List[Tuple[torch.Tensor, int, float]]:
        """C2 — Los n episodios más diferentes al omega dado (para exploración)."""
        rows = self._load_all(env_name=env_name)
        if not rows:
            return []
        scored = [
            (float(F.cosine_similarity(omega.unsqueeze(0), vec.unsqueeze(0))),
             vec, action, reward)
            for vec, action, reward in rows
        ]
        scored.sort(key=lambda x: x[0])
        return [(s[1], s[2], s[3]) for s in scored[:n]]

    def query_recent(self, n: int = 10, env_name: str = None) -> List[Tuple[torch.Tensor, int, float]]:
        """C3 — n episodios más recientes."""
        if env_name:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes "
                "WHERE env_name=? ORDER BY timestamp DESC LIMIT ?",
                (env_name, n)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes ORDER BY timestamp DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [(torch.tensor(json.loads(r[0]), dtype=torch.float32), r[1], r[2])
                for r in rows]

    def query_frontier(self, n: int = 10, env_name: str = None) -> List[Tuple[torch.Tensor, int, float]]:
        """C4 — n episodios con mayor recompensa (frontera de conocimiento)."""
        if env_name:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes "
                "WHERE env_name=? ORDER BY reward DESC LIMIT ?",
                (env_name, n)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT omega, action_idx, reward FROM episodes ORDER BY reward DESC LIMIT ?",
                (n,)
            ).fetchall()
        return [(torch.tensor(json.loads(r[0]), dtype=torch.float32), r[1], r[2])
                for r in rows]

    def envs_summary(self) -> dict:
        """Resumen de entornos registrados y cantidad de episodios por env."""
        rows = self.conn.execute(
            "SELECT env_name, COUNT(*), AVG(reward), MAX(reward) "
            "FROM episodes GROUP BY env_name"
        ).fetchall()
        return {r[0]: {"count": r[1], "avg_reward": round(r[2], 4), "max_reward": round(r[3], 4)}
                for r in rows}

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def summary(self, env_name: str = None) -> dict:
        n = self.count(env_name=env_name)
        if n == 0:
            return {"total": 0, "avg_reward": 0.0, "max_reward": 0.0,
                    "versions": [], "regimes": {}, "envs": {}}
        if env_name:
            row = self.conn.execute(
                "SELECT AVG(reward), MAX(reward) FROM episodes WHERE env_name=?",
                (env_name,)
            ).fetchone()
            versions = [r[0] for r in self.conn.execute(
                "SELECT DISTINCT version FROM episodes WHERE env_name=?", (env_name,)
            ).fetchall()]
            regimes = dict(self.conn.execute(
                "SELECT regime, COUNT(*) FROM episodes WHERE env_name=? GROUP BY regime",
                (env_name,)
            ).fetchall())
        else:
            row = self.conn.execute(
                "SELECT AVG(reward), MAX(reward) FROM episodes"
            ).fetchone()
            versions = [r[0] for r in self.conn.execute(
                "SELECT DISTINCT version FROM episodes"
            ).fetchall()]
            regimes = dict(self.conn.execute(
                "SELECT regime, COUNT(*) FROM episodes GROUP BY regime"
            ).fetchall())
        return {
            "total":      n,
            "avg_reward": round(row[0], 4),
            "max_reward": round(row[1], 4),
            "versions":   versions,
            "regimes":    regimes,
            "envs":       self.envs_summary(),
        }

    def close(self) -> None:
        self.conn.close()
