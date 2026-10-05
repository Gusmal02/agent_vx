"""
blackboard.py — Pizarra del agente.

Estado propio: lo que el agente computa y observa.
No es el corpus (referencia), no es el oráculo (consulta).
Es la memoria de trabajo del agente.
"""

import json
import sqlite3
import time
from pathlib import Path


class Blackboard:
    """
    Estado propio del agente. Persiste en SQLite.
    Tres tablas:
      observations  — resultados crudos de cada herramienta
      zeros         — ceros verificados con σ, t, desviación
      conjectures   — hipótesis que el agente formula por sí mismo
    """

    def __init__(self, agent_id: str, problem: str, db_dir: str = "results"):
        Path(db_dir).mkdir(exist_ok=True)
        self.agent_id = agent_id
        self.problem  = problem
        self._db = sqlite3.connect(
            f"{db_dir}/blackboard_{problem}_{agent_id}.sqlite",
            check_same_thread=False,
        )
        self._init_schema()

    def _init_schema(self):
        self._db.executescript("""
        CREATE TABLE IF NOT EXISTS observations (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            tool      TEXT NOT NULL,
            t_min     REAL,
            t_max     REAL,
            result    TEXT NOT NULL,
            surprise  INTEGER DEFAULT 0,
            ts        REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS zeros (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            sigma     REAL NOT NULL,
            t         REAL NOT NULL,
            deviation REAL NOT NULL,
            on_line   INTEGER NOT NULL,
            ts        REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS conjectures (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            statement TEXT NOT NULL,
            evidence  TEXT,
            confidence REAL DEFAULT 0.5,
            ts        REAL DEFAULT (strftime('%s','now'))
        );
        CREATE TABLE IF NOT EXISTS oracle_answers (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            question  TEXT NOT NULL,
            answer    TEXT NOT NULL,
            ts        REAL DEFAULT (strftime('%s','now'))
        );
        """)
        self._db.commit()

    # ── Observaciones ─────────────────────────────────────────────────────────

    def record(self, tool: str, result: dict,
               t_min: float = None, t_max: float = None,
               surprise: bool = False):
        self._db.execute(
            "INSERT INTO observations (tool, t_min, t_max, result, surprise) VALUES (?,?,?,?,?)",
            (tool, t_min, t_max, json.dumps(result), int(surprise))
        )
        self._db.commit()

    def recent_observations(self, n: int = 10) -> list[dict]:
        rows = self._db.execute(
            "SELECT tool, t_min, t_max, result, surprise FROM observations "
            "ORDER BY ts DESC LIMIT ?", (n,)
        ).fetchall()
        return [{"tool": r[0], "t_min": r[1], "t_max": r[2],
                 "result": json.loads(r[3]), "surprise": bool(r[4])}
                for r in rows]

    # ── Ceros ─────────────────────────────────────────────────────────────────

    def add_zero(self, sigma: float, t: float, deviation: float, on_line: bool) -> bool:
        """Agrega cero si no existe ya (tolerancia 0.05). Retorna True si fue nuevo."""
        existing = self._db.execute(
            "SELECT id FROM zeros WHERE ABS(t - ?) < 0.05", (t,)
        ).fetchone()
        if existing:
            return False
        self._db.execute(
            "INSERT INTO zeros (sigma, t, deviation, on_line) VALUES (?,?,?,?)",
            (sigma, t, deviation, int(on_line))
        )
        self._db.commit()
        return True

    def all_zeros(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT sigma, t, deviation, on_line FROM zeros ORDER BY t"
        ).fetchall()
        return [{"sigma": r[0], "t": r[1], "deviation": r[2], "on_line": bool(r[3])}
                for r in rows]

    def frontier_t(self, default: float = 50.0) -> float:
        """El t más alto que el agente ha explorado."""
        row = self._db.execute(
            "SELECT MAX(t_max) FROM observations WHERE t_max IS NOT NULL"
        ).fetchone()
        val = row[0] if row and row[0] else None
        if val is None:
            row2 = self._db.execute("SELECT MAX(t) FROM zeros").fetchone()
            val  = row2[0] if row2 and row2[0] else None
        return val if val else default

    def explored_ranges(self) -> list[tuple]:
        rows = self._db.execute(
            "SELECT t_min, t_max FROM observations WHERE t_min IS NOT NULL "
            "ORDER BY t_min"
        ).fetchall()
        return [(r[0], r[1]) for r in rows]

    def already_explored(self, t_min: float, t_max: float) -> bool:
        """¿Ya exploramos una región que cubre [t_min, t_max]?"""
        row = self._db.execute(
            "SELECT id FROM observations WHERE t_min <= ? AND t_max >= ?",
            (t_min, t_max)
        ).fetchone()
        return row is not None

    # ── Conjeturas ────────────────────────────────────────────────────────────

    def add_conjecture(self, statement: str, evidence: str = None,
                       confidence: float = 0.5):
        self._db.execute(
            "INSERT INTO conjectures (statement, evidence, confidence) VALUES (?,?,?)",
            (statement, evidence, confidence)
        )
        self._db.commit()

    def conjectures(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT statement, evidence, confidence FROM conjectures ORDER BY confidence DESC"
        ).fetchall()
        return [{"statement": r[0], "evidence": r[1], "confidence": r[2]}
                for r in rows]

    # ── Oráculo ───────────────────────────────────────────────────────────────

    def record_oracle(self, question: str, answer: str):
        self._db.execute(
            "INSERT INTO oracle_answers (question, answer) VALUES (?,?)",
            (question, answer)
        )
        self._db.commit()

    # ── Resumen para epistemic state ──────────────────────────────────────────

    def to_epistemic_state(self) -> dict:
        zeros = self.all_zeros()
        conjs = self.conjectures()
        obs   = self.recent_observations(20)
        off_line = [z for z in zeros if not z["on_line"]]

        supported = {}
        if zeros:
            on_pct = sum(1 for z in zeros if z["on_line"]) / len(zeros)
            supported["zeros_on_critical_line"] = {
                "score_mean": round(on_pct, 4),
                "statement":  f"{len(zeros)} ceros verificados, {on_pct*100:.1f}% en σ=0.5",
                "agent_id":   self.agent_id,
            }
        if off_line:
            supported["zeros_off_critical_line"] = {
                "score_mean": 0.0,
                "statement":  f"{len(off_line)} ceros con σ≠0.5 — posibles contraejemplos",
                "agent_id":   self.agent_id,
            }
        for c in conjs[:3]:
            key = c["statement"][:40].replace(" ", "_")
            supported[key] = {
                "score_mean": c["confidence"],
                "statement":  c["statement"],
                "agent_id":   self.agent_id,
            }

        return {
            "agent_id":             self.agent_id,
            "problem":              self.problem,
            "supported_hypotheses": supported,
            "cross_hypotheses":     [],
            "zeros_found":          len(zeros),
            "frontier_t":           self.frontier_t(),
            "off_line_zeros":       off_line,
            "conjectures":          conjs,
            "recent_observations":  obs[:5],
        }
