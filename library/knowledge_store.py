"""
library/knowledge_store.py — KnowledgeStore (db-rmf Nivel 0 — dominio matemático)

Análogo a ResonantLibrary pero para conocimiento de investigación en vez de episodios Gym.

Queries disponibles (espejo de C1-C4 de ResonantLibrary):
  K1 resonancia  : hallazgo más cercano al tema actual (matching por tags)
  K2 novedad     : temas menos explorados (para elegir siguiente experimento)
  K3 temporal    : hallazgos más recientes (contexto de la sesión)
  K4 frontera    : entradas con mayor confianza (lo mejor conocido)

Tres tipos de entrada:
  known       — verificado por humanos, skip_verify=1, confidence=1.0
  discovery   — encontrado por el agente, confidence variable
  failed      — experimento sin resultado, para evitar repetir

Satellite: cada worker corre con SatelliteKnowledgeStore. Al terminar, el coordinador
fusiona al store principal solo las entradas con confidence > percentil 60 de esa rama.
"""

import sqlite3
import json
import time
import hashlib
from pathlib import Path
from typing import Optional


# ── KnowledgeStore ────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    Base de conocimiento matemático persistente.
    Cada entrada: (domain, kind, topic_tags, statement, code_hash, confidence,
                   source, skip_verify, saved_at, session_id)
    """

    VERSION = "0.1.0"

    def __init__(self, db_path: str = "library/knowledge.sqlite"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn    = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                domain      TEXT    NOT NULL,
                kind        TEXT    NOT NULL DEFAULT 'discovery',
                topic_tags  TEXT    NOT NULL DEFAULT '[]',
                statement   TEXT    NOT NULL,
                implication TEXT    DEFAULT '',
                code_hash   TEXT    DEFAULT '',
                confidence  REAL    NOT NULL DEFAULT 0.5,
                source      TEXT    DEFAULT 'agent',
                skip_verify INTEGER DEFAULT 0,
                saved_at    REAL    NOT NULL,
                session_id  TEXT    DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_domain   ON knowledge(domain);
            CREATE INDEX IF NOT EXISTS idx_kind     ON knowledge(kind);
            CREATE INDEX IF NOT EXISTS idx_conf     ON knowledge(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_hash     ON knowledge(code_hash);
        """)
        self.conn.commit()

    # ── Escritura ─────────────────────────────────────────────────────────────

    def store(self,
              domain:      str,
              kind:        str,       # 'known' | 'discovery' | 'failed'
              tags:        list[str],
              statement:   str,
              confidence:  float = 0.5,
              implication: str   = "",
              code:        str   = "",
              source:      str   = "agent",
              skip_verify: bool  = False,
              session_id:  str   = "",
              ) -> int:
        """Guarda una entrada. Devuelve el ID."""
        code_hash = hashlib.md5(code.strip().encode()).hexdigest()[:12] if code else ""
        # Evitar duplicados exactos (mismo hash)
        if code_hash:
            dup = self.conn.execute(
                "SELECT id FROM knowledge WHERE code_hash=? AND domain=?",
                (code_hash, domain)
            ).fetchone()
            if dup:
                return dup[0]

        cur = self.conn.execute(
            """INSERT INTO knowledge
               (domain, kind, topic_tags, statement, implication, code_hash,
                confidence, source, skip_verify, saved_at, session_id)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (domain, kind, json.dumps(tags), statement, implication,
             code_hash, float(confidence), source, int(skip_verify),
             time.time(), session_id),
        )
        self.conn.commit()
        return cur.lastrowid

    def bulk_store(self, entries: list[dict]) -> int:
        """Inserta múltiples entradas de una vez (para seed inicial)."""
        count = 0
        for e in entries:
            self.store(**e)
            count += 1
        return count

    # ── Queries K1-K4 ─────────────────────────────────────────────────────────

    def K1_resonance(self, domain: str, topic_tags: list[str],
                     kind_filter: str | None = None,
                     top_n: int = 5) -> list[dict]:
        """
        K1 resonancia: entradas cuyas tags solapan con los tags de la consulta.
        Ordenadas por overlap (desc), luego confianza (desc).
        """
        rows = self._fetch_domain(domain, kind_filter)
        scored = []
        query_set = set(t.lower() for t in topic_tags)
        for row in rows:
            tags_row = set(t.lower() for t in json.loads(row["topic_tags"]))
            overlap  = len(query_set & tags_row) / max(1, len(query_set | tags_row))
            if overlap > 0:
                scored.append((overlap, row["confidence"], row))
        scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [r for _, _, r in scored[:top_n]]

    def K2_novelty(self, domain: str, explored_tags: list[str],
                   top_n: int = 5) -> list[dict]:
        """
        K2 novedad: temas cuyas tags tienen MENOR solapamiento con lo ya explorado.
        Útil para elegir el siguiente experimento.
        """
        rows = self._fetch_domain(domain, kind_filter="discovery")
        explored_set = set(t.lower() for t in explored_tags)
        scored = []
        for row in rows:
            tags_row = set(t.lower() for t in json.loads(row["topic_tags"]))
            overlap  = len(explored_set & tags_row) / max(1, len(explored_set | tags_row))
            novelty  = 1.0 - overlap
            scored.append((novelty, row))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_n]]

    def K3_temporal(self, domain: str, n: int = 10,
                    min_confidence: float = 0.0) -> list[dict]:
        """K3 temporal: N entradas más recientes para el dominio."""
        cur = self.conn.execute(
            """SELECT * FROM knowledge
               WHERE domain=? AND confidence>=?
               ORDER BY saved_at DESC LIMIT ?""",
            (domain, min_confidence, n)
        )
        return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]

    def K4_frontier(self, domain: str, top_n: int = 10,
                    min_confidence: float = 0.75) -> list[dict]:
        """K4 frontera: entradas con mayor confianza (lo mejor conocido)."""
        cur = self.conn.execute(
            """SELECT * FROM knowledge
               WHERE domain=? AND kind != 'failed' AND confidence>=?
               ORDER BY confidence DESC, saved_at DESC LIMIT ?""",
            (domain, min_confidence, top_n)
        )
        return [dict(zip([d[0] for d in cur.description], row)) for row in cur.fetchall()]

    def should_skip(self, domain: str, topic_tags: list[str]) -> Optional[dict]:
        """¿Hay algo conocido con skip_verify=1 que cubra estos tags? Si sí, no re-verificar."""
        matches = self.K1_resonance(domain, topic_tags, kind_filter="known", top_n=1)
        if matches and matches[0].get("skip_verify"):
            return matches[0]
        return None

    def already_ran(self, domain: str, code: str) -> Optional[dict]:
        """¿Ya corrimos este código exacto?"""
        code_hash = hashlib.md5(code.strip().encode()).hexdigest()[:12]
        cur = self.conn.execute(
            "SELECT * FROM knowledge WHERE code_hash=? AND domain=?",
            (code_hash, domain)
        )
        row = cur.fetchone()
        if row:
            return dict(zip([d[0] for d in cur.description], row))
        return None

    # ── Context para oracle ───────────────────────────────────────────────────

    def to_oracle_context(self, domain: str, topic_tags: list[str] | None = None,
                          max_chars: int = 1400) -> str:
        """
        Resumen compacto para prepender al prompt del oracle.
        Prioriza: known (skip) → frontier discoveries → recientes → failed.
        """
        parts = []

        # Conocidos: siempre mostrar
        known = self._fetch_domain(domain, kind_filter="known", order_by_conf=True, limit=6)
        if known:
            parts.append("HECHOS CONOCIDOS (no re-verificar):")
            for k in known:
                skip = " [SKIP]" if k.get("skip_verify") else ""
                parts.append(f"  • {k['statement']}{skip}")
                if k.get("implication"):
                    parts.append(f"    → {k['implication']}")

        # Top discoveries por confianza
        frontier = self.K4_frontier(domain, top_n=4, min_confidence=0.75)
        if frontier:
            parts.append("\nHALLAZGOS ROBUSTOS:")
            for d in frontier:
                parts.append(f"  ✓ [{d['confidence']:.2f}] {d['statement']}")

        # Resonancia con tema actual
        if topic_tags:
            resonant = self.K1_resonance(domain, topic_tags, kind_filter="discovery", top_n=3)
            if resonant:
                parts.append("\nRELEVANTE AL TEMA ACTUAL:")
                for r in resonant:
                    parts.append(f"  ~ [{r['confidence']:.2f}] {r['statement']}")

        # Fallidos: evitar repetir
        failed = self._fetch_domain(domain, kind_filter="failed", limit=3)
        if failed:
            parts.append("\nYA INTENTADO SIN RESULTADO:")
            for f in failed:
                parts.append(f"  ✗ {f['statement'][:100]}")

        return "\n".join(parts)[:max_chars]

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def summary(self, domain: str) -> dict:
        cur = self.conn.execute(
            """SELECT kind, COUNT(*) as n, AVG(confidence) as avg_conf
               FROM knowledge WHERE domain=? GROUP BY kind""",
            (domain,)
        )
        rows = cur.fetchall()
        return {row[0]: {"n": row[1], "avg_conf": round(row[2], 3)} for row in rows}

    # ── Interno ───────────────────────────────────────────────────────────────

    def _fetch_domain(self, domain: str, kind_filter: str | None = None,
                      order_by_conf: bool = False, limit: int = 100) -> list[dict]:
        q = "SELECT * FROM knowledge WHERE domain=?"
        params: list = [domain]
        if kind_filter:
            q += " AND kind=?"
            params.append(kind_filter)
        q += " ORDER BY " + ("confidence DESC" if order_by_conf else "saved_at DESC")
        q += f" LIMIT {limit}"
        cur = self.conn.execute(q, params)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


# ── SatelliteKnowledgeStore ───────────────────────────────────────────────────

class SatelliteKnowledgeStore(KnowledgeStore):
    """
    Copia temporal por worker. Misma API que KnowledgeStore.
    Al terminar, el coordinador fusiona al store principal si
    los hallazgos superan el percentil 60 de confianza de esa rama.
    """

    def __init__(self, agent_id: str, domain: str,
                 base_dir: str = "library"):
        db_path = f"{base_dir}/knowledge_{domain}_{agent_id}.sqlite"
        super().__init__(db_path)
        self.agent_id = agent_id
        self.domain   = domain

    def merge_into(self, main_store: KnowledgeStore,
                   percentile_threshold: float = 60.0) -> int:
        """
        Fusiona hallazgos al store principal.
        Solo transfiere entradas cuya confianza supera el percentil `threshold`
        de todas las entradas de tipo 'discovery' en esta rama.
        Devuelve el número de entradas fusionadas.
        """
        discoveries = self._fetch_domain(self.domain, kind_filter="discovery")
        if not discoveries:
            return 0

        confs = sorted(d["confidence"] for d in discoveries)
        idx   = max(0, int(len(confs) * percentile_threshold / 100) - 1)
        threshold = confs[idx]

        merged = 0
        for entry in discoveries:
            if entry["confidence"] >= threshold:
                main_store.store(
                    domain      = entry["domain"],
                    kind        = "discovery",
                    tags        = json.loads(entry["topic_tags"]),
                    statement   = entry["statement"],
                    confidence  = entry["confidence"],
                    implication = entry.get("implication", ""),
                    code        = "",
                    source      = f"satellite:{self.agent_id}",
                    session_id  = entry.get("session_id", ""),
                )
                merged += 1

        # También transferir los failed (siempre útiles para evitar repetición)
        for entry in self._fetch_domain(self.domain, kind_filter="failed"):
            main_store.store(
                domain    = entry["domain"],
                kind      = "failed",
                tags      = json.loads(entry["topic_tags"]),
                statement = entry["statement"],
                source    = f"satellite:{self.agent_id}",
            )

        return merged

    def cleanup(self) -> None:
        """Elimina el archivo SQLite temporal."""
        import os
        self.close()
        try:
            os.unlink(self.db_path)
        except Exception:
            pass


# ── Seed inicial para Riemann ──────────────────────────────────────────────────

RIEMANN_KNOWN: list[dict] = [
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["zeros", "critical_line", "verification"],
        "statement":   "Los primeros 10^13 ceros no triviales de ζ(s) están en Re(s)=0.5 (Gourdon 2004; Platt & Trudgian 2021)",
        "implication": "Cualquier contraejemplo tiene Im(s) > 10^13 — buscar allí, no re-verificar los primeros ceros",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["zero_free_region", "sigma", "classical"],
        "statement":   "Región libre de ceros: σ > 1 - c/log(t) para c>0 (de la Vallée Poussin 1899; Vinogradov-Korobov 1958)",
        "implication": "Los ceros están lejos de σ=1 — la dificultad está en el interior de la franja crítica",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["gue", "random_matrix", "pair_correlation", "montgomery"],
        "statement":   "Correlación par de ceros ~ GUE (Montgomery 1973; evidencia masiva de Odlyzko 1987-2001)",
        "implication": "La estadística de espaciado es indistinguible de matrices unitarias aleatorias — explorar por qué",
        "confidence":  0.99,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["explicit_formula", "prime_counting", "von_mangoldt"],
        "statement":   "ψ(x) = x - Σ_{ρ} x^ρ/ρ - log(2π) - (1/2)log(1-x^{-2}) (von Mangoldt 1895)",
        "implication": "RH ↔ error en π(x) = O(√x log²x). No re-verificar la fórmula; sí explorar el error residual",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["functional_equation", "symmetry", "xi"],
        "statement":   "ζ(s) = 2^s π^{s-1} sin(πs/2) Γ(1-s) ζ(1-s) — simetría alrededor de Re=1/2 (Riemann 1859)",
        "implication": "Los ceros no triviales vienen en pares (ρ, 1-ρ̄). La línea crítica es el eje de simetría",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["moments", "keating_snaith", "random_matrix"],
        "statement":   "E[|ζ(1/2+it)|^{2k}] ~ C_k (log T)^{k²} — conjetura Keating-Snaith (2000), apoyo numérico fuerte",
        "implication": "Frontera activa: los momentos de ζ en la línea crítica tienen estructura GUE — explorar k>2",
        "confidence":  0.95,
        "source":      "literature",
        "skip_verify": False,
    },
    {
        "domain":      "riemann",
        "kind":        "known",
        "tags":        ["hilbert_polya", "operator", "spectrum"],
        "statement":   "Programa Hilbert-Pólya: existe un operador hermítico cuyo espectro son Im(ρ) — no probado",
        "implication": "Si el operador existe, RH es consecuencia de ser hermítico. Explorar operadores de Dirac en espacios no commutative",
        "confidence":  0.5,
        "source":      "conjecture",
        "skip_verify": False,
    },
]

PNP_KNOWN: list[dict] = [
    {
        "domain":      "pnp",
        "kind":        "known",
        "tags":        ["np_complete", "sat", "cook_levin"],
        "statement":   "SAT es NP-completo (Cook 1971; Levin 1973). Resolver SAT en tiempo polinomial ↔ P=NP",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "pnp",
        "kind":        "known",
        "tags":        ["barrier", "natural_proofs", "razborov_rudich"],
        "statement":   "Barrera de pruebas naturales: cualquier prueba 'natural' de P≠NP implicaría inexistencia de PRGs (Razborov-Rudich 1994)",
        "implication": "Las técnicas combinatorias estándar no pueden separar P de NP",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "pnp",
        "kind":        "known",
        "tags":        ["barrier", "algebrization", "aaronson_wigderson"],
        "statement":   "Barrera de algebrización: relativizaciones algebrizadas no separan P de NP (Aaronson-Wigderson 2009)",
        "implication": "Necesitamos técnicas no-relativizantes y no-naturales",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
    {
        "domain":      "pnp",
        "kind":        "known",
        "tags":        ["circuit", "ac0", "hastad", "parity"],
        "statement":   "PARITY ∉ AC⁰ (Håstad 1986). Profundidad d necesita exp(n^{1/(d-1)}) gates",
        "confidence":  1.0,
        "source":      "literature",
        "skip_verify": True,
    },
]


def seed_known_facts(store: KnowledgeStore, domain: str) -> int:
    """Puebla el store con los hechos conocidos de `domain`. Idempotente."""
    facts_map = {"riemann": RIEMANN_KNOWN, "pnp": PNP_KNOWN}
    facts = facts_map.get(domain, [])
    count = 0
    for f in facts:
        existing = store.conn.execute(
            "SELECT id FROM knowledge WHERE domain=? AND statement=?",
            (f["domain"], f["statement"])
        ).fetchone()
        if not existing:
            store.store(**f)
            count += 1
    return count
