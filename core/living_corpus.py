"""
core/living_corpus.py — Fachada del corpus vivo

Wrapper ligero sobre library/knowledge_store.KnowledgeStore.
Mantiene la misma API que usaban los workers existentes,
pero persiste en SQLite (no JSON) y es compatible con SatelliteKnowledgeStore.

Dos modos:
  main     — store principal compartido entre todos los workers
  satellite — store temporal por worker; se fusiona al terminar
"""

from library.knowledge_store import (
    KnowledgeStore,
    SatelliteKnowledgeStore,
    seed_known_facts,
    RIEMANN_KNOWN,
    PNP_KNOWN,
)

__all__ = [
    "LivingCorpus",
    "SatelliteCorpus",
    "KnowledgeStore",
    "SatelliteKnowledgeStore",
    "seed_known_facts",
]


class LivingCorpus:
    """
    Corpus vivo principal para un dominio.
    Thread-safe (SQLite WAL mode).
    """

    def __init__(self, domain: str, db_dir: str = "library"):
        self.domain = domain
        self._store = KnowledgeStore(f"{db_dir}/knowledge_{domain}.sqlite")
        # Poblar hechos conocidos si la DB está vacía
        seeded = seed_known_facts(self._store, domain)
        if seeded:
            print(f"  [LivingCorpus/{domain}] {seeded} hechos conocidos cargados ✓")

    # ── Consulta ──────────────────────────────────────────────────────────────

    def should_skip(self, topic_keywords: list[str]) -> dict | None:
        return self._store.should_skip(self.domain, topic_keywords)

    def already_ran(self, code: str) -> dict | None:
        return self._store.already_ran(self.domain, code)

    def get_frontier(self) -> list[dict]:
        """Entradas de alta confianza — dónde explorar."""
        return self._store.K4_frontier(self.domain, top_n=5, min_confidence=0.7)

    def get_discoveries(self, min_confidence: float = 0.7) -> list[dict]:
        results = self._store.K4_frontier(
            self.domain, top_n=20, min_confidence=min_confidence
        )
        return [r for r in results if r.get("kind") == "discovery"]

    def resonance(self, topic_tags: list[str], top_n: int = 5) -> list[dict]:
        """K1: ¿qué sabemos sobre este tema?"""
        return self._store.K1_resonance(self.domain, topic_tags, top_n=top_n)

    def novelty(self, explored_tags: list[str], top_n: int = 5) -> list[dict]:
        """K2: ¿qué temas son menos explorados?"""
        return self._store.K2_novelty(self.domain, explored_tags, top_n=top_n)

    # ── Escritura ─────────────────────────────────────────────────────────────

    def add_discovery(self, topic: str, code: str, conclusion: str,
                      confidence: float, tags: list[str] | None = None,
                      implication: str = "", agent_id: str = "") -> int:
        return self._store.store(
            domain      = self.domain,
            kind        = "discovery",
            tags        = tags or topic.split("_"),
            statement   = conclusion,
            confidence  = confidence,
            implication = implication,
            code        = code,
            source      = f"agent:{agent_id}" if agent_id else "agent",
        )

    def add_failure(self, code: str, reason: str,
                    tags: list[str] | None = None, agent_id: str = "") -> int:
        return self._store.store(
            domain  = self.domain,
            kind    = "failed",
            tags    = tags or ["unknown"],
            statement = reason[:200],
            source  = f"agent:{agent_id}" if agent_id else "agent",
        )

    # ── Context para oracle ───────────────────────────────────────────────────

    def to_oracle_context(self, topic_tags: list[str] | None = None,
                          max_chars: int = 1400) -> str:
        return self._store.to_oracle_context(self.domain, topic_tags, max_chars)

    def summary(self) -> dict:
        return self._store.summary(self.domain)


class SatelliteCorpus(LivingCorpus):
    """
    Corpus temporal por worker. Al terminar, el coordinador llama merge().
    Solo las entradas con confidence > percentil 60 pasan al store principal.
    """

    def __init__(self, agent_id: str, domain: str, db_dir: str = "library"):
        self.domain  = domain
        self._store  = SatelliteKnowledgeStore(agent_id, domain, db_dir)
        self._sat    = self._store  # referencia directa para merge
        # Los satélites NO se pueblan con known facts (los lee el coordinador)

    def merge(self, main_corpus: LivingCorpus,
              percentile_threshold: float = 60.0) -> int:
        """
        Fusiona hallazgos al corpus principal.
        Devuelve el número de entradas transferidas.
        """
        n = self._sat.merge_into(main_corpus._store, percentile_threshold)
        print(f"  [SatelliteCorpus/{self._sat.agent_id}] {n} entradas fusionadas ✓")
        return n

    def cleanup(self) -> None:
        self._sat.cleanup()
