"""
core/research_memory.py — Memoria dinámica de investigación (v0.1.0)

Corpus dinámico: los subagentes pueden indexar nuevos resultados.
La próxima iteración lee lo que indexaron.

Basado en el insight de hipotesis_mente_creativa:
  interest(c) = novelty(c) × max_resonance(c)
  El campo es un recombinador puro — necesita input externo para mantenerse creativo.
  Esta memoria ES el input externo persistente entre ciclos.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class MemoryEntry:
    entry_id:   str
    domain:     str
    source:     str          # "main", "subagent", "oracle", "invention"
    tags:       list[str]
    summary:    str          # descripción en lenguaje natural
    score:      float        # qué tan útil fue (0–1)
    cycle:      int
    timestamp:  float = field(default_factory=time.time)
    result:     Optional[dict] = None


class ResearchMemory:
    """
    Memoria dinámica de investigación.

    Los subagentes escriben con index().
    El agente principal lee con query().
    El corpus dinámico se genera con to_corpus_knowledge().
    """

    def __init__(self, domain: str, persist_path: Optional[str] = None):
        self._domain    = domain
        self._entries:  list[MemoryEntry] = []
        self._counter   = 0
        self._path      = Path(persist_path) if persist_path else None

        if self._path and self._path.exists():
            self._load()

    # ── Escritura ─────────────────────────────────────────────────────────────

    def index(self,
              summary:  str,
              score:    float,
              cycle:    int,
              source:   str = "main",
              tags:     Optional[list[str]] = None,
              result:   Optional[dict] = None,
              ) -> str:
        """
        Indexa un resultado en la memoria.
        Devuelve el entry_id asignado.
        """
        self._counter += 1
        entry = MemoryEntry(
            entry_id  = f"mem_{self._counter:04d}",
            domain    = self._domain,
            source    = source,
            tags      = tags or [],
            summary   = summary[:500],
            score     = round(max(0.0, min(1.0, score)), 4),
            cycle     = cycle,
            result    = result,
        )
        self._entries.append(entry)

        if self._path:
            self._save()

        return entry.entry_id

    # ── Lectura ───────────────────────────────────────────────────────────────

    def query(self,
              tags:      Optional[list[str]] = None,
              min_score: float = 0.0,
              source:    Optional[str] = None,
              top_k:     int = 10,
              ) -> list[MemoryEntry]:
        """
        Devuelve entradas que coincidan con los filtros, ordenadas por score desc.
        """
        results = self._entries

        if tags:
            results = [e for e in results
                       if any(t in e.tags for t in tags)]
        if min_score > 0:
            results = [e for e in results if e.score >= min_score]
        if source:
            results = [e for e in results if e.source == source]

        results = sorted(results, key=lambda e: e.score, reverse=True)
        return results[:top_k]

    def recent(self, n: int = 5) -> list[MemoryEntry]:
        """Últimas n entradas por timestamp."""
        return sorted(self._entries, key=lambda e: e.timestamp, reverse=True)[:n]

    # ── Corpus dinámico ───────────────────────────────────────────────────────

    def to_corpus_knowledge(self, top_k: int = 8) -> str:
        """
        Convierte las mejores entradas en conocimiento para el corpus.
        Se inyecta en el prompt del oráculo y en el contexto del tejido.
        """
        best = self.query(min_score=0.5, top_k=top_k)
        if not best:
            return "(sin resultados indexados aún)"

        lines = []
        for e in best:
            source_label = {"subagent": "Subagente", "oracle": "Oráculo",
                            "invention": "Invención", "main": "Agente"}.get(e.source, e.source)
            lines.append(
                f"  [{source_label} ciclo {e.cycle}, score={e.score:.2f}] {e.summary}"
            )
        return "\n".join(lines)

    def tension_pairs(self, min_score: float = 0.45) -> list[tuple[MemoryEntry, MemoryEntry]]:
        """
        Detecta pares de entradas en tensión:
        mismo dominio, scores similares, tags contradictorios.
        Útil para disparar creative_crisis().
        """
        candidates = self.query(min_score=min_score, top_k=20)
        pairs = []

        TENSION_PAIRS = [
            ({"espectral", "gue", "wigner"}, {"algebraic", "gct", "algebraico"}),
            ({"espectral", "wigner"},         {"zeros", "critica", "linea"}),
            ({"algebraic", "perm"},           {"sat", "np-hard"}),
        ]

        for e1 in candidates:
            for e2 in candidates:
                if e1.entry_id >= e2.entry_id:
                    continue
                tags1 = set(e1.tags)
                tags2 = set(e2.tags)
                for side_a, side_b in TENSION_PAIRS:
                    if (tags1 & side_a) and (tags2 & side_b):
                        pairs.append((e1, e2))
                        break

        return pairs

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def summary(self) -> dict:
        n = len(self._entries)
        by_source = {}
        for e in self._entries:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        avg = round(sum(e.score for e in self._entries) / max(1, n), 3)
        return {
            "total":     n,
            "by_source": by_source,
            "avg_score": avg,
            "top_tags":  self._top_tags(5),
        }

    def _top_tags(self, k: int) -> list[str]:
        counts: dict[str, int] = {}
        for e in self._entries:
            for t in e.tags:
                counts[t] = counts.get(t, 0) + 1
        return sorted(counts, key=lambda t: counts[t], reverse=True)[:k]

    # ── Persistencia ─────────────────────────────────────────────────────────

    def _save(self):
        try:
            data = [asdict(e) for e in self._entries]
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception:
            pass

    def _load(self):
        try:
            data = json.loads(self._path.read_text())
            for d in data:
                e = MemoryEntry(**d)
                self._entries.append(e)
                self._counter = max(self._counter, int(e.entry_id.split("_")[1]))
        except Exception:
            pass
