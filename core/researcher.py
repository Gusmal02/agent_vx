"""
core/researcher.py — ResearcherAgent v0.1
═══════════════════════════════════════════
Busca papers en arxiv, extrae hipótesis testables con numpy,
y construye un corpus dinámico que enriquece el swarm.

Flujo:
  1. Buscar papers relevantes en arxiv API (sin API key)
  2. Por cada abstract → oracle extrae hipótesis cuantitativas + código numpy
  3. Guardar corpus_{problem}.json en results/
  4. Entre rondas: Monitor retroalimenta qué hipótesis buscar más

Corpus usado por:
  - Workers: como acciones adicionales en SURVEY
  - Oracle: como contexto adicional en VERIFY
"""

import json
import re
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path


ARXIV_API = "http://export.arxiv.org/api/query"

# Queries específicas por problema
PROBLEM_QUERIES = {
    "causal":    "causal inference backdoor adjustment invariant risk minimization",
    "continual": "continual learning catastrophic forgetting elastic weight consolidation",
    "riemann":   "Riemann hypothesis zeta function zero distribution critical line",
    "pnp":       "P versus NP complexity SAT phase transition circuit lower bounds",
}

# Fallback si no hay oracle
SYNTHETIC_HYPOTHESES = {
    "causal": [
        "En presencia de confounders, el ajuste por backdoor reduce el sesgo estimado en >50%",
        "IRM converge a mejores generalizaciones que ERM cuando hay distributional shift",
        "El threshold óptimo de descubrimiento de estructura causal depende del tamaño muestral",
    ],
    "continual": [
        "EWC reduce la tasa de olvido catastrófico en >30% comparado con fine-tuning vanilla",
        "La importancia de los parámetros correlaciona inversamente con la plasticidad de la red",
    ],
}


class ResearcherAgent:
    """
    Agente que construye corpus desde literatura académica.
    Se ejecuta antes de la primera ronda y puede actualizarse entre rondas.
    """

    def __init__(self, problem: str, api_key: str | None = None,
                 results_dir: str = "results"):
        self.problem     = problem
        self.api_key     = api_key
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self._oracle_calls = 0
        self._oracle_budget = 6  # máximo llamadas para researcher

    # ── Flujo principal ───────────────────────────────────────────────────────

    def build_corpus(self, n_papers: int = 5,
                     max_hyps_per_paper: int = 2,
                     existing_findings: dict | None = None) -> dict:
        """
        Construye o actualiza el corpus.
        existing_findings: hallazgos ya robustos → orienta la búsqueda.
        """
        print(f"\n  [Researcher] ── Construyendo corpus: {self.problem} ──")

        # Si hay hallazgos previos, buscar papers que los contradigan o amplíen
        query = self._build_query(existing_findings)
        print(f"  [Researcher] query: {query}")

        papers = self._search_arxiv(query, n_papers)
        print(f"  [Researcher] {len(papers)} papers encontrados en arxiv")

        experiments = []
        for i, paper in enumerate(papers):
            print(f"  [Researcher] procesando [{i+1}/{len(papers)}]: {paper['title'][:60]}...")
            hyps = self._extract_hypotheses(paper, max_hyps_per_paper)
            experiments.extend(hyps)
            time.sleep(0.3)  # amable con arxiv

        # Cargar corpus existente y añadir sin duplicar
        existing_corpus = self._load_existing()
        merged = self._merge_corpus(existing_corpus, experiments)

        corpus = {
            "problem":       self.problem,
            "updated_at":    datetime.now().isoformat(),
            "n_papers":      len(papers),
            "n_experiments": len(merged),
            "experiments":   merged,
        }
        path = self._save_corpus(corpus)
        print(f"  [Researcher] corpus listo → {path}  "
              f"({len(merged)} hipótesis, {len(papers)} papers)")
        return corpus

    def _build_query(self, existing_findings: dict | None) -> str:
        """Orienta la búsqueda según hallazgos previos o usa query base."""
        base = PROBLEM_QUERIES.get(self.problem, self.problem)
        if not existing_findings:
            return base
        # Añadir términos de los hallazgos robustos para buscar papers contrarios
        terms = list(existing_findings.keys())[:3]
        extra = " ".join(t.replace("_", " ") for t in terms)
        return f"{base} {extra}"

    # ── Arxiv ─────────────────────────────────────────────────────────────────

    def _search_arxiv(self, query: str, max_results: int) -> list[dict]:
        # Usar all: (todos los campos) para mayor cobertura que ti: (solo título)
        # urllib.parse.urlencode convierte espacios a +, que arxiv entiende como AND
        params = urllib.parse.urlencode({
            "search_query": f"all:{query}",
            "start":        0,
            "max_results":  max_results,
            "sortBy":       "relevance",
            "sortOrder":    "descending",
        })
        url = f"{ARXIV_API}?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "agentevx/0.4"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                xml = resp.read().decode("utf-8")
            return self._parse_arxiv(xml)
        except Exception as e:
            print(f"  [Researcher] arxiv error: {e}")
            return []

    def _parse_arxiv(self, xml: str) -> list[dict]:
        papers = []
        for entry in re.split(r"<entry>", xml)[1:]:
            title   = re.search(r"<title[^>]*>(.*?)</title>", entry, re.DOTALL)
            summary = re.search(r"<summary[^>]*>(.*?)</summary>", entry, re.DOTALL)
            link    = re.search(r'href="(https://arxiv\.org/abs/[^"]+)"', entry)
            arxiv_id = re.search(r'<id>(https://arxiv\.org/abs/([^<]+))</id>', entry)
            if title and summary:
                papers.append({
                    "title":    title.group(1).strip().replace("\n", " "),
                    "abstract": summary.group(1).strip()[:900],
                    "url":      link.group(1) if link else "",
                    "id":       arxiv_id.group(2).strip() if arxiv_id else "",
                })
        return papers

    # ── Extracción de hipótesis ───────────────────────────────────────────────

    def _extract_hypotheses(self, paper: dict,
                             max_hyps: int) -> list[dict]:
        if self.api_key and self._oracle_calls < self._oracle_budget:
            result = self._extract_with_oracle(paper, max_hyps)
            if result:
                return result
        return self._extract_simple(paper, max_hyps)

    def _extract_with_oracle(self, paper: dict, max_hyps: int) -> list[dict]:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)
            prompt = f"""Paper: "{paper['title']}"
Abstract: {paper['abstract'][:600]}

Extrae {max_hyps} hipótesis CUANTITATIVAS testables.
Requisitos del entorno de ejecución:
- Solo numpy disponible (import numpy as np)
- Código < 15 líneas
- Debe terminar con: _result = {{"score": float, "confirms": bool, "metric": str}}
- La hipótesis DEBE poder ser falsa (falsificable)
- Usa datos sintéticos generados con numpy (no archivos externos)

Responde SOLO en JSON:
{{"hypotheses": [
  {{
    "statement": "hipótesis cuantitativa en una oración",
    "code": "import numpy as np\\nnp.random.seed(42)\\n# experimento\\n_result = {{\\"score\\": 0.0, \\"confirms\\": True, \\"metric\\": \\"nombre\\"}}",
    "expected_if_true": "condición de confirmación observable"
  }}
]}}"""

            resp = client.messages.create(
                model="claude-sonnet-4-6", max_tokens=1200,
                messages=[{"role": "user", "content": prompt}]
            )
            self._oracle_calls += 1
            text = resp.content[0].text
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if not m:
                return []
            data = json.loads(m.group())
            hyps = data.get("hypotheses", [])
            pid  = paper.get("id", "unknown").replace("/", "_")
            return [
                {
                    "id":               f"{pid}_h{i}",
                    "source_url":       paper.get("url", ""),
                    "source_title":     paper["title"],
                    "hypothesis":       h.get("statement", ""),
                    "code":             h.get("code", ""),
                    "expected_if_true": h.get("expected_if_true", ""),
                    "origin":           "arxiv+oracle",
                }
                for i, h in enumerate(hyps[:max_hyps])
                if h.get("code")  # solo si tiene código ejecutable
            ]
        except Exception as e:
            print(f"  [Researcher/oracle] error: {e}")
            return []

    def _extract_simple(self, paper: dict, max_hyps: int) -> list[dict]:
        """Fallback textual (sin código ejecutable) — útil como contexto para oracle."""
        pid = paper.get("id", "unknown").replace("/", "_")
        return [{
            "id":               f"{pid}_text",
            "source_url":       paper.get("url", ""),
            "source_title":     paper["title"],
            "hypothesis":       f"[{self.problem}] Hipótesis de: {paper['title'][:70]}",
            "code":             "",
            "expected_if_true": "ver abstract",
            "origin":           "arxiv_text_only",
        }]

    # ── Corpus ────────────────────────────────────────────────────────────────

    def _load_existing(self) -> list[dict]:
        path = self.results_dir / f"corpus_{self.problem}.json"
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    c = json.load(f)
                return c.get("experiments", [])
            except Exception:
                pass
        return []

    def _merge_corpus(self, existing: list[dict],
                      new_items: list[dict]) -> list[dict]:
        existing_ids = {e["id"] for e in existing}
        added = [x for x in new_items if x["id"] not in existing_ids]
        return existing + added

    def _save_corpus(self, corpus: dict) -> Path:
        out = self.results_dir / f"corpus_{self.problem}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(corpus, f, indent=2, ensure_ascii=False)
        return out

    # ── Utilidad para Monitor ─────────────────────────────────────────────────

    def summarize_for_oracle(self, max_items: int = 5) -> str:
        """
        Retorna un resumen del corpus para añadir como contexto al oracle.
        Usado en VERIFY para enriquecer los experimentos propuestos.
        """
        existing = self._load_existing()
        if not existing:
            return ""
        items = [e for e in existing if e.get("hypothesis")][:max_items]
        lines = [f"- {e['hypothesis']}" for e in items]
        return "HIPÓTESIS DEL CORPUS (literatura académica):\n" + "\n".join(lines)
