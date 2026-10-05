"""
core/meta_monitor.py — MetaMonitorAgent v0.1
═════════════════════════════════════════════
Agente LLM de alto nivel que observa TODOS los swarms en paralelo.

- Corre en background thread dentro del coordinador
- Cada `interval_min` minutos, lee todos los epistemic_states disponibles
- Llama al oracle para sintetizar conexiones cross-problema y direcciones futuras
- Escribe results/meta_guidance.json → workers lo leen en SURVEY
- Imprime resumen visible en terminal

Arquitectura:
  run_v020.py
    └─ MetaMonitorAgent.start() → background thread
         └─ synthesize() cada N minutos
              ├─ lee epistemic_state_*.json
              ├─ llama oracle (Sonnet por defecto)
              └─ escribe meta_guidance.json
"""

import json
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-5-5"
FABLE  = "claude-fable-5-1"

META_GUIDANCE_FILE = "meta_guidance.json"


class MetaMonitorAgent:
    """
    Observador LLM que sintetiza el estado de múltiples swarms y genera
    guidance estratégica que los workers absorben en su siguiente ciclo.
    """

    def __init__(self,
                 problems: list[str],
                 api_key:  str | None,
                 model:    str = SONNET,
                 budget_usd: float = 1.0,
                 interval_min: float = 20.0,
                 results_dir: str = "results"):
        self.problems      = problems
        self.api_key       = api_key
        self.model         = model
        self.budget_usd    = budget_usd
        self.interval_sec  = interval_min * 60
        self.results_dir   = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self._spent        = 0.0
        self._call_count   = 0
        self._stop_event   = threading.Event()
        self._client       = None
        self._available    = False

        if api_key:
            try:
                from core.anthropic_http import Anthropic
                self._client    = Anthropic(api_key=api_key)
                self._available = True
                print(f"  [MetaMonitor] disponible ✓  modelo={model}  "
                      f"intervalo={interval_min:.0f}min  budget=${budget_usd:.2f}")
            except Exception as e:
                print(f"  [MetaMonitor] error al inicializar: {e}")
        else:
            print("  [MetaMonitor] sin API key — deshabilitado")

    # ── Control ───────────────────────────────────────────────────────────────

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self._run_loop, daemon=True,
                             name="MetaMonitor")
        t.start()
        return t

    def stop(self):
        self._stop_event.set()

    # ── Loop principal ────────────────────────────────────────────────────────

    def _run_loop(self):
        # Primera síntesis tras el primer intervalo
        while not self._stop_event.wait(timeout=self.interval_sec):
            if self._spent >= self.budget_usd:
                print(f"  [MetaMonitor] budget agotado (${self._spent:.3f}) — deteniendo")
                break
            self.synthesize()

    # ── Síntesis ──────────────────────────────────────────────────────────────

    def synthesize(self) -> Optional[dict]:
        """
        Lee estados epistémicos de todos los problemas activos,
        llama al oracle para síntesis cross-problema, y escribe guidance.
        """
        if not self._available:
            return None

        snapshots = self._read_all_states()
        if not snapshots:
            print("  [MetaMonitor] sin estados disponibles aún")
            return None

        prompt = self._build_prompt(snapshots)
        result = self._call_oracle(prompt)
        if not result:
            return None

        guidance = {
            "updated_at": datetime.now().isoformat(),
            "call_n":     self._call_count,
            "model":      self.model,
            "cost_usd":   round(self._spent, 4),
            "snapshots":  {p: s.get("n_supported", 0) for p, s in snapshots.items()},
            "insights":   result.get("insights", []),
            "directions": result.get("directions", {}),
            "cross_connection": result.get("cross_connection", ""),
        }

        out = self.results_dir / META_GUIDANCE_FILE
        with open(out, "w", encoding="utf-8") as f:
            json.dump(guidance, f, indent=2, ensure_ascii=False)

        print(f"\n  {'═'*60}")
        print(f"  [MetaMonitor #{self._call_count}]  ${self._spent:.3f}/{self.budget_usd:.2f}")
        for p, s in snapshots.items():
            print(f"    {p:12s} → {s.get('n_supported',0)} robustos  "
                  f"{s.get('n_candidates',0)} candidatos")
        if guidance["cross_connection"]:
            print(f"  Conexión cross: {guidance['cross_connection'][:120]}")
        for p, d in guidance["directions"].items():
            print(f"  Dir [{p}]: {str(d)[:100]}")
        print(f"  {'═'*60}\n")

        return guidance

    # ── Lectura de estados ────────────────────────────────────────────────────

    def _read_all_states(self) -> dict:
        """Lee el estado epistémico más reciente de cada problema."""
        snapshots = {}
        for problem in self.problems:
            # Preferir archivo sin agent_id (estado de sesión principal)
            candidates = sorted(
                self.results_dir.glob(f"epistemic_state_{problem}*.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if not candidates:
                continue
            try:
                with open(candidates[0], encoding="utf-8") as f:
                    state = json.load(f)
                # Extraer resumen compacto
                rob = state.get("robust_supported", {})
                cand = state.get("candidate_supported", {})
                hyps = state.get("hypotheses", [])
                snapshots[problem] = {
                    "n_supported":    len(rob),
                    "n_candidates":   len(cand),
                    "robust_keys":    list(rob.keys())[:5],
                    "top_hyps":       [h.get("statement","")[:80] for h in hyps[:3]],
                    "richness":       state.get("richness", 0),
                    "cycles":         state.get("total_cycles", 0),
                }
            except Exception as e:
                print(f"  [MetaMonitor] error leyendo {candidates[0].name}: {e}")
        return snapshots

    # ── Prompt ────────────────────────────────────────────────────────────────

    def _build_prompt(self, snapshots: dict) -> str:
        sections = []
        for problem, s in snapshots.items():
            rob_str  = ", ".join(s["robust_keys"]) or "ninguno aún"
            hyp_str  = "\n".join(f"  - {h}" for h in s["top_hyps"]) or "  (sin hipótesis)"
            sections.append(
                f"## {problem.upper()} — {s['cycles']} ciclos | richness={s['richness']:.3f}\n"
                f"Hallazgos robustos ({s['n_supported']}): {rob_str}\n"
                f"Hipótesis activas:\n{hyp_str}"
            )
        state_text = "\n\n".join(sections)

        return f"""Eres un meta-investigador matemático que supervisa múltiples swarms de agentes.
Cada swarm explora un problema matemático abierto de forma autónoma.

ESTADO ACTUAL DE LOS SWARMS:
{state_text}

Tu tarea:
1. Identificar UNA conexión no obvia entre dos o más problemas (si existe).
2. Para cada problema activo, sugerir UNA dirección concreta de exploración
   que los agentes probablemente no hayan considerado.
3. Si un problema tiene 0 hallazgos robustos, sugerir por qué y qué probar.

Restricciones:
- Direcciones ejecutables en Python (numpy, sympy, scipy disponibles)
- Máximo 2 oraciones por dirección
- No repetir lo que ya está en los hallazgos robustos

Responde SOLO en JSON:
{{
  "cross_connection": "conexión entre problemas en 1-2 oraciones, o 'none'",
  "insights": ["observación 1", "observación 2"],
  "directions": {{
    "{list(snapshots.keys())[0]}": "dirección concreta para este problema"
    {(',"' + list(snapshots.keys())[1] + '": "dirección"') if len(snapshots) > 1 else ''}
  }}
}}"""

    # ── Llamada al oracle ─────────────────────────────────────────────────────

    def _call_oracle(self, prompt: str) -> Optional[dict]:
        if not self._client or self._spent >= self.budget_usd:
            return None
        try:
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            text_blocks = [b for b in resp.content if hasattr(b, "text")]
            if not text_blocks:
                return None
            text    = text_blocks[0].text.strip()
            in_tok  = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens

            from core.oracle_client import MODEL_PRICES
            p_in, p_out = MODEL_PRICES.get(self.model, (3.0, 15.0))
            cost = (in_tok / 1_000_000) * p_in + (out_tok / 1_000_000) * p_out
            self._spent     += cost
            self._call_count += 1

            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return json.loads(m.group())
            return {"insights": [text[:200]], "directions": {}, "cross_connection": ""}
        except Exception as e:
            print(f"  [MetaMonitor] error oracle: {e}")
            return None

    # ── Utilidad para workers ─────────────────────────────────────────────────

    @classmethod
    def read_guidance(cls, problem: str, results_dir: str = "results") -> str:
        """
        Llamado desde workers en SURVEY para leer guidance del MetaMonitor.
        Retorna texto para añadir como contexto al oracle.
        """
        path = Path(results_dir) / META_GUIDANCE_FILE
        if not path.exists():
            return ""
        try:
            with open(path, encoding="utf-8") as f:
                g = json.load(f)
            parts = []
            if g.get("cross_connection") and g["cross_connection"] != "none":
                parts.append(f"Conexión cross-problema: {g['cross_connection']}")
            direction = g.get("directions", {}).get(problem, "")
            if direction:
                parts.append(f"Meta-dirección sugerida: {direction}")
            for ins in g.get("insights", [])[:2]:
                parts.append(f"Insight: {ins}")
            return "\nMETA-MONITOR GUIDANCE:\n" + "\n".join(parts) if parts else ""
        except Exception:
            return ""

    def summary(self) -> dict:
        return {
            "calls":      self._call_count,
            "spent_usd":  round(self._spent, 4),
            "budget_usd": self.budget_usd,
            "model":      self.model,
        }
