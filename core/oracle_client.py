"""
core/oracle_client.py — Oráculo Claude API (v0.0.9)

El oráculo NO toma decisiones por el agente.
Solo responde preguntas específicas y devuelve:
  - Código Python ejecutable en el sandbox
  - Hipótesis estructuradas para que el agente evalúe
  - Conexiones cross-domain cuando se le piden

Modelos:
  - Sonnet 4.6 (claude-sonnet-4-6): llamadas regulares de estrategia/código
  - Opus 5.5  (claude-opus-5-5):    solo hipótesis críticas (confianza > 0.85)

Presupuesto: máximo 4 llamadas/hora, $0.011 promedio por llamada.
"""

import time
import json
import re
from typing import Optional


SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-5-5"
FABLE  = "claude-fable-5-1"

# Precios por MTok (in, out) en USD
MODEL_PRICES = {
    SONNET: (3.0,  15.0),
    OPUS:   (15.0, 75.0),
    FABLE:  (3.0,  15.0),   # aproximado hasta confirmar pricing
}

# Límites de llamadas — se ajustan según modelo primario
_LIMITS_BY_MODEL = {
    SONNET: {"per_hour": 6,  "opus_cap": 2},
    OPUS:   {"per_hour": 10, "opus_cap": 999},  # sin cap cuando es el modelo principal
    FABLE:  {"per_hour": 8,  "opus_cap": 2},
}
CALL_WINDOW = 3600


class OracleClient:
    """
    Wrapper del API de Anthropic para consultas estratégicas.
    Se invoca solo cuando el agente está estancado o tiene hipótesis críticas.
    primary_model: modelo base para llamadas regulares (sonnet/opus/fable).
    """

    def __init__(self, api_key: str, budget_usd: float = 2.50,
                 primary_model: str = SONNET):
        self._api_key       = api_key
        self._budget        = budget_usd
        self._spent         = 0.0
        self._call_log: list = []
        self._client        = None
        self._available     = False
        self._primary_model = primary_model
        _lim = _LIMITS_BY_MODEL.get(primary_model, _LIMITS_BY_MODEL[SONNET])
        self._max_per_hour  = _lim["per_hour"]
        self._opus_cap      = _lim["opus_cap"]

        self._domain_knowledge: str = ""

        if api_key and api_key != "PLACEHOLDER":
            try:
                from core.anthropic_http import Anthropic
                self._client   = Anthropic(api_key=api_key)
                self._available = True
                print(f"  [Oracle] disponible ✓  (presupuesto: ${budget_usd:.2f})")
            except Exception as e:
                print(f"  [Oracle] error al inicializar: {e}")
        else:
            print("  [Oracle] sin API key — deshabilitado")

    def set_domain_knowledge(self, knowledge: str) -> None:
        """Inyecta conocimiento de dominio que se prepende a cada prompt del oracle."""
        self._domain_knowledge = knowledge.strip()

    # ── Rate limiting ─────────────────────────────────────────────────────────

    def _can_call(self, model: str | None = None) -> bool:
        if model is None:
            model = self._primary_model
        if not self._available:
            return False
        if self._spent >= self._budget:
            print(f"  [Oracle] presupuesto agotado (${self._spent:.3f})")
            return False
        now = time.time()
        recent = [c for c in self._call_log if now - c["ts"] < CALL_WINDOW]
        if len(recent) >= self._max_per_hour:
            return False
        if model == OPUS:
            opus_total = sum(1 for c in self._call_log if c["model"] == OPUS)
            if opus_total >= self._opus_cap:
                print(f"  [Oracle] límite Opus ({opus_total}/{self._opus_cap})")
                return False
        return True

    def _record_call(self, model: str, in_tok: int, out_tok: int) -> float:
        p_in, p_out = MODEL_PRICES.get(model, (3.0, 15.0))
        cost = (in_tok / 1_000_000) * p_in + (out_tok / 1_000_000) * p_out
        self._spent += cost
        self._call_log.append({"ts": time.time(), "model": model, "cost": cost})
        return cost

    # ── Llamadas al oráculo ───────────────────────────────────────────────────

    def suggest_direction(self,
                          problem: str,
                          hypotheses: list,
                          recent_scores: dict,
                          failed_actions: list,
                          ) -> Optional[dict]:
        """
        Pregunta al oráculo qué dirección explorar cuando hay estancamiento.
        Devuelve: {"direction": str, "code": str, "reasoning": str}
        """
        if not self._can_call():
            return None

        hyp_text = "\n".join(
            f"- [{h['confidence']:.2f}] {h['statement']}"
            for h in hypotheses[-5:]
        ) or "Ninguna hipótesis registrada aún."

        scores_text = "\n".join(
            f"  {action}: avg={stats['avg']:.3f}, n={stats['n']}"
            for action, stats in recent_scores.items()
        )

        failed_text = ", ".join(failed_actions[-5:]) or "ninguno"

        prompt = f"""Eres un asistente matemático. El agente está explorando: {problem.upper()}.

HIPÓTESIS ACTUALES:
{hyp_text}

RENDIMIENTO POR ACCIÓN (promedio score):
{scores_text}

ACCIONES QUE HAN FALLADO RECIENTEMENTE: {failed_text}

El agente está estancado. Sugiere UNA dirección concreta de exploración que:
1. No repita lo que ya falló
2. Conecte con las hipótesis actuales
3. Sea ejecutable en Python con numpy/sympy/mpmath

Responde SOLO en JSON con esta estructura exacta:
{{
  "direction": "descripción en una oración de qué explorar",
  "code": "código Python ejecutable (usa np, sp, mpmath, _result = ...)",
  "reasoning": "por qué esta dirección es prometedora"
}}"""

        return self._call(self._primary_model, prompt, label="suggest_direction")

    def verify_hypothesis(self,
                          hypothesis: str,
                          domain: str,
                          supporting_evidence: list,
                          ) -> Optional[dict]:
        """
        Evaluación crítica de una hipótesis de alta confianza.
        Solo se llama cuando confidence > 0.85.
        Usa Opus 5.5 por su capacidad de razonamiento profundo.
        Devuelve: {"valid": bool, "confidence": float, "verification_code": str, "gaps": list}
        """
        if not self._can_call(OPUS):
            return None

        evidence_text = "\n".join(f"- {e}" for e in supporting_evidence[-3:])

        prompt = f"""Evalúa esta hipótesis matemática sobre {domain.upper()}:

HIPÓTESIS: {hypothesis}

EVIDENCIA EMPÍRICA:
{evidence_text}

Analiza:
1. ¿Es matemáticamente correcta o es un artefacto del método de medición?
2. ¿Qué experimento computacional la falsificaría?
3. ¿Qué tan cerca está de resultados conocidos en la literatura?

Responde SOLO en JSON:
{{
  "valid": true/false,
  "confidence": 0.0-1.0,
  "verification_code": "código Python que intenta falsificar la hipótesis",
  "gaps": ["gap1", "gap2"],
  "known_result": "resultado conocido más cercano o null"
}}"""

        return self._call(OPUS, prompt, label="verify_hypothesis")

    def propose_verify_experiments(self,
                                   problem: str,
                                   supported_summary: str,
                                   cross_summary: str,
                                   ) -> Optional[dict]:
        """
        Propone 2 experimentos falsificables para la fase VERIFY.
        Devuelve: {"experiments": [{"hypothesis": str, "code": str, "expected_if_true": str}, ...]}
        """
        if not self._can_call():
            return None

        prompt = f"""Eres un asistente de investigación. El agente ha terminado de explorar: {problem.upper()}.

HIPÓTESIS SOPORTADAS (evidencia empírica robusta):
{supported_summary}

{cross_summary}

Tu tarea: proponer UN SOLO experimento Python ejecutable que verifique o falsifique la hipótesis más fuerte.
Restricciones del entorno:
- Disponible: numpy (np), sympy (sp) — NO scipy, NO pandas, NO archivos externos. Para correlaciones usa np.corrcoef; para regresión usa np.polyfit; para estadísticas usa numpy puro
- El código debe terminar exactamente con: _result = {{"score": float, "confirms": bool}}
- El experimento debe PODER FALLAR (hipótesis falsificable)
- El código debe ser breve (menos de 20 líneas)

Responde SOLO en JSON, sin texto adicional:
{{"experiments": [{{"hypothesis": "enunciado breve", "code": "import numpy as np\\n# código corto\\n_result = {{\\"score\\": 0.0, \\"confirms\\": True}}", "expected_if_true": "condición de confirmación"}}]}}"""

        return self._call(self._primary_model, prompt, label="propose_verify", max_tokens=1500)

    def cross_domain_insight(self,
                              riemann_hypotheses: list,
                              pnp_hypotheses: list,
                              ) -> Optional[dict]:
        """
        Busca conexiones entre hipótesis de Riemann y P≠NP.
        Se llama cuando ambos problemas tienen hipótesis activas.
        Devuelve: {"connection": str, "exploration_code": str}
        """
        if not self._can_call():
            return None

        r_text = "\n".join(f"- {h['statement']}" for h in riemann_hypotheses[-3:])
        p_text = "\n".join(f"- {h['statement']}" for h in pnp_hypotheses[-3:])

        prompt = f"""Busca conexiones matemáticas entre estas hipótesis:

RIEMANN:
{r_text or 'Sin hipótesis aún.'}

P≠NP:
{p_text or 'Sin hipótesis aún.'}

¿Hay alguna conexión no obvia entre estas líneas de investigación?
Por ejemplo: ¿estructuras espectrales relacionadas? ¿complejidad descriptiva de los ceros?

Responde SOLO en JSON:
{{
  "connection": "descripción de la conexión o 'none' si no hay",
  "exploration_code": "código Python que explora la conexión, o null"
}}"""

        return self._call(self._primary_model, prompt, label="cross_domain")

    # ── Llamada interna ───────────────────────────────────────────────────────

    def _call(self, model: str, prompt: str, label: str = "", max_tokens: int = 600) -> Optional[dict]:
        try:
            full_prompt = (
                f"CONOCIMIENTO DE DOMINIO:\n{self._domain_knowledge}\n\n{prompt}"
                if self._domain_knowledge else prompt
            )
            resp = self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": full_prompt}],
            )
            # Filtrar solo TextBlock (ignorar ThinkingBlock de extended thinking)
            text_blocks = [b for b in resp.content if hasattr(b, "text")]
            if not text_blocks:
                return None
            text    = text_blocks[0].text.strip()
            in_tok  = resp.usage.input_tokens
            out_tok = resp.usage.output_tokens
            cost    = self._record_call(model, in_tok, out_tok)

            print(f"  [Oracle/{label}] {model.split('-')[1]} "
                  f"${cost:.4f} | in={in_tok} out={out_tok} | "
                  f"total=${self._spent:.3f}/{self._budget:.2f}")

            # Extraer JSON de la respuesta
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            return {"raw": text}

        except Exception as e:
            print(f"  [Oracle/{label}] error: {e}")
            return None

    # ── Estado ────────────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        return self._available and self._spent < self._budget

    def summary(self) -> dict:
        now = time.time()
        recent = [c for c in self._call_log if now - c["ts"] < CALL_WINDOW]
        return {
            "available":    self._available,
            "calls_total":  len(self._call_log),
            "calls_hour":   len(recent),
            "spent_usd":    round(self._spent, 4),
            "budget_usd":   self._budget,
            "remaining":    round(self._budget - self._spent, 4),
        }
