"""
core/metacognitive.py — Capa metacognitiva: el agente se conoce a sí mismo

El agente puede:
  1. Leer su propio código fuente
  2. Proponer modificaciones a sus propios métodos
  3. Generar nuevas funciones Python para el ToolRegistry
  4. Consultar su historial de intentos fallidos
  5. Saber exactamente qué herramientas tiene disponibles

"Hacer consciente al protoagente de sí mismo."
"""

import ast
import inspect
import textwrap
from pathlib import Path
from typing import Optional


class MetacognitiveLayer:
    """
    Interfaz de auto-conocimiento del agente.

    Permite al agente leer su propio código, entender su estructura,
    proponer mejoras y registrar hipótesis.
    """

    def __init__(self, agent, tool_registry, math_sandbox):
        self._agent       = agent
        self._tools       = tool_registry
        self._sandbox     = math_sandbox
        self._hypotheses: list = []
        self._attempts:   list = []
        self._self_edits: list = []

    # ── Auto-lectura del código ───────────────────────────────────────────────

    def read_own_method(self, method_name: str) -> Optional[str]:
        """Lee el código fuente de un método del agente."""
        method = getattr(type(self._agent), method_name, None)
        if method is None:
            return None
        try:
            return inspect.getsource(method)
        except (OSError, TypeError):
            return None

    def read_own_file(self, relative_path: str) -> Optional[str]:
        """Lee un archivo del propio código fuente del proyecto."""
        p = Path(relative_path)
        if not p.exists():
            return None
        return p.read_text(encoding="utf-8")

    def list_own_methods(self) -> list:
        """Lista todos los métodos públicos del agente."""
        return [
            name for name in dir(type(self._agent))
            if not name.startswith("_") and callable(getattr(type(self._agent), name, None))
        ]

    # ── Propuesta de herramientas ─────────────────────────────────────────────

    def propose_tool(self,
                     name: str,
                     description: str,
                     code: str,
                     test_expr: Optional[str] = None,
                     ) -> dict:
        """
        El agente propone una nueva herramienta (función Python).
        Se verifica sintácticamente antes de registrarla.
        Si test_expr se provee, se evalúa en el sandbox para validación.
        """
        # Verificar sintaxis
        try:
            ast.parse(textwrap.dedent(code))
        except SyntaxError as e:
            return {"ok": False, "reason": f"SyntaxError: {e}"}

        # Test opcional en sandbox
        test_result = None
        if test_expr:
            full_code = textwrap.dedent(code) + f"\n_result = {test_expr}"
            r = self._sandbox.run(full_code, label=f"test_{name}")
            if not r["ok"]:
                return {"ok": False, "reason": f"Test falló: {r['error']}"}
            test_result = r.get("result")

        ok, msg = self._tools.invent(name, code, description)
        self._self_edits.append({
            "type": "propose_tool", "name": name, "ok": ok, "test": test_result
        })
        return {"ok": ok, "msg": msg, "test_result": test_result}

    def propose_code_modification(self,
                                  file_path: str,
                                  old_snippet: str,
                                  new_snippet: str,
                                  reason: str = "",
                                  ) -> dict:
        """
        El agente propone modificar su propio código.
        NO aplica automáticamente — registra la propuesta para evaluación
        humana o para ser probada en un clon.
        """
        record = {
            "file": file_path,
            "old":  old_snippet[:200],
            "new":  new_snippet[:200],
            "reason": reason,
            "applied": False,
        }
        self._self_edits.append(record)
        return {"ok": True, "proposal_id": len(self._self_edits) - 1}

    # ── Hipótesis ─────────────────────────────────────────────────────────────

    def record_hypothesis(self,
                          statement: str,
                          domain: str = "math",
                          confidence: float = 0.5,
                          ) -> int:
        """Registra una hipótesis generada por el agente."""
        hyp = {
            "id":         len(self._hypotheses),
            "statement":  statement,
            "domain":     domain,
            "confidence": confidence,
            "verified":   None,
            "attempts":   0,
        }
        self._hypotheses.append(hyp)
        return hyp["id"]

    def update_hypothesis(self, hyp_id: int,
                          verified: Optional[bool] = None,
                          confidence: Optional[float] = None) -> None:
        if hyp_id < len(self._hypotheses):
            h = self._hypotheses[hyp_id]
            if verified  is not None: h["verified"]   = verified
            if confidence is not None: h["confidence"] = confidence
            h["attempts"] += 1

    def record_attempt(self, description: str, result: str, domain: str = "math") -> None:
        """Registra un intento (exitoso o fallido) para no repetirlo."""
        self._attempts.append({
            "id":          len(self._attempts),
            "description": description,
            "result":      result,
            "domain":      domain,
        })

    def failed_attempts(self, domain: str = None) -> list:
        if domain:
            return [a for a in self._attempts if a["domain"] == domain]
        return list(self._attempts)

    # ── Resumen ───────────────────────────────────────────────────────────────

    def self_report(self) -> dict:
        return {
            "methods_available":   len(self.list_own_methods()),
            "tools_registered":    self._tools.summary()["total"],
            "tools_invented":      self._tools.summary()["invented"],
            "hypotheses":          len(self._hypotheses),
            "hypotheses_verified": sum(1 for h in self._hypotheses if h["verified"] is True),
            "attempts":            len(self._attempts),
            "self_edits_proposed": len(self._self_edits),
        }
