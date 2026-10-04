"""
core/proto_tissue_agent_v2.py — ProtoTissueAgentV2

Ciclo de decisión completo (Nivel 2 de arquitectura_agente_vx.md):

  perceive(ω)
    → subconscious.update(ω)
    → filter_candidates(candidates)
    → [si stagnation] _resolve_stagnation()   # crisis creativa
    → [top-k] simulate_candidate(c)           # Orch-OR sandbox
    → decide(simulated, context)              # exec + intuition_score
    → remember(winner) / limbo.add(dead_ends)
    → check_revival(context)                  # BufferLimbo

Subcampos (N=50 recomendado):
  perc_ids:  0..N//5        — perceptivo
  imag_ids:  N//5..2N//5   — imaginativo (sandbox Orch-OR)
  sub_ids:   2N//5..3N//5  — subconsciente (priming)
  exec_ids:  3N//5..4N//5  — ejecutivo (winners)
  mem_ids:   4N//5..N      — memoria dead-ends
"""

from __future__ import annotations

import math
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from core.resonant_tissue import ResonantTissue
from core.subconscious import TwoLayerMind
from core.context_memory import BufferLimbo
from core.domain_context import ContextModule
from core.domain_memory import MemoryModule
from core.tool_registry import ToolRegistry


# ── Utilidad: grafo BA ────────────────────────────────────────────────────────

def _build_ba_graph(N: int, m: int = 3, seed: int = 42) -> torch.Tensor:
    import random
    rng = random.Random(seed)
    edges: set = set()
    for i in range(min(m + 1, N)):
        for j in range(i + 1, min(m + 1, N)):
            edges.add((i, j)); edges.add((j, i))
    degree = [0] * N
    for i, j in edges:
        degree[i] += 1
    for new_node in range(m + 1, N):
        total_deg = sum(degree[:new_node]) or 1
        probs = [degree[k] / total_deg for k in range(new_node)]
        chosen: set = set()
        attempts = 0
        while len(chosen) < m and attempts < 10 * m:
            r = rng.random(); cum = 0.0
            for k, p in enumerate(probs):
                cum += p
                if r < cum:
                    chosen.add(k); break
            attempts += 1
        for k in chosen:
            edges.add((new_node, k)); edges.add((k, new_node))
            degree[new_node] += 1; degree[k] += 1
    src, dst = zip(*edges) if edges else ([], [])
    return torch.tensor([list(src), list(dst)], dtype=torch.long)


# ── TissueSnapshot ────────────────────────────────────────────────────────────

class TissueSnapshot:
    """Captura y restaura el estado completo de un ResonantTissue."""

    def __init__(self, tissue: ResonantTissue):
        self._nodes_q         = [n.q.clone() for n in tissue.nodes]
        self._nodes_q_local   = [n.q_local.clone() for n in tissue.nodes]
        self._nodes_act_log   = [list(n._activation_log) for n in tissue.nodes]
        self._edges_r         = {k: e.r.clone() for k, e in tissue.edges.items()}
        self._edges_err       = {k: e._last_error for k, e in tissue.edges.items()}

    def restore(self, tissue: ResonantTissue) -> None:
        for i, n in enumerate(tissue.nodes):
            n.q              = self._nodes_q[i].clone()
            n.q_local        = self._nodes_q_local[i].clone()
            n._activation_log = list(self._nodes_act_log[i])
        for k, e in tissue.edges.items():
            e.r             = self._edges_r[k].clone()
            e._last_error   = self._edges_err[k]


# ── ProtoTissueAgentV2 ────────────────────────────────────────────────────────

class ProtoTissueAgentV2:
    """
    Agente resonante con ciclo de decisión completo (Nivel 2 vX).

    Parámetros
    ----------
    N                   : nodos del tejido (múltiplo de 5 recomendado, ≥25)
    M                   : osciladores por nodo
    K                   : resonadores por arista
    seed                : semilla reproducible
    stagnation_threshold: pasos sin mejora antes de activar crisis creativa
    sim_top_k           : candidatos a simular en Orch-OR antes de decidir
    filter_threshold    : umbral de novelty en mem para considerar dead-end
    """

    def __init__(self,
                 N: int = 50,
                 M: int = 5,
                 K: int = 3,
                 seed: int = 42,
                 stagnation_threshold: int = 10,
                 sim_top_k: int = 3,
                 filter_threshold: float = 0.50,
                 library=None,
                 confidence_threshold: float = 0.93,
                 t_settle: int = 5,
                 warmup_steps: int = 50,
                 subconscious_update_freq: int = 1,
                 use_orch_or: bool = True):
        torch.manual_seed(seed)
        self.N = N; self.M = M; self.K = K
        self.sim_top_k = sim_top_k
        self.filter_threshold = filter_threshold

        # ── Subcampos: 5 partes iguales ──────────────────────────────────────
        step = N // 5
        self.perc_ids = list(range(0,         step))
        self.imag_ids = list(range(step,      2 * step))
        self.sub_ids  = list(range(2 * step,  3 * step))
        self.exec_ids = list(range(3 * step,  4 * step))
        self.mem_ids  = list(range(4 * step,  N))

        # ── Tejido principal ─────────────────────────────────────────────────
        edge_index = _build_ba_graph(N, m=3, seed=seed)
        self.tissue = ResonantTissue(N, M, K, edge_index, seed=seed)

        # ── Subconsciente (TwoLayerMind) ─────────────────────────────────────
        self.t_settle = t_settle
        self.subconscious = TwoLayerMind(
            N=step, K_C=2.0, K_S=1.2,
            omega_std=0.20, seed=seed,
        )
        self.subconscious.warmup(steps=warmup_steps)

        # ── BufferLimbo ──────────────────────────────────────────────────────
        self.limbo = BufferLimbo(ctx_threshold_deg=30.0)

        # ── Estancamiento ────────────────────────────────────────────────────
        self.stagnation_threshold  = stagnation_threshold
        self._stagnation_count     = 0
        self._last_novelty_perc    = 1.0

        # ── Biblioteca resonante (opcional) ──────────────────────────────────
        self.library                   = library
        self.confidence_threshold      = confidence_threshold
        self.subconscious_update_freq  = subconscious_update_freq
        self._step_counter             = 0
        self._use_orch_or              = use_orch_or
        self._current_env: Optional[str] = None   # env activo (set_env)

        # ── Contadores de diagnóstico ─────────────────────────────────────────
        self.n_perceived          = 0
        self.n_decided            = 0
        self.n_filtered           = 0
        self.n_simulated          = 0
        self.n_crises             = 0
        self.n_revivals           = 0
        self.n_dead_ends          = 0
        self.n_winners            = 0
        self.n_fast_decisions     = 0
        self.n_resonant_decisions = 0

        # ── Clasificación de fallos (v0.0.4) ─────────────────────────────────
        self.failure_log: List[dict] = []

        # ── Módulos de dominio (v0.0.6) ───────────────────────────────────────
        self.context_module  = ContextModule(window_size=10, dim=3)
        self.memory_module   = MemoryModule(library, env_name="unknown")
        self.tool_registry   = ToolRegistry()
        self._clone_mode: bool = False   # True en clones — no escriben en biblioteca

    # ═══════════════════════════════════════════════════════════════════════════
    # Interfaz principal
    # ═══════════════════════════════════════════════════════════════════════════

    def set_env(self, env_name: str) -> None:
        """Registra el entorno activo. Filtra la biblioteca y memoria por este entorno."""
        self._current_env = env_name
        self.memory_module.set_env(env_name)

    def perceive(self, omega: torch.Tensor) -> float:
        """
        Actualiza subcampo perceptivo y subconsciente.
        Devuelve novelty en perc (qué tan nuevo es este estado).
        """
        self.tissue.insert_pattern(
            node_ids=self.perc_ids, omega=omega, alpha=0.20, n_expose=3,
        )
        self._step_counter += 1
        if self._step_counter % self.subconscious_update_freq == 0:
            self.subconscious.insert(omega, alpha_learn=0.15, T_settle=self.t_settle)
        nov = self.tissue.novelty(omega, node_ids=self.perc_ids)
        self._check_stagnation(nov)
        self.n_perceived += 1
        return nov

    def filter_candidates(self,
                          candidates: List[torch.Tensor],
                          threshold: Optional[float] = None,
                          ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Separa candidatos en buenos y malos según novelty en mem_ids.
        Baja novelty en mem = similar a dead-end conocido → malo.
        Devuelve (good, bad). Si todos son malos, devuelve todos como good.
        """
        thr = threshold if threshold is not None else self.filter_threshold
        good, bad = [], []
        for c in candidates:
            nov_mem = self.tissue.novelty(c, node_ids=self.mem_ids)
            if nov_mem < thr:
                bad.append(c)
            else:
                good.append(c)
        self.n_filtered += len(bad)
        return (list(candidates), []) if not good else (good, bad)

    def simulate_candidate(self, c: torch.Tensor) -> float:
        """
        Orch-OR: inserta c en imag_ids de forma efímera, mide novelty en exec_ids,
        luego revierte. Mayor resonancia en exec = mejor candidato.
        Devuelve novelty_exec_simulada (menor = más prometedor).
        """
        snap = TissueSnapshot(self.tissue)
        self.tissue.insert_pattern(
            node_ids=self.imag_ids, omega=c, alpha=0.25, n_expose=3,
        )
        nov_exec = self.tissue.novelty(c, node_ids=self.exec_ids)
        snap.restore(self.tissue)
        self.n_simulated += 1
        return nov_exec

    def decide(self,
               candidates: List[torch.Tensor],
               context_omega: Optional[torch.Tensor] = None,
               ) -> torch.Tensor:
        """
        Decide entre candidatos usando:
          score(c) = nov_exec(c) − w_int * intuition_score(c)

        Menor score = mejor candidato (más familiar al ejecutivo + intuición).
        Si context_omega se provee, integra intuition_score del subconsciente.
        """
        if not candidates:
            raise ValueError("decide() necesita al menos un candidato")

        w_int = 0.30  # peso de la intuición

        scored = []
        for c in candidates:
            nov_exec = self.tissue.novelty(c, node_ids=self.exec_ids)
            intuition = 0.0
            if context_omega is not None:
                iscore = self.subconscious.intuition_score(c)
                intuition = float(iscore.get("intuition_score", 0.0))
            score = nov_exec - w_int * intuition
            scored.append((c, score))

        self.n_decided += 1
        return min(scored, key=lambda x: x[1])[0]

    def remember_winner(self, omega: torch.Tensor) -> None:
        """Registra un winner en exec_ids y en el subconsciente."""
        self.tissue.insert_pattern(
            node_ids=self.exec_ids, omega=omega, alpha=0.30, n_expose=5,
        )
        self.subconscious.insert(omega, alpha_learn=0.25, T_settle=self.t_settle)
        self._stagnation_count = 0   # resolver algo rompe el estancamiento
        self.n_winners += 1

    def remember_dead_end(self,
                          omega: torch.Tensor,
                          context_omega: Optional[torch.Tensor] = None,
                          ) -> None:
        """
        Registra un dead-end en mem_ids.
        Si se provee context_omega, también lo guarda en BufferLimbo.
        """
        self.tissue.insert_pattern(
            node_ids=self.mem_ids, omega=omega, alpha=0.30, n_expose=5,
        )
        if context_omega is not None:
            self.limbo.add_dead(omega, context_omega)
        self.n_dead_ends += 1

    def check_revival(self,
                      context_omega: torch.Tensor,
                      ) -> List[torch.Tensor]:
        """
        Consulta BufferLimbo: devuelve candidatos revivibles en el contexto actual.
        Candidatos revividos se eliminan del Limbo.
        """
        revivals = self.limbo.check_revival(context_omega)
        revived = []
        for idea, _angle, _cos in revivals:
            revived.append(idea.omega)
            self.limbo.mark_winner(idea, context_omega)
        self.n_revivals += len(revived)
        return revived

    # ═══════════════════════════════════════════════════════════════════════════
    # Ciclo completo V2
    # ═══════════════════════════════════════════════════════════════════════════

    def run_cycle(self,
                  omega: torch.Tensor,
                  candidates: List[torch.Tensor],
                  context_omega: Optional[torch.Tensor] = None,
                  candidate_indices: Optional[List[int]] = None,
                  ) -> Tuple[torch.Tensor, dict]:
        """
        Ciclo de decisión adaptativo.

        Régimen RÁPIDO: si la biblioteca tiene un match con cos >= confidence_threshold,
        devuelve esa acción directamente sin simulación Orch-OR.

        Régimen RESONANTE: ciclo V2 completo (perceive → filter → Orch-OR → decide).

        candidate_indices: índices enteros de cada candidato (para mapear library → omega).
        """
        # Enriquecer contexto con ventana episódica (v0.0.6)
        ctx_vec = self.context_module.context_vector()
        if ctx_vec is not None:
            # Blend suave: 80% omega actual + 20% historial reciente
            ctx = F.normalize(0.8 * omega + 0.2 * ctx_vec, dim=-1)
        else:
            ctx = context_omega if context_omega is not None else omega

        # ── Régimen RÁPIDO (biblioteca / memory_module) ───────────────────────
        if self.library is not None and candidate_indices is not None:
            result = self.library.query_resonance(omega, min_samples=30, env_name=self._current_env)
            if result is not None:
                action_idx, cos_sim, lib_reward = result
                if cos_sim >= self.confidence_threshold and action_idx < len(candidates):
                    nov_perc = self.perceive(omega)
                    winner   = candidates[action_idx]
                    self.n_fast_decisions += 1
                    return winner, {
                        "nov_perc":         nov_perc,
                        "regime":           "fast",
                        "library_cos":      cos_sim,
                        "library_reward":   lib_reward,
                        "stagnation":       self._stagnation_count,
                        "crisis_fired":     False,
                        "n_revived":        0,
                        "n_filtered":       0,
                        "n_simulated":      0,
                        "pool_size":        len(candidates),
                        "extra_candidates": 0,
                    }

        # ── Régimen RESONANTE (ciclo V2 completo) ─────────────────────────────
        # 1. Percepción
        nov_perc = self.perceive(omega)

        # 2. Candidatos revivibles del Limbo
        revived = self.check_revival(ctx)
        all_candidates = list(candidates) + revived

        # 3. Filtrar dead-ends conocidos
        good, bad = self.filter_candidates(all_candidates)

        # 4. Crisis creativa si hay estancamiento
        extra_candidates: List[torch.Tensor] = []
        crisis_fired = False
        if self._stagnation_count >= self.stagnation_threshold:
            extra_candidates = self._resolve_stagnation(ctx)
            crisis_fired = True
            self._stagnation_count = 0

        pool = good + extra_candidates if extra_candidates else good

        # 5. Simulación Orch-OR en top-k candidatos
        # Solo simula cuando la biblioteca no tiene datos suficientes (exploración activa).
        # Con biblioteca activa, la decisión ya es confiable — evitar costo de snapshot.
        lib_size = self.library.count() if self.library is not None else 0
        run_orch_or = self._use_orch_or and lib_size < 30

        if run_orch_or:
            if len(pool) > self.sim_top_k:
                pre_scored = sorted(
                    pool,
                    key=lambda c: self.tissue.novelty(c, node_ids=self.exec_ids)
                )
                to_simulate = pre_scored[:self.sim_top_k]
            else:
                to_simulate = pool
            for c in to_simulate:
                self.simulate_candidate(c)
        else:
            to_simulate = pool  # sin Orch-OR: decide sobre el pool directamente

        # 6. Decisión
        winner = self.decide(to_simulate, context_omega=ctx)

        # Aprendizaje débil por paso (solo resonante, alpha bajo para no fijar antes del episodio)
        self.tissue.insert_pattern(
            node_ids=self.exec_ids, omega=winner, alpha=0.05, n_expose=1,
        )

        self.n_resonant_decisions += 1
        info = {
            "nov_perc":         nov_perc,
            "regime":           "resonant",
            "stagnation":       self._stagnation_count,
            "crisis_fired":     crisis_fired,
            "n_revived":        len(revived),
            "n_filtered":       len(bad),
            "n_simulated":      len(to_simulate),
            "pool_size":        len(pool),
            "extra_candidates": len(extra_candidates),
        }
        return winner, info

    # ═══════════════════════════════════════════════════════════════════════════
    # Utilidades internas
    # ═══════════════════════════════════════════════════════════════════════════

    def _check_stagnation(self, novelty_perc: float) -> None:
        """Incrementa contador si la novelty perceptiva no mejora."""
        if novelty_perc >= self._last_novelty_perc * 0.95:
            self._stagnation_count += 1
        else:
            self._stagnation_count = max(0, self._stagnation_count - 1)
        self._last_novelty_perc = novelty_perc

    def _resolve_stagnation(self,
                             context_omega: torch.Tensor,
                             ) -> List[torch.Tensor]:
        """
        Crisis creativa: genera candidatos nuevos cuando el agente está atascado.

        Estrategia:
          1. Encontrar los dos nodos del exec_ids más separados angularmente.
          2. c_emergente = normalizar(polo_pos − polo_neg) + ruido ortogonal.
          3. Evaluar en imaginación; devolver los que reducen bimodalidad.
        """
        self.n_crises += 1
        c = self.tissue.centroids_total()
        exec_c = c[torch.tensor(self.exec_ids)]  # (n_exec, 4)

        # Pares más separados en exec
        best_cos, best_i, best_j = 2.0, 0, 1
        n = exec_c.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                cos_ij = float(
                    (F.normalize(exec_c[i], dim=-1) @
                     F.normalize(exec_c[j], dim=-1)).clamp(-1, 1)
                )
                if cos_ij < best_cos:
                    best_cos, best_i, best_j = cos_ij, i, j

        c_pos = F.normalize(exec_c[best_i, 1:], dim=-1)   # xyz del cuaternión
        c_neg = F.normalize(exec_c[best_j, 1:], dim=-1)

        # c_emergente: dirección a 90° del eje pos–neg
        axis = F.normalize(c_pos - c_neg, dim=-1)
        perp = F.normalize(context_omega.float() - (context_omega.float() @ axis) * axis, dim=-1)
        c_emergente = F.normalize(axis + perp, dim=-1)

        # Añadir variantes con pequeño ruido
        candidates_out: List[torch.Tensor] = [c_emergente]
        for _ in range(2):
            noise = F.normalize(torch.randn(3), dim=-1) * 0.1
            candidates_out.append(F.normalize(c_emergente + noise, dim=-1))

        return candidates_out

    def remember_episode(self,
                         episode_steps: List[Tuple[torch.Tensor, int, torch.Tensor]],
                         total_reward: float,
                         reward_threshold: float = 30.0,
                         reward_max: float = 500.0,
                         version: str = "v0.0.4",
                         ) -> str:
        """
        Registra un episodio completo en tejido + biblioteca.

        Devuelve el tipo de fallo clasificado si el episodio fue malo:
          'ignorance' — fallo por falta de datos de este entorno
          'stale'     — fallo pese a tener biblioteca, posible memoria caducada
          'limit'     — fallo consistente con biblioteca amplia (límite arquitectónico)
          'ok'        — episodio bueno, sin fallo
        """
        env_name = self._current_env or "unknown"
        norm_reward = total_reward / max(1.0, abs(reward_max))
        n_ep = len(episode_steps)

        if reward_threshold >= 0:
            tissue_threshold = reward_threshold * 0.4
        else:
            tissue_threshold = reward_threshold / 0.4

        is_good_tissue  = total_reward >= tissue_threshold
        is_good_library = total_reward >= reward_threshold
        norm_clamped    = max(0.0, min(1.0, norm_reward))

        # Cerrar episodio en context_module (v0.0.6)
        self.context_module.close_episode()

        if is_good_tissue:
            n_exp = max(2, min(10, int(2 + 8 * norm_clamped)))
            alpha = min(0.40, 0.15 + 0.25 * norm_clamped)
            for obs_omega, _, winner_omega in episode_steps:
                self.tissue.insert_pattern(
                    node_ids=self.exec_ids,
                    omega=winner_omega, alpha=alpha, n_expose=n_exp,
                )
            if is_good_library:
                for obs_omega, _, __ in episode_steps:
                    self.subconscious.insert(obs_omega, alpha_learn=alpha * 0.8, T_settle=self.t_settle)
            self._stagnation_count = 0
            self.n_winners += n_ep

            # Consolidar memoria semántica con los omegas del episodio bueno (v0.0.6)
            ep_omegas = [o for o, _, __ in episode_steps]
            ep_rewards = [total_reward / max(1, n_ep)] * n_ep
            self.memory_module.consolidate_semantic(ep_omegas, ep_rewards, min_reward=0.0)

            if is_good_library and self.library is not None and not self._clone_mode:
                lib_steps = [(obs_omega, action_idx) for obs_omega, action_idx, _ in episode_steps]
                self.library.store_episode(
                    lib_steps, total_reward,
                    reward_max=reward_max, regime="resonant",
                    version=version, env_name=env_name,
                )
            return "ok"
        else:
            for obs_omega, _, __ in episode_steps:
                self.tissue.insert_pattern(
                    node_ids=self.mem_ids, omega=obs_omega, alpha=0.20, n_expose=3,
                )
            self.n_dead_ends += n_ep

            # Clasificar tipo de fallo
            lib_env_count = self.library.count(env_name=env_name) if self.library else 0
            if lib_env_count < 30:
                failure_type = "ignorance"
            elif lib_env_count < 100:
                failure_type = "stale"
            else:
                failure_type = "limit"

            self.failure_log.append({
                "env":          env_name,
                "reward":       total_reward,
                "threshold":    reward_threshold,
                "lib_count":    lib_env_count,
                "failure_type": failure_type,
            })
            return failure_type

    # ═══════════════════════════════════════════════════════════════════════════
    # Hebbian + Calibración
    # ═══════════════════════════════════════════════════════════════════════════

    def step_hebbian(self) -> None:
        """Un paso de plasticidad Hebbian en las aristas del tejido."""
        self.tissue.step(alpha_node=0.05, eta_edge=0.02)

    def calibrate_zone(self,
                       zone_ids: List[int],
                       omegas: List[torch.Tensor],
                       n_expose: int = 5) -> None:
        """Inserta omegas en un subconjunto de nodos del tejido."""
        for omega in omegas:
            self.tissue.insert_pattern(
                node_ids=zone_ids, omega=omega, alpha=0.30, n_expose=n_expose,
            )

    def warmup(self, omegas: List[torch.Tensor], n_expose: int = 5) -> None:
        """Calibra los 5 subcampos con una lista de omegas de referencia."""
        for omega in omegas:
            for zone in [self.perc_ids, self.imag_ids, self.sub_ids,
                         self.exec_ids, self.mem_ids]:
                self.tissue.insert_pattern(
                    node_ids=zone, omega=omega, alpha=0.20, n_expose=n_expose,
                )
            self.subconscious.insert(omega, T_settle=self.t_settle)

    # ═══════════════════════════════════════════════════════════════════════════
    # Reporte
    # ═══════════════════════════════════════════════════════════════════════════

    def failure_summary(self) -> dict:
        """Resumen de fallos clasificados: conteos por tipo y diagnóstico dominante."""
        counts = {"ignorance": 0, "stale": 0, "limit": 0}
        for f in self.failure_log:
            counts[f["failure_type"]] = counts.get(f["failure_type"], 0) + 1
        total_fails = sum(counts.values())
        dominant = max(counts, key=counts.get) if total_fails > 0 else "none"
        return {
            "total_failures": total_fails,
            "by_type":        counts,
            "dominant":       dominant,
        }

    def report(self) -> dict:
        env_name    = self._current_env
        lib_summary = self.library.summary(env_name=env_name) if self.library is not None else None
        return {
            "N": self.N, "M": self.M, "K": self.K,
            "current_env": env_name,
            "subcampos": {
                "perc":  len(self.perc_ids),
                "imag":  len(self.imag_ids),
                "sub":   len(self.sub_ids),
                "exec":  len(self.exec_ids),
                "mem":   len(self.mem_ids),
            },
            "n_perceived":          self.n_perceived,
            "n_decided":            self.n_decided,
            "n_filtered":           self.n_filtered,
            "n_simulated":          self.n_simulated,
            "n_crises":             self.n_crises,
            "n_revivals":           self.n_revivals,
            "n_dead_ends":          self.n_dead_ends,
            "n_winners":            self.n_winners,
            "n_fast_decisions":     self.n_fast_decisions,
            "n_resonant_decisions": self.n_resonant_decisions,
            "stagnation":           self._stagnation_count,
            "limbo_size":           len(self.limbo.ideas) if hasattr(self.limbo, 'ideas') else -1,
            "library":              lib_summary,
            "failures":             self.failure_summary(),
        }
