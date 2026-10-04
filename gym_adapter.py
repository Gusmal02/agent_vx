"""
gym_adapter.py — Adaptador Gymnasium → ProtoTissueAgentV2

v0.0.2 — Cambios estructurales:
  - Candidatos obs-condicionados: c_i = normalize(obs_omega + w * action_i)
    El agente elige la mezcla (obs, acción) más familiar a su ejecutivo.
    Esto permite que exec aprenda asociaciones obs→acción, no solo distribución de acciones.
  - Exploración epsilon-greedy al principio para bootstrapear la biblioteca.
    epsilon decae conforme la biblioteca crece.
  - remember_episode() al final del episodio (aprendizaje a nivel episodio).
  - Solo remember_dead_end en paso terminal.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple


# ── Encoder de observación ────────────────────────────────────────────────────

class ObsEncoder:
    """Codifica observaciones numpy → ω ∈ S² via proyección tanh + lineal fija."""

    def __init__(self, obs_dim: int, seed: int = 42):
        torch.manual_seed(seed)
        self.W = F.normalize(torch.randn(obs_dim, 3), dim=0)

    def encode(self, obs: np.ndarray) -> torch.Tensor:
        x = torch.tensor(obs, dtype=torch.float32)
        x = torch.tanh(x * 0.3)
        return F.normalize(x @ self.W, dim=-1)


# ── Encoder de acción ─────────────────────────────────────────────────────────

class ActionEncoder:
    """
    Cada acción discreta → un vector base en S² (distribución Fibonacci).
    Los candidatos finales son obs-condicionados (ver build_candidates).
    """

    def __init__(self, n_actions: int, seed: int = 42):
        torch.manual_seed(seed + 100)
        golden = (1 + 5 ** 0.5) / 2
        self.action_bases: List[torch.Tensor] = []
        for i in range(n_actions):
            theta = torch.acos(torch.tensor(1 - 2 * (i + 0.5) / n_actions))
            phi   = torch.tensor(2 * 3.14159265 * i / golden)
            x = torch.sin(theta) * torch.cos(phi)
            y = torch.sin(theta) * torch.sin(phi)
            z = torch.cos(theta)
            self.action_bases.append(F.normalize(torch.stack([x, y, z]), dim=-1))
        self.n_actions = n_actions

    def build_candidates(self,
                         obs_omega: torch.Tensor,
                         blend: float = 0.6,
                         ) -> List[torch.Tensor]:
        """
        Candidatos obs-condicionados: c_i = normalize(blend*obs + (1-blend)*a_i)

        El ganador refleja qué mezcla (obs, acción) es más familiar al exec.
        El `blend` controla cuánto domina la observación vs la acción base.
        """
        return [
            F.normalize(blend * obs_omega + (1 - blend) * a, dim=-1)
            for a in self.action_bases
        ]

    def winner_to_action(self,
                         winner: torch.Tensor,
                         obs_omega: torch.Tensor,
                         blend: float = 0.6,
                         ) -> int:
        """
        Dada la decisión ganadora, recupera el índice de acción más cercano
        midiendo coseno con los candidatos reconstruidos.
        """
        candidates = self.build_candidates(obs_omega, blend)
        best_cos, best_a = -2.0, 0
        for a, c in enumerate(candidates):
            cos = float(F.cosine_similarity(winner.unsqueeze(0), c.unsqueeze(0)))
            if cos > best_cos:
                best_cos, best_a = cos, a
        return best_a


# ── Loop de evaluación ────────────────────────────────────────────────────────

def evaluar_agente(agente,
                   env_name: str = "CartPole-v1",
                   n_episodios: int = 10,
                   max_steps: int = 500,
                   verbose: bool = True,
                   seed: int = 42,
                   reward_threshold: float = 30.0,
                   reward_max: float = 500.0,
                   version: str = "v0.0.4",
                   epsilon_start: float = 0.5,
                   epsilon_min: float = 0.05,
                   epsilon_decay_ep: int = 20,
                   blend: float = 0.6,
                   ) -> dict:
    """
    Evaluación con candidatos obs-condicionados y epsilon-greedy.

    epsilon decae de epsilon_start a epsilon_min en epsilon_decay_ep episodios.
    Acción aleatoria cuando la biblioteca es pequeña → bootstrapea el aprendizaje.
    """
    import gymnasium as gym

    env = gym.make(env_name)
    obs_dim   = env.observation_space.shape[0]
    n_actions = env.action_space.n

    obs_enc = ObsEncoder(obs_dim, seed=seed)
    act_enc = ActionEncoder(n_actions, seed=seed)
    indices = list(range(n_actions))

    # Registrar entorno activo en el agente (env-tagging v0.0.4)
    if hasattr(agente, 'set_env'):
        agente.set_env(env_name)

    recompensas  = []
    fallos_ep    = []
    pasos_total  = 0
    regimes_ep   = []
    rng = np.random.default_rng(seed)

    for ep in range(n_episodios):
        obs, _ = env.reset(seed=seed + ep)
        recompensa_ep = 0.0
        done   = False
        paso   = 0
        episode_buffer: List[Tuple[torch.Tensor, int, torch.Tensor]] = []
        fast_steps, resonant_steps, random_steps = 0, 0, 0

        # Epsilon actual: decae con el episodio
        epsilon = max(
            epsilon_min,
            epsilon_start * (1 - ep / max(1, epsilon_decay_ep))
        )

        while not done and paso < max_steps:
            omega      = obs_enc.encode(obs)
            context    = omega.clone()
            candidatos = act_enc.build_candidates(omega, blend=blend)

            # ── Exploración aleatoria (epsilon-greedy) ─────────────────────────
            if rng.random() < epsilon:
                accion = int(rng.integers(n_actions))
                winner = candidatos[accion]
                random_steps += 1
            else:
                winner, info = agente.run_cycle(
                    omega, candidatos, context,
                    candidate_indices=indices,
                )
                accion = act_enc.winner_to_action(winner, omega, blend=blend)
                if info.get("regime") == "fast":
                    fast_steps += 1
                else:
                    resonant_steps += 1

            obs, reward, terminated, truncated, _ = env.step(accion)
            done = terminated or truncated

            recompensa_ep += float(reward)
            episode_buffer.append((omega, accion, winner))

            if done:
                agente.remember_dead_end(winner, context_omega=context)

            paso += 1

        # ── Aprendizaje a nivel episodio ──────────────────────────────────────
        fallo = agente.remember_episode(
            episode_steps=episode_buffer,
            total_reward=recompensa_ep,
            reward_threshold=reward_threshold,
            reward_max=reward_max,
            version=version,
        )

        recompensas.append(recompensa_ep)
        fallos_ep.append(fallo)
        pasos_total += paso
        regimes_ep.append({
            "fast": fast_steps, "resonant": resonant_steps, "random": random_steps
        })

        if verbose:
            r_str = f"fast={fast_steps} res={resonant_steps} rnd={random_steps}"
            quality = "BUENO" if fallo == "ok" else f"[{fallo}]"
            print(f"  Ep {ep+1:2d}: recompensa={recompensa_ep:.0f}  "
                  f"pasos={paso}  ε={epsilon:.2f}  [{r_str}]  {quality}")

    env.close()

    return {
        "episodios":        recompensas,
        "recompensa_media": float(np.mean(recompensas)),
        "recompensa_max":   float(np.max(recompensas)),
        "pasos_total":      pasos_total,
        "regimes":          regimes_ep,
        "fallos":           fallos_ep,
        "reporte_agente":   agente.report(),
    }
