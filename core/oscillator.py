"""
Dinámica de oscilador cuaternión individual.
Estado: s_i = (q_i, ω_i, m_i)
"""

import torch


# ─── Álgebra cuaternión ──────────────────────────────────────────────────────

def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Producto cuaternión a ⊗ b. Convención (w, x, y, z). Shape: (..., 4)."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ], dim=-1)


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)


def omega_to_quat(omega: torch.Tensor) -> torch.Tensor:
    """Convierte ω ∈ ℝ³ al cuaternión puro [0, ω]. Shape: (..., 3) → (..., 4)."""
    zeros = torch.zeros(*omega.shape[:-1], 1, dtype=omega.dtype, device=omega.device)
    return torch.cat([zeros, omega], dim=-1)


def tangent_proj(v: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Proyección de v al espacio tangente de S³ en q: v − ⟨v,q⟩ q."""
    dot = (v * q).sum(dim=-1, keepdim=True)
    return v - dot * q


# ─── Derivadas ───────────────────────────────────────────────────────────────

def dqdt(q: torch.Tensor,
         omega: torch.Tensor,
         q_neighbors_mean: torch.Tensor,
         K: float,
         xi: torch.Tensor | None = None) -> torch.Tensor:
    """
    dq_i/dt = ½ [0,ω_i] ⊗ q_i  +  K · P_{q_i}(q̄_i)  +  ξ_i

    q                  : (N, 4)
    omega              : (N, 3)
    q_neighbors_mean   : (N, 4)  promedio ponderado de vecinos
    """
    rotation = 0.5 * quat_mul(omega_to_quat(omega), q)
    coupling = K * tangent_proj(q_neighbors_mean, q)
    result = rotation + coupling
    if xi is not None:
        result = result + xi
    return result


def domegadt(omega: torch.Tensor,
             omega_neighbors_mean: torch.Tensor,
             gamma_omega: float,
             beta: torch.Tensor | None = None) -> torch.Tensor:
    """
    dω_i/dt = −γ_ω (ω_i − ω̄_i)  +  β_i
    """
    result = -gamma_omega * (omega - omega_neighbors_mean)
    if beta is not None:
        result = result + beta
    return result


def dmdt(m: torch.Tensor,
         delta_q_norm: torch.Tensor,
         gamma_m: float) -> torch.Tensor:
    """
    dm_i/dt = −γ_m · m_i  +  |δq_i|

    m            : (N, 1)  memoria escalar (magnitud acumulada)
    delta_q_norm : (N, 1)  |q_i(t) − q_i*(t)|
    """
    return -gamma_m * m + delta_q_norm
