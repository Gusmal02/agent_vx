"""
memory.insert — operaciones de escritura/lectura sobre un ResonantField.

Funciones exportadas:
  encode(c, omega_scale)                 → (c_norm, omega_c)
  insert(field, c, omega_scale, alpha_learn, alpha_protect)
  insert_multi(field, c, n_expose, omega_scale, alpha_learn, alpha_protect)
  resonance_scores_mag(omega, omega_c, omega_scale) → scores (N,)
"""

import torch


# ── helpers ───────────────────────────────────────────────────────────────────

def encode(c: torch.Tensor, omega_scale: float = 1.5):
    """
    Convierte un vector de concepto c ∈ R^d en una frecuencia omega_c ∈ R^3
    compatible con el campo.

    Retorna (c_norm, omega_c) donde:
      c_norm  = c / (‖c‖ + ε)          normalizado
      omega_c = omega_scale × proyección de c_norm a R^3
    """
    c = c.float()
    c_norm = c / (c.norm() + 1e-8)

    # Proyectar a 3D: si c tiene ≥3 dims, tomar primeras 3; si no, pad con ceros
    if c_norm.numel() >= 3:
        omega_c = c_norm.flatten()[:3] * omega_scale
    else:
        pad = torch.zeros(3, device=c.device)
        pad[:c_norm.numel()] = c_norm.flatten()
        omega_c = pad * omega_scale

    return c_norm, omega_c


def resonance_scores_mag(
    omega: torch.Tensor,   # (N, 3)  frecuencias del campo
    omega_c: torch.Tensor, # (3,)    frecuencia del concepto
    omega_scale: float = 1.5,
) -> torch.Tensor:
    """
    Similitud coseno entre cada omega_i del campo y omega_c.
    Retorna scores ∈ [-1, 1] de shape (N,).
    """
    omega_c = omega_c.to(omega.device)
    norms_field = omega.norm(dim=1, keepdim=True).clamp(min=1e-8)   # (N, 1)
    norm_c      = omega_c.norm().clamp(min=1e-8)                     # scalar
    scores = (omega @ omega_c) / (norms_field.squeeze(1) * norm_c)  # (N,)
    return scores


# ── escritura ─────────────────────────────────────────────────────────────────

def insert(
    field,
    c: torch.Tensor,
    omega_scale: float = 1.5,
    alpha_learn: float = 0.15,
    alpha_protect: float = 0.05,
) -> None:
    """
    Inserta el concepto c en `field` (ResonantField).

    Selecciona el nodo con mayor resonancia y mueve su omega
    hacia omega_c con tasa alpha_learn.
    También incrementa su protection_factor.
    """
    _, omega_c = encode(c, omega_scale)
    omega_c = omega_c.to(field.device)

    scores = resonance_scores_mag(field.omega, omega_c, omega_scale)
    best   = scores.argmax().item()

    # Plasticidad hebbiana: mover omega[best] hacia omega_c
    field.omega[best] = (
        (1.0 - alpha_learn) * field.omega[best] + alpha_learn * omega_c
    )

    # Registro de tiempo de inserción
    field.t_insert[best] = field.t

    # Protección contra olvido
    field.protection_factor[best] = min(
        1.0,
        float(field.protection_factor[best]) + alpha_protect,
    )


def insert_multi(
    field,
    c: torch.Tensor,
    n_expose: int = 3,
    omega_scale: float = 1.5,
    alpha_learn: float = 0.15,
    alpha_protect: float = 0.05,
) -> None:
    """
    Repite insert n_expose veces con tasa de aprendizaje decreciente
    (simula exposición repetida / consolidación).
    """
    for i in range(n_expose):
        lr = alpha_learn * (0.7 ** i)   # decay suave por exposición
        insert(field, c,
               omega_scale=omega_scale,
               alpha_learn=lr,
               alpha_protect=alpha_protect / n_expose)
