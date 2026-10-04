"""
agente vx — Agente Resonante con ciclo de decisión completo

Uso:
  python main.py                    # modo demostración automática
  python main.py --interactivo      # modo texto interactivo
  python main.py --test             # verifica que las 5 piezas funcionan
"""

import sys
import argparse
import torch
import torch.nn.functional as F

from core.proto_tissue_agent_v2 import ProtoTissueAgentV2


# ── Encoder de texto simple (palabras clave → vector ω) ──────────────────────

_VOCAB = {
    # dominio conceptual
    "crisis":     torch.tensor([ 0.9,  0.2,  0.1]),
    "alerta":     torch.tensor([ 0.8,  0.4,  0.0]),
    "explorar":   torch.tensor([ 0.1,  0.9,  0.2]),
    "nuevo":      torch.tensor([ 0.0,  0.8,  0.4]),
    "conocido":   torch.tensor([ 0.5,  0.0,  0.8]),
    "recordar":   torch.tensor([ 0.4,  0.1,  0.9]),
    "aprender":   torch.tensor([ 0.2,  0.5,  0.8]),
    "decidir":    torch.tensor([ 0.7,  0.6,  0.1]),
    "resolver":   torch.tensor([ 0.6,  0.7,  0.2]),
    "analizar":   torch.tensor([ 0.3,  0.8,  0.5]),
}

def encode_text(text: str) -> torch.Tensor:
    """Convierte texto a vector ω promediando vectores de palabras clave."""
    words = text.lower().split()
    vecs = [_VOCAB[w] for w in words if w in _VOCAB]
    if not vecs:
        # Texto sin palabras conocidas → vector determinista por hash
        h = sum(ord(c) for c in text)
        torch.manual_seed(h % 10000)
        return F.normalize(torch.randn(3), dim=-1)
    return F.normalize(torch.stack(vecs).mean(0), dim=-1)


def generar_candidatos(omega: torch.Tensor, n: int = 5) -> list:
    """Genera candidatos variados alrededor de omega."""
    candidatos = []
    for i in range(n):
        noise = torch.randn(3) * (0.3 + i * 0.1)
        candidatos.append(F.normalize(omega + noise, dim=-1))
    return candidatos


# ── Modo demostración ─────────────────────────────────────────────────────────

def demo():
    print("=" * 60)
    print("Agente Resonante vX — Demostración")
    print("=" * 60)

    agente = ProtoTissueAgentV2(
        N=50, M=5, K=3, seed=42,
        stagnation_threshold=5,
        sim_top_k=3,
        filter_threshold=0.45,
    )

    # Escenarios de ejemplo
    escenarios = [
        ("crisis alerta",   "Dominio A: señal de alerta"),
        ("explorar nuevo",  "Dominio B: exploración"),
        ("conocido recordar", "Dominio C: recuperación"),
        ("crisis alerta",   "Repite dominio A"),
        ("aprender decidir resolver", "Dominio complejo"),
        ("nuevo explorar analizar",   "Dominio B ampliado"),
        ("crisis alerta",   "Tercera vez dominio A"),
        ("conocido recordar aprender", "Fusión C+aprendizaje"),
    ]

    print()
    for i, (texto, etiqueta) in enumerate(escenarios, 1):
        omega = encode_text(texto)
        candidatos = generar_candidatos(omega)
        contexto = omega.clone()

        winner, info = agente.run_cycle(omega, candidatos, context_omega=contexto)

        print(f"[{i}] {etiqueta}")
        print(f"    Entrada: '{texto}'")
        print(f"    ω = [{omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f}]")
        print(f"    nov_perc={info['nov_perc']:.3f}  "
              f"filtrados={info['n_filtered']}  "
              f"simulados={info['n_simulated']}  "
              f"revividos={info['n_revived']}")
        if info['crisis_fired']:
            print(f"    *** CRISIS CREATIVA ACTIVADA — {info['extra_candidates']} candidatos emergentes ***")
        print()

    print("─" * 60)
    r = agente.report()
    print(f"Resumen: {r['n_perceived']} percepciones, {r['n_decided']} decisiones")
    print(f"  winners={r['n_winners']}  dead_ends={r['n_dead_ends']}")
    print(f"  simulaciones Orch-OR={r['n_simulated']}  crisis={r['n_crises']}")
    print(f"  revivals Limbo={r['n_revivals']}  Limbo actual={r['limbo_size']} ideas")


# ── Modo interactivo ──────────────────────────────────────────────────────────

def interactivo():
    print("=" * 60)
    print("Agente Resonante vX — Modo Interactivo")
    print("Escribe texto en español. 'salir' para terminar.")
    print("Palabras clave:", ", ".join(_VOCAB.keys()))
    print("=" * 60)

    agente = ProtoTissueAgentV2(
        N=50, M=5, K=3, seed=42,
        stagnation_threshold=8,
        sim_top_k=3,
        filter_threshold=0.45,
    )

    contexto_anterior = None
    paso = 0

    while True:
        try:
            texto = input(f"\n[{paso+1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if texto.lower() in ("salir", "exit", "q"):
            break
        if not texto:
            continue

        omega = encode_text(texto)
        candidatos = generar_candidatos(omega)
        contexto = contexto_anterior if contexto_anterior is not None else omega

        winner, info = agente.run_cycle(omega, candidatos, context_omega=contexto)

        intuit = agente.subconscious.intuition_score(omega)
        print(f"  ω = [{omega[0]:.2f}, {omega[1]:.2f}, {omega[2]:.2f}]")
        print(f"  Novedad perceptiva: {info['nov_perc']:.3f}  "
              f"({'nuevo' if info['nov_perc'] > 0.6 else 'familiar'})")
        print(f"  Intuición: {intuit['intuition_score']:.3f}  — {intuit['interpretation']}")
        if info['n_revived'] > 0:
            print(f"  Revividas {info['n_revived']} ideas del pasado (contexto cambió)")
        if info['crisis_fired']:
            print(f"  *** Crisis creativa: {info['extra_candidates']} ideas emergentes generadas ***")
        print(f"  Limbo: {agente.limbo.summary()}")

        contexto_anterior = omega
        paso += 1

    print("\nSesión terminada.")
    r = agente.report()
    print(f"Total: {r['n_perceived']} ciclos, {r['n_crises']} crisis, {r['n_revivals']} revivals")


# ── Modo test ─────────────────────────────────────────────────────────────────

def test():
    print("Verificando piezas del agente vx...")
    from core.proto_tissue_agent_v2 import ProtoTissueAgentV2, TissueSnapshot
    from core.subconscious import TwoLayerMind
    from core.context_memory import BufferLimbo
    from core.resonant_tissue import ResonantTissue

    agente = ProtoTissueAgentV2(N=50, M=5, K=3, seed=42, stagnation_threshold=8)

    c_A = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=-1)
    c_B = F.normalize(torch.tensor([0.0, 1.0, 0.0]), dim=-1)

    # Test 1: ciclos básicos
    for _ in range(10):
        cands = [F.normalize(c_A + 0.1*torch.randn(3), dim=-1) for _ in range(4)]
        agente.run_cycle(c_A, cands, context_omega=c_A)
    print("  [✓] Ciclos básicos")

    # Test 2: snapshot efímero
    snap = TissueSnapshot(agente.tissue)
    agente.simulate_candidate(c_B)
    diff = max(float((n.q - snap._nodes_q[i]).abs().max())
               for i, n in enumerate(agente.tissue.nodes))
    assert diff < 1e-5, f"snapshot no restauró (diff={diff})"
    print("  [✓] TissueSnapshot efímero")

    # Test 3: crisis creativa
    agente._stagnation_count = 99
    emergentes = agente._resolve_stagnation(c_A)
    agente._stagnation_count = 0
    assert len(emergentes) > 0
    print("  [✓] Crisis creativa")

    # Test 4: BufferLimbo
    agente.remember_dead_end(c_B, context_omega=c_A)
    revividos = agente.check_revival(c_B)
    assert len(revividos) > 0
    print("  [✓] BufferLimbo revival")

    # Test 5: subconsciente
    for _ in range(3):
        agente.subconscious.insert(c_A, alpha_learn=0.20)
    iscore = agente.subconscious.intuition_score(c_A)
    assert iscore["intuition_score"] >= 0.0
    print("  [✓] Subconsciente activo")

    print("\n>>> AGENTE VX OPERATIVO ✓ <<<")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agente Resonante vX")
    parser.add_argument("--interactivo", action="store_true")
    parser.add_argument("--test",        action="store_true")
    args = parser.parse_args()

    if args.test:
        test()
    elif args.interactivo:
        interactivo()
    else:
        demo()
