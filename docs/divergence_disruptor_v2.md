# DivergenceDisruptor v2 — Diseño

## Problema con v1

El disruptor actual tiene dos capas:
1. **Capa matemática real**: winding number via mpmath — clasifica regiones correctamente
2. **Capa de ataque falsa**: el LLM genera condiciones de ataque como strings (`"n=1000 grande"`, `"high_noise"`) — no son verificaciones numéricas

El resultado es que "ROMPE" y "RESISTE" son opiniones del modelo, no hechos matemáticos.
Los agentes además usan seeds fijos → mismos hallazgos 5 rondas seguidas → no hay exploración.

---

## Diseño v2

### Flujo completo

```
Hallazgos robustos
       │
       ▼
[1] Clasificación por winding number         ← ya existe, funciona
       │
       ├── winding = 0  → ESTABLE, omitir
       │
       └── winding ≠ 0  → TARGET
                │
                ▼
[2] Bisección numérica                       ← nuevo
    dividir región en cuadrantes
    solo entrar donde winding ≠ 0
    hasta width < 0.001 y height < 0.001
                │
                ▼
[3] Verificación del cero aislado            ← nuevo
    mpmath.findroot cerca del centroide
    obtener σ_cero (parte real exacta)
                │
                ├── |σ_cero - 0.5| < 1e-6  → CONFIRMA RH en esa región
                │
                └── |σ_cero - 0.5| ≥ 1e-6  → CONTRAEJEMPLO a RH
                                               guardar (σ, t, |σ-0.5|)
```

### Qué desaparece

- El oráculo LLM para generar ataques — eliminado
- Las condiciones inventadas (`high_noise`, `n=1000`) — eliminadas
- El resultado "ROMPE/RESISTE" basado en texto — reemplazado por número real

### Qué queda

- Clasificación winding (matemática, se mantiene)
- Bisección (ya existe en `test_divergence_disruptor.py`, se integra)
- `mpmath.findroot` para localizar el cero exacto (nuevo)
- Comparación `|σ - 0.5|` para decidir (nuevo)

---

## Implementación

### Método central: `verify_zero(sigma_est, t_est)`

```python
def verify_zero(self, sigma_est, t_est):
    """
    Dado un estimado (σ, t) de un cero, usa mpmath.findroot para
    localizarlo con precisión y verifica si está en σ=0.5.

    Retorna:
        {
          "sigma": float,       # parte real del cero encontrado
          "t": float,           # parte imaginaria
          "on_critical_line": bool,   # |σ - 0.5| < 1e-6
          "deviation": float,   # |σ - 0.5|
        }
    """
    z0 = mpmath.mpc(sigma_est, t_est)
    root = mpmath.findroot(mpmath.zeta, z0)
    sigma_root = float(root.real)
    t_root     = float(root.imag)
    deviation  = abs(sigma_root - 0.5)
    return {
        "sigma":            sigma_root,
        "t":                t_root,
        "on_critical_line": deviation < 1e-6,
        "deviation":        deviation,
    }
```

### Método central: `disrupt(action, sigma_range, t_range)`

```python
def disrupt(self, action, sigma_range, t_range):
    """
    Pipeline completo para un target:
      1. Bisectar hasta aislar el cero
      2. Verificar σ del cero
      3. Retornar resultado matemático

    Retorna dict con keys:
      "action", "zeros_found", "contraejemplos", "confirmaciones"
    """
    zeros = self._bisect(sigma_range, t_range, depth=6)
    contraejemplos = []
    confirmaciones = []

    for sigma_est, t_est, w in zeros:
        result = self.verify_zero(sigma_est, t_est)
        result["winding"] = w
        result["action"]  = action
        if result["on_critical_line"]:
            confirmaciones.append(result)
        else:
            contraejemplos.append(result)

    return {
        "action":          action,
        "zeros_found":     len(zeros),
        "contraejemplos":  contraejemplos,
        "confirmaciones":  confirmaciones,
    }
```

### Output por ronda

```
[Disruptor] atacando 3 targets...
  zero_density_sweep    winding=+2 → 2 ceros aislados
    cero 1: σ=0.500000  t=14.1347  desviación=3.2e-8  → EN LÍNEA CRÍTICA ✓
    cero 2: σ=0.500000  t=21.0220  desviación=1.1e-8  → EN LÍNEA CRÍTICA ✓
  montgomery_correlation winding=-1 → 1 cero aislado
    cero 1: σ=0.500000  t=25.0109  desviación=5.4e-9  → EN LÍNEA CRÍTICA ✓
  explicit_formula_check winding=+3 → 3 ceros aislados
    cero 1: σ=0.500000  t=30.4249  desviación=2.1e-8  → EN LÍNEA CRÍTICA ✓
    ...
[Disruptor] 0 contraejemplos  /  6 confirmaciones numéricas de RH
```

Si algún cero aparece con σ≠0.5:
```
  *** CONTRAEJEMPLO CANDIDATO ***
  cero: σ=0.487312  t=1847.23  desviación=0.012688
  → guardar en results/contraejemplos_riemann.json
```

---

## Exploración genuina: seeds variables

El problema de fondo es que los workers usan seeds fijos → misma exploración 5 rondas.

Cambio necesario en el coordinador: los seeds deben variar por ronda.

```python
# Actual (mal):
seeds = [base_seed + i * 7 for i in range(n_agents)]  # fijos

# Propuesto:
seeds = [base_seed + round_num * 100 + i * 7 for i in range(n_agents)]
```

Así ronda 1 explora t∈[14,50], ronda 2 t∈[50,200], etc. — cada ronda cubre territorio nuevo.

---

## Archivos a modificar

| Archivo | Cambio |
|---|---|
| `run_v020.py` | `DivergenceDisruptor.disrupt()` reemplaza al oráculo LLM |
| `run_v020.py` | Seeds variables por ronda en el coordinador |
| `run_v020.py` | Output del disruptor muestra σ real, desviación, contraejemplos |

El código de bisección (`bisect_to_zero`) ya existe en `test_divergence_disruptor.py` — se mueve a `DivergenceDisruptor`.
