#!/usr/bin/env bash
# start.sh — punto de entrada para sesión en la nube
# Uso: bash start.sh [problema] [agentes] [rondas] [horas]
#   bash start.sh riemann 3 5 0.5
#   bash start.sh causal  4 3 1.0

set -e

PROBLEM=${1:-causal}
N_AGENTS=${2:-3}
N_ROUNDS=${3:-3}
MAX_HOURS=${4:-0.5}

# ── 1. .env ─────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    if [ -n "$ANTHROPIC_API_KEY" ]; then
        echo "ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY" > .env
        echo "[start] .env creado desde variable de entorno"
    else
        echo "[start] ERROR: falta .env o variable ANTHROPIC_API_KEY"
        echo "  Crear .env con: echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env"
        exit 1
    fi
fi

# ── 2. Dependencias ──────────────────────────────────────────────────────────
echo "[start] verificando dependencias..."
python -m pip install --quiet -r requirements.txt

# ── 3. Directorios ───────────────────────────────────────────────────────────
mkdir -p results library

# ── 4. Smoke test rápido ─────────────────────────────────────────────────────
echo "[start] smoke test imports..."
python -c "
from core.living_corpus import LivingCorpus
from core.sandbox_v2 import SandboxV2
from library.knowledge_store import KnowledgeStore
c = LivingCorpus('$PROBLEM')
print('  LivingCorpus OK —', c.summary())
sb = SandboxV2(timeout=10)
r = sb.execute('_result = {\"ok\": True}')
print('  SandboxV2 OK —', r['result'])
"

# ── 5. Lanzar coordinator ────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  problema=$PROBLEM  agentes=$N_AGENTS  rondas=$N_ROUNDS  horas=$MAX_HOURS"
echo "══════════════════════════════════════════════════"
echo ""

python run_v020.py \
    --problem   "$PROBLEM" \
    --n-agents  "$N_AGENTS" \
    --rounds    "$N_ROUNDS" \
    --max-hours "$MAX_HOURS"
