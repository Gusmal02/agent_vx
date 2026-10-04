@echo off
cd /d "C:\Users\Gustavo\Documents\tripleten\proyectos\agente vx"
start "Riemann v0.0.9" cmd /k "uv run python run_v009.py --problem riemann --hours 8 2>&1 | tee log_riemann.txt"
timeout /t 3 /nobreak > nul
start "PNP v0.0.9"     cmd /k "uv run python run_v009.py --problem pnp    --hours 8 2>&1 | tee log_pnp.txt"
echo Ambos agentes lanzados.
