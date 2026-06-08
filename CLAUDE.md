# LRP Coach — project context for Claude

## What this is
A local AI marathon coaching assistant for Cecilia Arenas (LRP club runner, Paris).
Generates a personalised periodised plan, adapts it weekly from Garmin FIT data and a
feeling score, and writes coaching notes via a local LLM on Apple Silicon.
Not deployed to any cloud — runs entirely on her MacBook M4, free.

## Goals / constraints
- **Free and local**: no paid APIs, no cloud GPU, no data leaving the machine.
- **Apple Silicon only**: mlx-lm requires it; do not suggest HF Spaces or cloud deployment for this project.
- **Multilingual athlete**: Cecilia's notes may come in FR/EN/ES — keep that in mind for any text analysis.
- **Real coaching replacement**: the adaptation logic must be meaningful (not just cosmetic). Think like a human coach who reads pace, HR drift, and feeling together.
- **Ask before pushing**: always confirm before `git push` or any action that touches Cecilia's GitHub account.

## Stack
- Python 3.10+ · Gradio 4.44.1
- fitdecode + pandas (FIT parsing)
- mlx-lm · mlx-community/Qwen2.5-7B-Instruct-4bit (local LLM, lazy-loaded)
- Training science: Daniels VDOT + Monod-Billat Critical Velocity (hybrid zones)
- State: `data/state.json` (gitignored)

## File map
```
app.py                  Gradio UI — 5 tabs, all persistence logic
coach/
  zones.py              VDOT formula, CV, zone builder, Zones dataclass
  plan.py               Periodised plan generation (Base/Build/Peak/Taper)
  adapt.py              Weekly scoring (0-100) → PROGRESS/MAINTAIN/CONSOLIDATE/RECOVER
  fit.py                FIT parser: pace, HR, HR drift, cadence, elevation gain
  llm.py                mlx-lm coaching note generator (lazy model load)
  adjustments.py        Mid-plan override logic + LLM prompt builder
  state.py              load/save/update for data/state.json
data/
  state.json            Persisted athlete profile, zones, plan, history (gitignored)
  .gitkeep
scripts/
  launch.sh             Start server + open browser (idempotent)
  stop.sh               Kill server by PID file or port scan
  setup_launcher.sh     Build LRP Coach.app in /Applications (run once)
```

## How to run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./scripts/launch.sh        # or: python app.py
./scripts/stop.sh
```
First-time macOS launcher:
```bash
./scripts/setup_launcher.sh   # creates /Applications/LRP Coach.app
```

## Coding conventions
- No comments unless the WHY is non-obvious.
- No extra abstractions beyond what the current task needs.
- Rule-based logic for plan mutations; LLM only writes human-facing text.
- The Zones dataclass is the single source of truth for all pace targets.
- `state_mod.load()` at the top of any function that reads state — never cache state across requests.
- FIT metrics: distance (km), duration (s), avg_pace_s (s/km), avg_hr, max_hr, hr_drift_pct, avg_cadence_spm, elevation_gain_m.

## Known quirks
- **gradio_client boolean schema bug**: patched in `.venv` — `get_type()` and `_json_schema_to_python_type()` in `gradio_client/utils.py` guard against `isinstance(schema, bool)`. If the venv is recreated the patch must be reapplied.
- **mlx-lm first run**: model downloads ~4 GB on first coaching note request; subsequent calls are fast (~60-80 tok/s on M4).
- **`gh` CLI path**: `/opt/homebrew/bin/gh` (not on default PATH in tool context).

## Companion project
`/Users/ceciliaarenas/code/lrp-run-analyzer` — public HF Space demo (FIT analysis + multilingual note classifier). Deployed to both GitHub (`ceciarenasb/lrp-run-analyzer`) and HF Space (`ceciarenas/lrp-run-analyzer`).
