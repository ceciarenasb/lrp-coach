# LRP Coach — Personal Marathon Training Assistant

A local AI coaching app for marathon training. Generates a personalised week-by-week plan using the Daniels VDOT + Critical Velocity (SVC) hybrid method, adapts the plan each week based on your actual FIT file data and feeling score, and writes coaching notes via a local LLM running on Apple Silicon — no subscriptions, no API keys, no data leaving your machine.

Built as a portfolio piece carved out of a real coaching workflow tailored to [LRP (Let's Run Paris)](https://www.strava.com/clubs/letsrunparis).

## Features

- **Hybrid training zones** — Daniels E/M/T/I/R paces combined with Critical Velocity (Monod-Billat) for a distinct Z4 interval zone
- **Personalised plan generation** — periodised Base → Build → Peak → Taper structure with configurable run days, LRP club runs, strength, and Zwift/cycling sessions
- **Weekly FIT-based adaptation** — uploads Garmin .fit files, scores pace, HR trend, HR drift, volume, and feeling (0–100), then adjusts next week automatically: PROGRESS / MAINTAIN / CONSOLIDATE / RECOVER
- **Mid-plan adjustments** — physio, travel, illness: tell the coach what's changed and the plan mutates in place for the affected weeks
- **Local AI coaching notes** — [Qwen2.5-7B-Instruct 4-bit](https://huggingface.co/mlx-community/Qwen2.5-7B-Instruct-4bit) via mlx-lm, ~60–80 tok/s on M4; falls back gracefully if model isn't downloaded
- **Full state persistence** — profile, zones, plan, and history saved to `data/state.json`; form fields pre-populate on next launch
- **macOS app launcher** — one-click `.app` bundle with custom icon, pinnable to the Dock

## Tech stack

| Layer | Library |
|---|---|
| UI | Gradio 4.44.1 |
| FIT parsing | fitdecode + pandas |
| LLM inference | mlx-lm (Apple Silicon only) |
| Model | mlx-community/Qwen2.5-7B-Instruct-4bit |
| Training science | Daniels VDOT + Monod-Billat Critical Velocity |

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4) — required for mlx-lm
- Python 3.10+
- ~4 GB free disk space for the LLM model (downloaded on first use)

## Setup

```bash
git clone https://github.com/ceciarenasb/lrp-coach.git
cd lrp-coach
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

**Terminal:**
```bash
./scripts/launch.sh   # start server + open browser
./scripts/stop.sh     # stop server
```

**macOS app (first-time setup):**
```bash
./scripts/setup_launcher.sh   # creates LRP Coach.app in /Applications
```

Then double-click **LRP Coach** from Finder or Spotlight.

The app runs at `http://localhost:7860`.

## How it works

### 1. Setup & Plan
Enter your profile (goal race, target time, recent race benchmarks), weekly schedule, and injury status. The app computes your VDOT, derives Critical Velocity, builds training zones, and generates a full periodised plan.

### 2. Run History
Upload `.fit` files from any run. New dates are merged into the history log — already-saved runs are never duplicated. Metrics extracted: distance, duration, pace, avg/max HR, HR drift, cadence, elevation gain.

### 3. My Plan
The full week-by-week plan, updated live after every check-in or adjustment.

### 4. Adjustments
Tell the coach about a disruption (physio, travel, illness). Set which weeks are affected and what to change — replace LRP club runs, remove quality sessions, scale volume. The local LLM writes a coaching response explaining the change.

### 5. Weekly Check-in
Upload the week's FIT files, rate your overall feeling (1–5). The app:
1. Scores the week 0–100 across volume, quality pace, HR trend, HR drift, and feeling
2. Decides PROGRESS (+6% vol) / MAINTAIN / CONSOLIDATE (−5%) / RECOVER (−15%)
3. Mutates next week's plan accordingly
4. Writes a personalised coaching note

## Project structure

```
lrp-coach/
├── app.py                  # Gradio UI and tab logic
├── requirements.txt
├── coach/
│   ├── zones.py            # VDOT + Critical Velocity + zone builder
│   ├── plan.py             # Periodised plan generation
│   ├── adapt.py            # Weekly scoring and adaptation logic
│   ├── fit.py              # FIT file parser (pace, HR, cadence, elevation)
│   ├── llm.py              # mlx-lm coaching note generator
│   ├── adjustments.py      # Mid-plan override logic
│   └── state.py            # JSON persistence
├── data/
│   └── state.json          # Saved profile, zones, plan, history (gitignored)
└── scripts/
    ├── launch.sh
    ├── stop.sh
    └── setup_launcher.sh   # Builds LRP Coach.app with custom icon
```

## Companion project

[lrp-run-analyzer](https://github.com/ceciarenasb/lrp-run-analyzer) — a lightweight Gradio demo (deployed to Hugging Face Spaces) for FIT file analysis and multilingual run-note classification (EN/FR/ES).
