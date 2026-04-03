# Production ML System for Manufacturing Failure Detection (Bosch Dataset)

## Project Overview
This repository delivers a production-oriented ML decision system for Bosch manufacturing failure detection. The system is designed for extremely imbalanced outcomes (about 0.5% positives) and focuses on operational decisions, not only offline leaderboard metrics.

## Problem Statement
Given part-level signals from production lines, estimate failure risk and convert model scores into practical actions:
- auto reject high-risk units,
- send additional units to manual inspection under capacity limits,
- pass the remaining units.

## Key Results
From current production outputs:
- Minimum-cost operating point (`FN=100`, `FP=5`):
  - Recall: `0.4505`
  - Precision: `0.1401`
- Inspection-driven operating point (`10%` inspection budget):
  - Recall: `0.6320`
  - Precision: `0.0368`
- Monitoring (Evidently):
  - Dataset drift share: `0.0`
  - Prediction drift detected: `false`

These results come from:
- `outputs/production_decision_summary.json`
- `outputs/batch_simulation_summary.json`
- `outputs/monitoring/evidently_summary.json`

## System Architecture (High Level)
1. Risk scores are provided from precomputed prediction artifacts.
2. Decision layer converts scores into policy actions (threshold and inspection budget).
3. Batch simulator evaluates operational behavior under production-like windows.
4. Monitoring generates drift diagnostics via Evidently.
5. API and dashboard expose decisions and policy behavior for operators.

## Repository Structure
```text
bosch-production-line-performance/
├── src/
├── apps/
├── scripts/
├── configs/
├── docs/
├── outputs/
├── data/
├── Dockerfile.api
├── Dockerfile.dashboard
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Run Locally
Install dependencies:
```bash
pip install -r requirements.txt
```

Run full pipeline:
```bash
python scripts/run_full_system.py
```

Expected outputs:
- `outputs/production_decision_summary.json`
- `outputs/batch_simulation_summary.json`
- `outputs/monitoring/evidently_summary.json`
- `outputs/monitoring/evidently_report.html`

## Run with Docker
```bash
docker compose up --build
```

## Demo Instructions
Start API:
```bash
uvicorn apps.api.main:app --host 0.0.0.0 --port 8000
```
- Health check: `GET /health`
- Single prediction: `POST /predict`
- Batch prediction: `POST /batch_predict`

Start dashboard:
```bash
streamlit run apps/streamlit_dashboard/app.py
```

## Documentation
- Architecture notes: `docs/architecture.md`
- Case study: `docs/CASE_STUDY_BOSCH_PRODUCTION_SYSTEM.md`
