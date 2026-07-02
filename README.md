# DataPilot AI

Autonomous AI CSV analytics workspace built with FastAPI, React, TypeScript, Tailwind, ECharts, and scikit-learn.

## Quick Start

Backend:

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

Create `backend/.env` from `backend/.env.example` to enable OpenAI-written insights and chat.

## Design

The LLM is used only as planner/interpreter/explainer. Statistics, visualization data, RAG artifacts, and ML predictions are computed by backend services.
