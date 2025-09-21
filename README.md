# RateLimit-AI

This repo is a **simulation harness** for experimenting with rate limiting and quota enforcement under GenAI workloads.  
We will incrementally build out components to reproduce different traffic patterns and compare algorithms.

---

## Current Setup

### Backend Service
- A FastAPI app (`backend/app.py`) that simulates a GenAI inference backend.
- Endpoint:
  - `POST /infer`
    ```json
    { "tenant": "T1", "model": "medium", "prompt_tokens": 200 }
    ```
    → returns number of output tokens generated.
- Output tokens are sampled from a lognormal distribution per model:
  - **small**: mean ~12 tokens
  - **medium**: mean ~28 tokens
  - **large**: mean ~180 tokens
- Output grows with input (`prompt_tokens`).

---

## Running the Simulation Backend

### 1. Environment
```bash
git clone git@github.com:<your-username>/ratelimit-ai.git
cd ratelimit-ai/backend
python3 -m venv .venv
source .venv/bin/activate   # use `source .venv/bin/activate.fish` for fish shell
pip install -r requirements.txt
