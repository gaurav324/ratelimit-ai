from typing import Optional
import os, time, uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse
import httpx

from cost_estimator import est_tokens_out

# ---- config ----
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:9000")  # where backend/app.py runs
BACKEND_TIMEOUT = float(os.getenv("BACKEND_TIMEOUT_SEC", "0.5"))

# ---- prometheus metrics ----
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQS = Counter("gateway_requests_total", "Total gateway requests", ["code"])
BACKEND_ERR = Counter("gateway_backend_errors_total", "Backend errors")
LAT = Histogram(
    "gateway_latency_seconds",
    "Latency of /infer end-to-end (seconds)",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5)
)

app = FastAPI(title="ratelimit-ai gateway v0 (pass-through)")

# a single async client reused across requests
client: Optional[httpx.AsyncClient] = None

@app.on_event("startup")
async def startup():
    global client
    client = httpx.AsyncClient(timeout=BACKEND_TIMEOUT)

@app.on_event("shutdown")
async def shutdown():
    global client
    if client:
        await client.aclose()

@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/infer")
async def infer(req: Request):
    """
    Pass-through: accept the same JSON body as the backend expects and forward it.
    Adds an x-request-id header (for future correlation).
    """
    t0 = time.perf_counter()
    rid = str(uuid.uuid4())

    try:
        payload = await req.json()
    except Exception as e:
        REQS.labels("400").inc()
        return JSONResponse({"error": f"invalid json: {e}"}, status_code=400)

    tenant = payload.get("tenant", "T0001")
    model = payload.get("model", "small")
    prompt_tokens = payload.get("prompt_tokens", 0)

    # --- compute estimate (tokens_out_est + prompt_tokens) ---
    try:
        out_est = est_tokens_out(model, prompt_tokens)
    except Exception as e:
        # If anything goes wrong in estimation, fall back to prompt-only cost
        out_est = 0
    cost_est = int(prompt_tokens or 0) + int(out_est)

    # forward to backend with our request id and the estimate
    try:
        assert client is not None
        r = await client.post(
            f"{BACKEND_URL}/infer",
            json=payload,
            headers={
                "x-request-id": rid,
                "x-cost-est": str(cost_est),   # <---- new header
                "x-out-est": str(out_est),     # optional: for debugging only
            },
        )
    except Exception as e:
        LAT.observe(time.perf_counter() - t0)
        BACKEND_ERR.inc()
        REQS.labels("502").inc()
        return JSONResponse({"error": f"backend error: {e}"}, status_code=502)

    LAT.observe(time.perf_counter() - t0)
    REQS.labels(str(r.status_code)).inc()

    # pass body through as JSON if possible, else as text
    try:
        data = r.json()
    except Exception:
        return PlainTextResponse(r.text, status_code=r.status_code)

    # append visibility fields without changing backend payload keys
    if isinstance(data, dict):
        data.setdefault("gateway_request_id", rid)
        data.setdefault("cost_est", cost_est)
        data.setdefault("out_est", out_est)

    return JSONResponse(data, status_code=r.status_code)