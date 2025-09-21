from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse, PlainTextResponse
import random, math, time

# Prometheus
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI(title="ratelimit-ai backend v0.1 (py3.9 + metrics)")

# ----- Metrics -----
REQS = Counter(
    "backend_requests_total",
    "Total backend requests",
    ["status"]  # "200" or "500"
)
TOKENS = Counter(
    "backend_tokens_total",
    "Total output tokens (sum)",
    ["model"]   # small/medium/large
)
LAT = Histogram(
    "backend_latency_seconds",
    "Latency of /infer handler (seconds)",
    buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
)

# ----- Token sampler (same logic; heavy-tailed) -----
def sample_tokens_out(model: str = "small", prompt_tokens: Optional[int] = None) -> int:
    params = {"small": (12.0, 0.9), "medium": (28.0, 1.0), "large": (180.0, 1.2)}
    mu, sigma = params.get(model, params["small"])
    base = random.lognormvariate(math.log(mu), sigma)

    # bias by input tokens (linear-ish)
    if prompt_tokens:
        # more input tokens → longer outputs, but cap the multiplier
        multiplier = min(3.0, 1.0 + (prompt_tokens / 1024.0))
        base *= multiplier

    return max(1, int(base))


@app.get("/healthz")
async def healthz():
    return PlainTextResponse("ok")

@app.get("/metrics")
async def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/infer")
async def infer(payload: dict):
    t0 = time.perf_counter()
    try:
        tenant = payload.get("tenant", "T0001")
        model = payload.get("model", "small")
        prompt_tokens = payload.get("prompt_tokens")

        tokens_out = sample_tokens_out(model, prompt_tokens)

        # record metrics
        LAT.observe(time.perf_counter() - t0)
        TOKENS.labels(model).inc(tokens_out)
        REQS.labels("200").inc()

        return JSONResponse({
            "tenant": tenant,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "tokens_out": tokens_out
        })
    except Exception as e:
        LAT.observe(time.perf_counter() - t0)
        TOKENS.labels(model).inc(0)
        REQS.labels("500").inc()
        return JSONResponse({"error": str(e)}, status_code=500)
