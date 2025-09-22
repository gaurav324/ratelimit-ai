import math

def est_tokens_out(model: str, prompt_tokens: int) -> int:
    """
    Estimate the number of output tokens for a GenAI request.

    Why we need this:
    - Gateway must decide on admit/deny *before* backend generates the response.
    - Output tokens often dominate cost, so ignoring them skews fairness.
    - We use a simple heuristic: base prior per model, scaled by prompt length.

    Args:
        model (str): model class, one of ["small", "medium", "large"]
        prompt_tokens (int): number of input tokens in the request

    Returns:
        int: estimated output tokens
    """

    # --- prior means for each model (roughly based on lognormal parameters in backend) ---
    priors = {
        "small": 12,     # average ~12 output tokens
        "medium": 28,    # average ~28 output tokens
        "large": 180     # average ~180 output tokens
    }

    # look up base prior for the given model, default to medium if unknown
    base = priors.get(model, priors["medium"])

    # scale factor: longer prompts tend to produce longer completions
    # prompt_tokens/1024 means:
    #   - 1024 input tokens → ~2x longer output than base
    #   - 2048 input tokens → ~3x longer output (capped below)
    mult = 1.0 + max(0, int(prompt_tokens or 0)) / 1024.0

    # cap multiplier to avoid extreme overestimates (e.g., max 3x base)
    mult = min(mult, 3.0)

    # final estimate
    est = int(base * mult)

    # always at least 1 token
    return max(1, est)

