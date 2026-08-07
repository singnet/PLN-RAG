import dspy

from config import get_settings


def create_lm() -> dspy.LM:
    """Create the configured OpenAI-compatible DSPy language model."""
    cfg = get_settings()
    kwargs = {
        "api_key": cfg.openai_api_key,
        "cache": False,
    }
    if cfg.openai_base_url:
        kwargs["base_url"] = cfg.openai_base_url
    return dspy.LM(cfg.openai_model, **kwargs)
