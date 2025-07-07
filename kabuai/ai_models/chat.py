import os

from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

CHAT_MODEL = os.getenv("CHAT_MODEL") or ""
CHAT_MODEL_LIGHT = os.getenv("CHAT_MODEL_LIGHT") or ""
CHAT_MODEL_HEAVY = os.getenv("CHAT_MODEL_HEAVY") or ""

TEMPERATURE = float(os.getenv("TEMPERATURE") or 0)


rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=0.1,
    max_bucket_size=10,
)

chat_model_light = init_chat_model(
    model=CHAT_MODEL_LIGHT,
    temperature=TEMPERATURE,
)
chat_model = init_chat_model(
    model=CHAT_MODEL,
    temperature=TEMPERATURE,
    max_tokens=2048,
)
chat_model_heavy = init_chat_model(
    model=CHAT_MODEL_HEAVY,
    temperature=TEMPERATURE,
    max_tokens=4096,
)
