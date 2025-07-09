import os

from langchain_core.language_models import BaseLLM
from langchain_google_genai.llms import GoogleGenerativeAI
from langchain_ollama.llms import OllamaLLM

LLM_MODEL = os.getenv("LLM_MODEL") or ""
LLM_MODEL_LIGHT = os.getenv("LLM_MODEL_LIGHT") or ""
LLM_MODEL_HEAVY = os.getenv("LLM_MODEL_HEAVY") or ""

TEMPERATURE = float(os.getenv("TEMPERATURE") or 0)

llm_light: BaseLLM
llm: BaseLLM
llm_heavy: BaseLLM

if LLM_MODEL_LIGHT.startswith("google_genai"):
    llm_light = GoogleGenerativeAI(
        model=LLM_MODEL_LIGHT.split(":")[1],
        temperature=TEMPERATURE,
    )
elif LLM_MODEL_LIGHT.startswith("ollama"):
    llm_light = OllamaLLM(
        model=LLM_MODEL_LIGHT.split(":")[1],
        temperature=TEMPERATURE,
    )
else:
    raise ValueError(f"Unsupported LLM model: {LLM_MODEL_LIGHT}")

if LLM_MODEL_HEAVY.startswith("google_genai"):
    llm_heavy = GoogleGenerativeAI(
        model=LLM_MODEL_HEAVY.split(":")[1],
        temperature=TEMPERATURE,
        max_tokens=8192,
    )
elif LLM_MODEL_HEAVY.startswith("ollama"):
    llm_heavy = OllamaLLM(
        model=LLM_MODEL_HEAVY.split(":")[1],
        temperature=TEMPERATURE,
    )
else:
    raise ValueError(f"Unsupported LLM model: {LLM_MODEL_HEAVY}")

if LLM_MODEL.startswith("google_genai"):
    llm = GoogleGenerativeAI(
        model=LLM_MODEL.split(":")[1],
        temperature=TEMPERATURE,
        max_tokens=4096,
    )
elif LLM_MODEL.startswith("ollama"):
    llm = OllamaLLM(
        model=LLM_MODEL.split(":")[1],
        temperature=TEMPERATURE,
    )
else:
    raise ValueError(f"Unsupported LLM model: {LLM_MODEL}")
