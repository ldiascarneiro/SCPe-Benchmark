# Created by Leandro Carneiro at 13/09/2025
# Description: 
# -------------------------------------------------------------


# config.py

# --------- Caminhos ----------
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "dados" / "input"
OUTPUT_DIR = BASE_DIR / "dados" / "output"
DATASET_CSV_PATH = INPUT_DIR / "SCPe_Dataset.csv"
CSV_DELIMITER = "#"
DATASET_ROW_INDEX = 0
LOCK_TO_FIRST_USEFUL_ROW = True
DATASET_START_ROW = 0
DATASET_NUM_ROWS_TO_PROCESS = -1   # -1 para todas as linhas do dataset
SAVE_DEBUG_ARTIFACTS = False

# --------- Modelo ----------
# "openai", "local_vllm" ou "ollama"
LLM_PROVIDER = "ollama"

OPENAI_API_KEY = ""

# Modelo quando for Ollama
LLM_MODEL_NUM_CTX = 16384
CHAT_MODEL_URL = "http://10.93.48.97:11436"
MAX_TOKENS = 3500
MAX_TOKENS_RETRY = 6000
OLLAMA_THINK = False
OLLAMA_STREAM = False
LLM_INVOKE_MAX_RETRIES = 2
LLM_INVOKE_RETRY_BACKOFF_SECONDS = 2.0
OLLAMA_PULL_MAX_RETRIES = 3
OLLAMA_PULL_RETRY_BACKOFF_SECONDS = 4.0
SECTION_MIN_CONTENT_WORDS = 80
SECTION_COMPLETION_SATURATION_RATIO = 0.97

# --------- Benchmark de modelos ----------
# Adicione novos modelos nesta lista para comparar no mesmo pipeline.
BENCHMARK_CONTINUE_ON_ERROR = True
BENCHMARK_MODELS = [
#    "deepseek-r1:8b",
#    "lfm2.5-thinking:1.2b",
#    "falcon3:10b",
#    "qwen3.5:9b",
#    "phi4:14b",

#    "qwen3:32b",
#    "deepseek-r1:32b",
#    "qwen3.5:27b",
#    "qwen3.5:35b",

    "deepseek-r1:70b",
    "llama3.3:70b",
    "qwen3.5:122b",
]

# --------- Métricas de geração ----------
GENERATION_MODEL_PARAMS_B = 32.0

# --------- Métricas de qualidade (G-Eval) ----------
GEVAL_MODEL_NAME = "gpt-4o-mini"
GEVAL_TEMPERATURE = 0.0
GEVAL_MAX_OUTPUT_TOKENS = 600
ADHERENCE_EMPTY_SECTION_PENALTY_PER_SECTION = 1.0
ADHERENCE_OMISSION_MAX_PENALTY = 4.0
