import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "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")
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.1-70b-instruct")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nvidia/nv-embedqa-e5-v5")  # NVIDIA embedding model
    CHROMA_DB_PATH = "./chroma_db"
    SIMILARITY_THRESHOLD = 0.7
    REPETITION_THRESHOLD = 0.70  # Lower threshold for text-based similarity

