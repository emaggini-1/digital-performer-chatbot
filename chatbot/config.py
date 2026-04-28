import os

INDEX_DIR = "chatbot/search_index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
TOP_K = 8
MAX_CONTEXT_TOKENS = 6000
