import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DB_PATH = os.path.join(BASE_DIR, '..', 'chroma_db')

LLM_MODEL_NAME = "gpt-4o-mini"
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large-instruct"
RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
