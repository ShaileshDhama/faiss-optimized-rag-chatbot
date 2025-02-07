import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DATA_DIR = "./knowledge_base/"
PICKLE_FILE = "./embeddings/embeddings_faiss.pkl"
INDEX_FILE = "./embeddings/faiss_index.idx"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
LLM_MODEL = "llama3.1"
LOG_FILE = "./logs/system_logs.txt"
CACHE_FILE = "./cache/query_cache.pkl"
NUM_WORKERS = 4  # Parallel Processing Workers
