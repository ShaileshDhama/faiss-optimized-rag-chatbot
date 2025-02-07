import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def create_faiss_index():
    """Loads finance knowledge base, creates embeddings, and saves FAISS index."""
    knowledge_files = [
        "knowledge_base/01_market_analysis.txt",
        "knowledge_base/02_algorithmic_trading.txt",
        "knowledge_base/03_quant_finance.txt",
        "knowledge_base/04_macro_economics.txt",
        "knowledge_base/05_risk_management.txt"
    ]

    chunks = []
    for file in knowledge_files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()
            chunks.extend(text.split("\n\n"))  # Split into chunks
    
    embeddings = np.array([model.encode(chunk) for chunk in chunks])

    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 Distance (Euclidean)
    index.add(embeddings)

    with open("embeddings/finance_embeddings.pkl", "wb") as f:
        pickle.dump((chunks, index), f)

    print("✅ FAISS Index Created & Saved")

def load_embeddings():
    """Loads FAISS index for retrieval."""
    with open("embeddings/finance_embeddings.pkl", "rb") as f:
        return pickle.load(f)
