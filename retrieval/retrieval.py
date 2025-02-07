import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings.embedding import load_embeddings
from config import EMBEDDING_MODEL

# Load embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def retrieve_relevant_chunks(query, top_k=5):
    """Retrieve top relevant finance-related knowledge chunks."""
    all_chunks, index = load_embeddings()
    query_embedding = model.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding, top_k)
    results = [all_chunks[i] for i in indices[0]]
    return results
