import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from config import FAISS_INDEX_PATH, FAISS_METADATA_PATH, EMBEDDING_MODEL
from logger import log_event

class EmbeddingHandler:
    def __init__(self):
        """Initialize embedding model and FAISS index."""
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.index = faiss.IndexFlatL2(self.model.get_sentence_embedding_dimension())
        self.metadata = []

        # Load existing FAISS index if available
        if os.path.exists(FAISS_INDEX_PATH):
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_METADATA_PATH, "rb") as f:
                self.metadata = pickle.load(f)
            log_event("FAISS index loaded successfully.")

    def encode_text(self, text):
        """Convert text to embeddings."""
        return self.model.encode(text)

    def add_to_index(self, text_chunks):
        """Add new text chunks to FAISS index."""
        embeddings = [self.encode_text(chunk) for chunk in text_chunks]
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.metadata.extend(text_chunks)

        # Save updated index and metadata
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(FAISS_METADATA_PATH, "wb") as f:
            pickle.dump(self.metadata, f)
        log_event("New embeddings added to FAISS.")

if __name__ == "__main__":
    embedder = EmbeddingHandler()
    embedder.add_to_index(["Sample financial data"])
