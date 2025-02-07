import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config import DATA_DIR, PICKLE_FILE, INDEX_FILE, EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, NUM_WORKERS
from utils.logger import log_event
from multiprocessing import Pool

# Initialize the embedding model
model = SentenceTransformer(EMBEDDING_MODEL)

def load_text(file_path):
    """Load text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def split_text(text):
    """Split text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def process_file(file):
    """Process individual file for parallel embedding."""
    text = load_text(os.path.join(DATA_DIR, file))
    return split_text(text)

def create_embeddings():
    """Generate and save embeddings for the knowledge base using multiprocessing."""
    log_event("Creating embeddings...")
    all_chunks = []
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".txt")]
    
    with Pool(NUM_WORKERS) as pool:
        results = pool.map(process_file, files)
        for chunks in results:
            all_chunks.extend(chunks)
    
    embeddings = model.encode(all_chunks, convert_to_numpy=True)
    
    # Store FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, INDEX_FILE)
    
    with open(PICKLE_FILE, "wb") as f:
        pickle.dump((all_chunks, embeddings), f)
    
    log_event("Embeddings created successfully!")
    return all_chunks, index
