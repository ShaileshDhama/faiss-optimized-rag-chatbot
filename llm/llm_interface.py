import ollama
from retrieval.retrieval import retrieve_relevant_chunks
from config import LLM_MODEL
from utils.logger import log_event
from utils.caching import get_cached_response, cache_response

def generate_response(query):
    """Generate response using LLM with retrieved context and caching."""
    log_event(f"Generating response for query: {query}")
    cached = get_cached_response(query)
    if cached:
        log_event("Using cached response.")
        return cached
    
    relevant_chunks = retrieve_relevant_chunks(query)
    context = "\n".join(relevant_chunks)
    
    prompt = f"""
    Answer the following query using only the provided context.
    
    Context: {context}
    
    Query: {query}
    """
    
    response = ollama.chat(model=LLM_MODEL, messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ])
    response_text = response["message"]["content"]
    cache_response(query, response_text)
    log_event("Response generated successfully.")
    return response_text
