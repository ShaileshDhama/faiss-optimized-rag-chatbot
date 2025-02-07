import ollama
from retrieval.retriever import FinanceRetriever
from config import LLM_MODEL
from logger import log_event

class LLMInterface:
    def __init__(self):
        """Initialize retriever for fetching relevant finance data."""
        self.retriever = FinanceRetriever()

    def generate_response(self, user_query):
        """Retrieve relevant knowledge and generate an AI response."""
        retrieved_chunks = self.retriever.retrieve(user_query)
        context = " ".join([chunk[0] for chunk in retrieved_chunks])

        prompt = f"""
        You are an AI finance assistant. Answer based on the provided knowledge only.
        Query: {user_query}
        Context: {context}
        """

        response = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
        log_event(f"Generated AI response for query: {user_query}")

        return response['message']['content']

if __name__ == "__main__":
    llm = LLMInterface()
    print(llm.generate_response("Explain Basel III regulations."))
