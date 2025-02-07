from llm.llm_interface import generate_response
from utils.logger import log_event

if __name__ == "__main__":
    log_event("Chatbot started.")
    while True:
        query = input("Ask a question: ")
        if query.lower() in ["exit", "quit"]:
            log_event("Chatbot shutting down.")
            break
        response = generate_response(query)
        print("Response:\n", response)
