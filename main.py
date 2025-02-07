from llm_interface import LLMInterface
from logger import log_event

def main():
    print("💡 Welcome to the AI Finance Chatbot! Ask your question:")
    chatbot = LLMInterface()
    while True:
        user_query = input("❓ Your Question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot.")
            log_event("Chatbot session ended.")
            break
        response = chatbot.generate_response(user_query)
        print(f"🤖 AI Response: {response}")

if __name__ == "__main__":
    log_event("Chatbot started.")
    main()
