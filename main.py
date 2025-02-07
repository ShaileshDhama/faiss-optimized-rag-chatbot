from llm.chatbot import FinanceChatbot

def main():
    print("💡 Welcome to the AI Finance Chatbot! Ask your question:")
    chatbot = FinanceChatbot()
    while True:
        user_query = input("❓ Your Question: ")
        if user_query.lower() in ["exit", "quit"]:
            print("👋 Exiting chatbot.")
            break
        response = chatbot.generate_response(user_query)
        print(f"🤖 AI Response: {response}")

if __name__ == "__main__":
    main()
