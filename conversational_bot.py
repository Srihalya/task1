from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

memory = ConversationBufferMemory()

chatbot = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=True
)

print("\n🤖 Conversational Knowledge Bot (type 'exit' to quit)\n")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Bot: Goodbye 👋")
        break

    response = chatbot.run(user_input)
    print("Bot:", response)
