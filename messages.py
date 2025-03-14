from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()


google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables")

model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', api_key=google_api_key)

messages = [
    SystemMessage(content = "You are a helpful assistant"),
    HumanMessage(content = "Tell me about Langchain")
]

result = model.invoke(messages)

messages.append(AIMessage(content =result.content))

print(messages)