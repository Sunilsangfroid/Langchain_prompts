from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4', temperature=0.5)

result = model.invoke("Give me a summary of the book 'The Great Gatsby'")

print(result.content)