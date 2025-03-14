from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Define the chat template
chat_template = ChatPromptTemplate([
    ('system', 'You are a helpful customer support agent'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history=[]
# load the chat history
with open('chat_history.txt') as f:
    chat_history.extend(f.readlines())
    
print(chat_history)


print("\n")
# create prompt
prompt = chat_template.invoke({'chat_history':chat_history,'query':'What is the status of my order?'})

print(prompt)