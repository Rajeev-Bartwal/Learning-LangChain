from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate , MessagesPlaceholder
from langchain_core.messages import HumanMessage

# chat Template

chat_template = ChatPromptTemplate([
  ('system' , "you are a helpful customer support agent"),
  MessagesPlaceholder(variable_name='chat_history'),
  ('human' , "{query}")
]) 

chat_history = []
# load chat history
with open('chathistory.txt' , mode='r') as file:
  chat_history.extend(file.readlines())

print(chat_history)

#  create prompt
prompt = chat_template.invoke({'chat_history' : chat_history , 'query' : "Where is my refund"})

print(prompt)