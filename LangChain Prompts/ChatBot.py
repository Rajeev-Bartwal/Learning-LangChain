from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
    temperature=1.0
)

model = ChatHuggingFace(llm=llm)

chat_history = [
  SystemMessage(content='You are a very Helpful Ai Assistent')
]
while True:
  user_input = input('You : ')
  chat_history.append(HumanMessage(content=user_input))
  if user_input == 'exit':
    break
  else :
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.content))
    print('Ai:',  result.content)

print(chat_history)