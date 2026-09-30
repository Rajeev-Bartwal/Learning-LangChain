from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
    temperature=1.0
)

model = ChatHuggingFace(llm=llm)

messages = [
  SystemMessage(content='You are a helpful assistent'),
  HumanMessage(content='Tell me about LangChain')
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)