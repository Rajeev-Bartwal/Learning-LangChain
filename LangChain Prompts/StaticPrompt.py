from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv 
import os

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3.8-27B",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
    temperature=0.7
)

model = ChatHuggingFace(llm=llm)

user_input = input('Enter your query\n')

result = model.invoke(user_input)

print(result.content)