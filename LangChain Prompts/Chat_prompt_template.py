from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from dotenv import load_dotenv
import os
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


chat_prompt_template = ChatPromptTemplate([
  ('system' ,"You are a helpful {domain} expert" ),
  ('human' ,"Explain in simple terms what is {topic}" )
])

prompt = chat_prompt_template.invoke({'domain':'cricket' , 'topic':'Dusra'})

print(prompt)

llm = HuggingFaceEndpoint(
      repo_id="Qwen/Qwen3.8-27B",
      huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY'),
      temperature=1.0
  )

model = ChatHuggingFace(llm=llm)