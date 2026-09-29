## LLm code is diff you need api key for that it is chatmodel 

from groq import Groq
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import os



load_dotenv()

client = ChatOpenAI(
  base_url='https://api.groq.com/openai/v1' ,
  api_key=os.getenv('GROQ_API_KEY'),
  model="openai/gpt-oss-120b"
  )

result = client.invoke('hello')


print(result.content)