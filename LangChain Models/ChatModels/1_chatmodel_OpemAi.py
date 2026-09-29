from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()

# Temprature parameter which is used for creativity means how creative response you want from your model.
client = ChatOpenAI(
  base_url='https://api.groq.com/openai/v1' ,
  api_key=os.getenv('GROQ_API_KEY'),
  model="openai/gpt-oss-120b",
  temperature=2
  )

result = client.invoke("Who is mahatma gandhi")
print(result.content)