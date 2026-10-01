from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os


## Without strOutPutparser


load_dotenv()

llm = HuggingFaceEndpoint(
      model="meta-llama/Llama-3.1-8B-Instruct",
      huggingfacehub_api_token=os.getenv("HF_API"),
      temperature=0.4
    )

model = ChatHuggingFace(llm=llm)

template1 = PromptTemplate(
  template='Write a detailed report on {topic}',
  input_variables=['topic']  
)

template2 = PromptTemplate(
  template="Give me the summary of this {text} in 5 lines",
  input_variables=['text']
)

prompt1 = template1.invoke({'topic' : 'black hole'})
result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text' : result1.content})
result2 = model.invoke(prompt2)

print(result1.content)

print('---------------------**********-----------------')
print(result2.content)