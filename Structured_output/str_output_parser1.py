from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import StrOutputParser


## With strOutPutparser Mainly used with Chains.


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

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({'topic':'Black hole'})

print(result)