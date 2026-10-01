from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel , Field



## JSONOutPutparser


load_dotenv()

llm = HuggingFaceEndpoint(
      model="meta-llama/Llama-3.1-8B-Instruct",
      huggingfacehub_api_token=os.getenv("HF_API"),
      temperature=0.4
    )

model = ChatHuggingFace(llm=llm)


parser = JsonOutputParser()


template = PromptTemplate(
  template= ' Give me the name , age and city of a fictional person \n {format_instruction}',
  input_variables=[],
  partial_variables={'format_instruction' : parser.get_format_instructions()}
)

prompt = template.format()

result = model.invoke(prompt)

person = parser.parse(result.content)

print(person)
print(type(person))