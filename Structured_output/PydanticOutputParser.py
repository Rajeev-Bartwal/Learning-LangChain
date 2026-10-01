from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser , PydanticOutputParser
from pydantic import BaseModel , Field

# Now we can use PydanticOutPut Parser.

load_dotenv()

llm = HuggingFaceEndpoint(
      model="meta-llama/Llama-3.1-8B-Instruct",
      huggingfacehub_api_token=os.getenv("HF_API"),
      temperature=0.4
    )

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
  name:str = Field(description='name of the person')
  age:int = Field(gt=18 , description='age of the person')
  city:str = Field(description='city name where the person belogs to')

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
  template='Give me a name, age and city name of a fictional {place} person \n {format_instruction}',
  input_variables=['place'],
  partial_variables={'format_instruction':parser.get_format_instructions()}
)

# prompt = template.invoke({'place' : 'Indian'})

# print(prompt)

# result = model.invoke(prompt)

# final_result = parser.parse(result.content)

chain = template | model | parser

final_result = chain.invoke({'place' : 'indian'})

print(final_result)

