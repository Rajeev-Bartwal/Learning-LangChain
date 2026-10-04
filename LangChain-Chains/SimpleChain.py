from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field 
from typing import List
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

class Fact(BaseModel):
  fact_no : int = Field(gt=0 , description='This is the serial no of fact')
  fact_desc : str = Field(description='This is the description of the fact')

class List_Of_Facts(BaseModel):
  facts : List[Fact]


llm = HuggingFaceEndpoint(
  model='XiaomiMiMo/MiMo-V2.6-Pro-RL',
  huggingfacehub_api_token=os.getenv('HF_API'),
  temperature=0.5
)

model = ChatHuggingFace(llm=llm)

parser = PydanticOutputParser(pydantic_object=List_Of_Facts)

prompt = PromptTemplate(
  template='Generate 5 Intresting Facts about {topic} {format_instruction}',
  input_variables=['topic'],
  partial_variables={'format_instruction' : parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({'topic' : 'Biology'})

print(result.model_dump_json(indent=2))

chain.get_graph().print_ascii()