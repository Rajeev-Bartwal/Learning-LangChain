from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel , Field



## StructuredOutputParser ye bhut outdated ho gya h  
# Now we can use PydanticOutPut Parser.


load_dotenv()

llm = HuggingFaceEndpoint(
      model="meta-llama/Llama-3.1-8B-Instruct",
      huggingfacehub_api_token=os.getenv("HF_API"),
      temperature=0.4
    )

model = ChatHuggingFace(llm=llm)