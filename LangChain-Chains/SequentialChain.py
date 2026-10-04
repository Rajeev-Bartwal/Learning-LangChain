from langchain_huggingface import HuggingFaceEndpoint , ChatHuggingFace
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatHuggingFace(
  llm = HuggingFaceEndpoint(
      model='XiaomiMiMo/MiMo-V2.6-Pro-RL',
      huggingfacehub_api_token=os.getenv('HF_API'),
      temperature=0.5
  )
)

prompt1 = PromptTemplate(
    template='Give me a detailed report on {Topic}',
    input_variables=['Topic']
)

prompt2 = PromptTemplate(
    template='Give me 5 importent Points from the following report\n {report}',
    input_variables=['report']
)

pareser = StrOutputParser()

chain = prompt1 | model | pareser | prompt2 | model | pareser

result = chain.invoke('from where i can find AIIMS CRE PYQ of 5 years')

print(result)

chain.get_graph().print_ascii()
