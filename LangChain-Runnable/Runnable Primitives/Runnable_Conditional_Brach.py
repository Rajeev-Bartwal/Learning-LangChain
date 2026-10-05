from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough , RunnableBranch , RunnableLambda , RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()


parser = StrOutputParser()

model = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        model="Qwen/Qwen3-4B-Instruct-2507",
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
        temperature=0
    )
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a detailed topic on \n {topic}',
    input_variables=['topic']
)

prompt1 = PromptTemplate(
  template='Summarize the following text {text}',
  input_variables=['text']
)

report_gen_chain = prompt | model | parser

branch_chain = RunnableBranch(
  (lambda x: len(x.split()) > 200 , prompt1 | model | parser),
  RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain , branch_chain)

result = final_chain.invoke({'topic' : 'War Between iran and usa'})

print(result)