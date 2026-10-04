from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel , RunnableSequence
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

prompt1 = PromptTemplate(
  template='Generate a Tweet about the topic {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Generate a LinkedIn Post about the topic {topic}',
  input_variables=['topic']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
  {
    'tweet' : RunnableSequence(prompt1 , model , parser),
    'LinkedIn_post' : RunnableSequence(prompt2 , model , parser)
  }
)

result = parallel_chain.invoke({'topic' : 'Ai'})

print(result)