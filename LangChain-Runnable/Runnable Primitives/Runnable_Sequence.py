from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
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

template = PromptTemplate(
  template='You are a very Good Comedian now tell me a Joke about {topic}',
  input_variables=['topic']
)

prompt2 = PromptTemplate(
  template='Explain me the following joke in one paragraph\n {text}',
  input_variables=['text']
)

# Both are same
chain = RunnableSequence(template , model , parser , prompt2 , model , parser)

# chain = template | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Cricket'})

print(result)