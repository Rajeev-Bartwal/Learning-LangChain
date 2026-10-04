from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence , RunnableParallel , RunnablePassthrough
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()


# Example of passThrough for when we need both the output and the input as well

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

# Will generate chain
Joke_chain = RunnableSequence(template , model , parser)

# chain = template | model | parser 

paralle_chain = RunnableParallel({
  'Joke' :   RunnablePassthrough(),
  'Joke_explanation' : RunnableSequence(prompt2 | model | parser)
})

final_chain = RunnableSequence(Joke_chain , paralle_chain)
result = final_chain.invoke({'topic' : 'Unemployement'})

print(result)