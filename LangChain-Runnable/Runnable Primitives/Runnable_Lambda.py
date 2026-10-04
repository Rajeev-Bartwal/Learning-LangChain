from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableLambda , RunnableParallel , RunnablePassthrough
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
  template='Give me a joke on the \n {topic}',
  input_variables=['topic']
)

joke_gen_chain = prompt | model | parser

def count_words(joke):
  return len(joke.split())

parllel_chain = RunnableParallel({
  'joke' : RunnablePassthrough(),
  'word_count' : RunnableLambda(lambda x: len(x.split()))
})

final_chain = joke_gen_chain | parllel_chain 

result = final_chain.invoke({'topic' : 'Laptop'})

final_result = """{} \n word count :  {}""".format(result['joke'] , result['word_count'])

print(final_result)