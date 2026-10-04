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