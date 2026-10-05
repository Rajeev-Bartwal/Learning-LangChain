from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

# 1. Load document
loader = TextLoader("Cricket.txt")
docs = loader.load()

print("Number of documents:", len(docs))
print("Metadata:", docs[0].metadata)


model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


prompt = PromptTemplate(
    template="""Give me a 300-word summary of the following text:
{text}
""",
    input_variables=["text"]
)


parser = StrOutputParser()


chain = prompt | model | parser


result = chain.invoke({
    "text": docs[0].page_content
})

print(result)