from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

loader = PyPDFLoader('When Does Crypto Tax Apply in India.pdf')

docs = loader.load()

print("\n--- First Page ---")
print(docs[0].page_content)

print("\n--- Metadata ---")
print(docs[0].metadata)