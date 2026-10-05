import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import DirectoryLoader , PyPDFLoader

loader = DirectoryLoader(
  path = 'books',
  glob= '*.pdf',
  loader_cls=PyPDFLoader
)

docs = loader.load()

print(len(docs))

# 1st pdf content
print(docs[0].page_content)
print(docs[3].metadata)