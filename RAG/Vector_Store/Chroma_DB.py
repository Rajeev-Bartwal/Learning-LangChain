from langchain_google_genai import ChatGoogleGenerativeAI , GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


loader = PyPDFLoader('ipl_vectorstore_practice_5_pages.pdf')

docs = loader.lazy_load()

# print(len(docs))
# print(docs[0])

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
)


splitter = RecursiveCharacterTextSplitter(
  chunk_size= 500,
  chunk_overlap = 50
)

chunks = splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)



vector_store = Chroma(
  collection_name='sample',
  embedding_function=embeddings,
  persist_directory="./chroma_db"
)

vector_store.add_documents(chunks)

# print(vector_store.get(include=['embeddings' , 'documents' , 'metadatas']))

print(vector_store.similarity_search(query='virat kohli plays for??' , k=2))