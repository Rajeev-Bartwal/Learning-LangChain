import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import  WebBaseLoader
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = 'https://www.rtings.com/laptop/reviews/best/laptop'
loader = WebBaseLoader(url)

docs = loader.load()

# print(docs[0].page_content)

parser = StrOutputParser()

model = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)


prompt = PromptTemplate(
    template="""Answer the following question \n 
    {question}
    from the following text - \n 
    {text}
    """,
    input_variables=['question' , "text"]
)

chain = prompt | model | parser

result = chain.invoke({'question': "Best Laptop in the best budget 90000" , 'text' : docs[0].page_content})

print(result)
