from langchain_text_splitters import RecursiveCharacterTextSplitter


text = """
Artificial Intelligence is transforming the way businesses operate across different industries. Companies are using AI to automate repetitive tasks, analyze large amounts of data, improve customer experiences, and make better decisions. AI-powered chatbots are becoming increasingly popular because they can provide instant responses to customer questions and support users 24 hours a day.

Large Language Models are a major part of modern AI applications. These models can understand and generate human language, summarize documents, answer questions, write code, and perform many other tasks. Frameworks such as LangChain make it easier for developers to build applications around these models.

When building a Retrieval Augmented Generation application, documents are usually divided into smaller pieces called chunks. These chunks are then converted into embeddings and stored in a vector database. When a user asks a question, the application searches for the most relevant chunks and provides them to the language model as context.

Choosing the right chunk size is important. If chunks are too small, important context may be lost. If chunks are too large, retrieval may become less precise and the language model may receive unnecessary information. A chunk overlap is often used so that information at the boundary between two chunks is not completely lost.

For example, a website chatbot can crawl a company's website, extract its text, split the content into chunks, generate embeddings, and store those embeddings in a vector database such as PostgreSQL with the pgvector extension. Later, when a user asks a question about the company, the application retrieves the most relevant information and generates an answer using an LLM.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print(len(chunks))
print(chunks) 