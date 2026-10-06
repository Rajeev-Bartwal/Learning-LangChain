from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    api_key=os.getenv('GEMINI_API_KEY')
)


text = """
The Himalayan mountains are home to many unique species of plants and animals, and the region attracts thousands of tourists every year. The weather in high-altitude areas can change very quickly, so travelers should always carry warm clothes and check local conditions before starting a trek.

Python is widely used in software development because its syntax is simple and it has a large collection of libraries. Developers use Python for web development, automation, data analysis, machine learning, artificial intelligence, and many other applications.

The Indian economy has grown significantly over the last few decades, with industries such as information technology, manufacturing, agriculture, and telecommunications playing important roles. Small businesses and startups also contribute significantly to employment and economic development.

A healthy sleep schedule is important for maintaining physical and mental well-being. Most adults need several hours of quality sleep every night, and maintaining a consistent bedtime can improve concentration, mood, and overall productivity.

Football is one of the most popular sports in the world. Professional teams spend years developing their players through training, tactical planning, fitness programs, and competitive matches. Major football tournaments attract millions of viewers from different countries.

Cloud computing allows companies to run applications and store data on remote servers instead of depending entirely on local computers. Services such as databases, storage, computing power, and machine learning infrastructure can be accessed through the internet whenever they are required.

The process of photosynthesis allows green plants to convert sunlight into chemical energy. Plants use carbon dioxide and water to produce glucose and release oxygen as a by-product. This process is essential for most life on Earth because it forms the foundation of many food chains.

Electric vehicles are becoming increasingly common because they can reduce dependence on fossil fuels and may produce fewer emissions during operation. Improvements in battery technology have increased driving ranges and reduced charging times, although charging infrastructure remains an important challenge.

Artificial intelligence is being used in customer support, healthcare, finance, education, and software development. Large language models can understand and generate human language, while computer vision systems can analyze images and videos. These technologies are changing how people interact with computers.

The Taj Mahal is one of India's most famous historical monuments and attracts visitors from around the world. It was built using white marble and is widely recognized for its architecture, symmetry, gardens, and historical significance.
"""


splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="standard_deviation",
    breakpoint_threshold_amount=1
)

chunks = splitter.create_documents([text])

for chunk in chunks:
    print(chunk.page_content)
    print("-" * 50)