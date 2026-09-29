from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()

client = InferenceClient(
    api_key=os.getenv('HUGGINGFACE_API_KEY')
)

documents = [
    "India is a country in South Asia.",
    "The capital of India is New Delhi.",
    "Mumbai is the financial capital of India.",
    "Bengaluru is known as the Silicon Valley of India.",
    "Python is a popular programming language.",
    "Java is widely used for backend development."
]

embedding = client.feature_extraction(
    documents,
    model="Qwen/Qwen3-Embedding-0.6B"
)

print(len(embedding))
