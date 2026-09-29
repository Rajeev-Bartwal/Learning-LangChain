from google.genai import types
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import os

load_dotenv()

players = [
    "Virat Kohli is an Indian international cricketer known for his aggressive batting, consistency, and leadership. He is regarded as one of the leading batters of his generation.",
    "MS Dhoni is a former Indian international cricketer and captain, known for his calm leadership, wicketkeeping, and finishing matches as a batter.",
    "Sachin Tendulkar is a former Indian international cricketer widely regarded as one of cricket's greatest batters. He played international cricket for India for over two decades.",
    "Rohit Sharma is an Indian international cricketer known for his elegant batting and ability to score large innings, especially in limited-overs cricket.",
    "Jasprit Bumrah is an Indian international fast bowler known for his unusual bowling action, pace, yorkers, and effectiveness in different match situations.",
]

query = 'Who is MS dhoni'

api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=api_key)

players_embeddings = client.models.embed_content(
    model="gemini-embedding-2",
    contents=[
        types.Content(
            parts=[types.Part(text=player)]
        )
        for player in players
    ],
)

query_embedding = client.models.embed_content(
        model="gemini-embedding-2",
        contents=query
)


player_vectors = [item.values for item in players_embeddings.embeddings]
query_vector = query_embedding.embeddings[0].values

scores = cosine_similarity([query_vector], player_vectors)[0]

ind , best_score = sorted(list(enumerate(scores)),key = lambda x:x[1])[-1]

print(query)
print(players[ind])
print(best_score)