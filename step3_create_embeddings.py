import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunk_size = 150

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=chunks
)

embeddings = [item.embedding for item in response.data]

print("Embedding dimension:", len(embeddings[0]))
print("Number of embeddings:", len(embeddings))
print("First 10 values of first embedding:")
print(embeddings[0][:10])
