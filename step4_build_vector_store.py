import os
import pickle

import faiss
import numpy as np
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

embeddings = np.array(
    [item.embedding for item in response.data],
    dtype="float32"
)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

faiss.write_index(index, "knowledge.index")

with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("Vector database created.")
print("Vectors stored:", index.ntotal)
