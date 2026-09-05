import os
import pickle

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

index = faiss.read_index("knowledge.index")

with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

question = input("Question: ")

response = client.embeddings.create(
    model="text-embedding-3-small",
    input=question
)

query = np.array(
    [response.data[0].embedding],
    dtype="float32"
)

distances, indices = index.search(query, 3)

print("\nRetrieved Chunks\n")

for i in indices[0]:
    print("-" * 50)
    print(chunks[i])

