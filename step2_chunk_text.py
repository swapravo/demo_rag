with open("knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

chunk_size = 150

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\nChunk {i}\n")
    print(chunk)

