import ollama

embedding = ollama.embeddings(model="jaluma/arabert-all-nli-triplet-matryoshka", prompt="ما هو الذكاء الاصطناعي؟")
print(f"Arabic Embedding Size: {len(embedding['embedding'])}")