import sys
import os

# Get the root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add src directory to Python path
sys.path.append(os.path.join(ROOT_DIR, 'src'))

from retriever import generate_embedding  # Now it should work

query_ar = "ما هو الذكاء الاصطناعي؟"
embedding_ar, _ = generate_embedding(query_ar, "arabic")
print(f"Arabic Embedding Size: {len(embedding_ar)}")  # Should be 1024

query_en = "What is artificial intelligence?"
embedding_en, _ = generate_embedding(query_en, "english")
print(f"English Embedding Size: {len(embedding_en)}")  # Should be 1024




import ollama
query = "ما هي فوائد الذكاء الاصطناعي؟"
embedding = ollama.embeddings(model="jaluma/arabert-all-nli-triplet-matryoshka", prompt=query)
print(len(embedding["embedding"]))  # Should print 768 or 1024

