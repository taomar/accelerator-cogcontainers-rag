import os
import uuid
import json
import csv
import textwrap
import requests
import ollama
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, VectorParams

# ================================
# 🔹 CONFIGURATION & INITIAL SETUP
# ================================

# Load API details from environment variables (.env)
AZURE_LANGUAGE_API_URL = os.getenv("AZURE_LANGUAGE_API_URL", "http://localhost:5000/text/analytics/v3.1/languages")

# Initialize Qdrant client (local instance)
client = QdrantClient("localhost", port=6333)

# Define embedding sizes based on model
EMBEDDING_SIZE = 1024  # Both English & Arabic use bge-m3 (same dimension)

# ================================
# 🔹 QDRANT COLLECTION SETUP
# ================================

for lang in ["en", "ar"]:
    collection_name = f"rag_docs_{lang}"

    if client.collection_exists(collection_name):
        existing_collection_info = client.get_collection(collection_name)

        # Ensure existing collection has the correct vector size
        existing_vector_size = existing_collection_info.config.params.vectors.size
        if existing_vector_size != EMBEDDING_SIZE:
            print(f"🚨 Collection `{collection_name}` has incorrect vector size ({existing_vector_size}). Deleting & recreating...")
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        print(f"🚀 Creating collection `{collection_name}` with vector size {EMBEDDING_SIZE}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance="Cosine"),
        )

print("✅ Qdrant collections are now correctly set up!")

# ================================
# 🔹 LANGUAGE DETECTION FUNCTION
# ================================

def detect_language(text):
    """Detects language using Azure AI Language API with robust error handling."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers, timeout=10)
        response.raise_for_status()  # Ensure API errors are caught
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "ar" if detected_lang == "ar" else "en"

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Language detection API error: {e} → Defaulting to English")
    except ValueError:
        print(f"⚠️ Invalid JSON response from API → Defaulting to English")

    return "en"  # Fallback to English if detection fails

# ================================
# 🔹 EMBEDDING GENERATION FUNCTION
# ================================

def generate_embedding(text):
    """Generates embeddings using bge-m3 for both Arabic & English."""
    response = ollama.embeddings(model="bge-m3", prompt=text)
    embedding = response["embedding"]

    # Ensure embedding size matches Qdrant vector size
    if len(embedding) < EMBEDDING_SIZE:
        embedding = np.pad(embedding, (0, EMBEDDING_SIZE - len(embedding)), 'constant', constant_values=0)
    elif len(embedding) > EMBEDDING_SIZE:
        embedding = embedding[:EMBEDDING_SIZE]  # Truncate if too large

    return embedding

# ================================
# 🔹 DOCUMENT INDEXING FUNCTION
# ================================

def index_document(text, lang):
    """Indexes a document into Qdrant after generating embeddings."""
    embedding = generate_embedding(text)
    if embedding is None:
        print(f"⚠️ Skipping indexing due to embedding error for text: {text[:30]}...")
        return

    collection_name = f"rag_docs_{lang}"
    if not client.collection_exists(collection_name):
        print(f"⚠️ Collection `{collection_name}` not found! Skipping indexing...")
        return

    point = {
        "id": str(uuid.uuid4()),  # Unique UUID for each chunk
        "vector": embedding,
        "payload": {"text": text, "language": lang}
    }

    client.upsert(collection_name=collection_name, points=[point])

# ================================
# 🔹 TEXT CHUNKING FUNCTION
# ================================

def chunk_text(text, chunk_size=200):
    """Splits text into smaller chunks for better retrieval performance."""
    return textwrap.wrap(text, chunk_size) if text.strip() else []

# ================================
# 🔹 DOCUMENT LOADING FUNCTION
# ================================

def load_documents():
    """Loads text, JSON, and CSV documents from the `data/` folder."""
    documents = []

    for file in os.listdir("data"):
        file_path = os.path.join("data", file)

        # Load text files
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        lang = detect_language(line.strip())
                        documents.append({"text": line.strip(), "lang": lang})

        # Load JSON files
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for doc in json_data:
                    if "text" in doc and doc["text"].strip():
                        lang = detect_language(doc["text"])
                        documents.append({"text": doc["text"], "lang": lang})

        # Load CSV files (assuming columns: text, lang)
        elif file.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        lang = detect_language(row["text"])
                        documents.append({"text": row["text"], "lang": lang})

    return documents

# ================================
# 🔹 INDEXING DOCUMENTS INTO QDRANT
# ================================

documents = load_documents()

if not documents:
    print("⚠️ No documents to index. Exiting...")
    exit()

# Process each document and split into smaller chunks before indexing
total_docs = len(documents)
for i, doc in enumerate(documents, start=1):
    print(f"📄 Processing document {i}/{total_docs}...")

    chunks = chunk_text(doc["text"])
    if not chunks:
        print(f"⚠️ Skipping empty chunk for document {i}")
        continue

    for chunk in chunks:
        index_document(chunk, doc["lang"])

print(f"✅ Successfully indexed {total_docs} documents from `data/` folder!")