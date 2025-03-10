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

# Load API details from environment variables (.env)
AZURE_LANGUAGE_API_URL = os.getenv("AZURE_LANGUAGE_API_URL", "http://localhost:5000/text/analytics/v3.1/languages")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# ✅ Correct embedding sizes based on the model
EMBEDDING_SIZES = {
    "en": 1024,  # bge-m3 (English)
    "ar": 768,   # jaluma/arabert (Arabic)
}

# ✅ Ensure Qdrant collections exist with correct dimensions
for lang, vector_size in EMBEDDING_SIZES.items():
    collection_name = f"rag_docs_{lang[:2]}"

    if client.collection_exists(collection_name):
        existing_collection_info = client.get_collection(collection_name)

        # 🔹 Fix: Correct way to check vector size
        existing_vector_size = existing_collection_info.config.params.vectors.size

        if existing_vector_size != vector_size:
            print(f"🚨 Collection `{collection_name}` has incorrect vector size ({existing_vector_size}). Deleting & recreating...")
            client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
        print(f"🚀 Creating collection `{collection_name}` with vector size {vector_size}")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine"),
        )

print("✅ Qdrant collections are now correctly set up!")

# Function to detect language using Azure AI Language API
def detect_language(text):
    """Detects language using Azure AI Language API with improved error handling."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            if detected_lang == "ar":
                return "ar"
            elif detected_lang == "en":
                return "en"

    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "en"  # Default to English if detection fails

# Function to generate embeddings using the correct model
def generate_embedding(text, language):
    """Generates embeddings using different models for Arabic & English."""
    model_name = "jaluma/arabert-all-nli-triplet-matryoshka" if language == "ar" else "bge-m3"
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # ✅ Fix: Ensure embedding size matches Qdrant (Arabic → 768, English → 1024)
    expected_size = EMBEDDING_SIZES[language]
    if len(embedding) != expected_size:
        embedding = np.array(embedding[:expected_size])  # Truncate if larger

    return embedding  # ✅ FIXED: No `.tolist()`, since it's already a list

# Function to index document into Qdrant
def index_document(text, lang):
    """Indexes a document into Qdrant."""
    embedding = generate_embedding(text, lang)
    if embedding is None:
        print(f"⚠️ Skipping indexing due to embedding error for text: {text[:30]}...")
        return
    
    collection_name = f"rag_docs_{lang[:2]}"

    if not client.collection_exists(collection_name):
        print(f"⚠️ Collection `{collection_name}` not found! Skipping indexing...")
        return

    point = {
        "id": str(uuid.uuid4()),
        "vector": embedding,  # ✅ FIXED: Removed `.tolist()`, since it's already a list
        "payload": {"text": text, "language": lang}
    }
    client.upsert(collection_name=collection_name, points=[point])

# Function to chunk text
def chunk_text(text, chunk_size=200):
    """Splits text into smaller chunks for better retrieval."""
    if not text.strip():
        return []  # 🔹 Return an empty list if text is empty
    return textwrap.wrap(text, chunk_size)

# Function to load documents from `data/` folder
def load_documents():
    """Loads text, JSON, and CSV documents from the data folder."""
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

    if not documents:
        print("⚠️ No valid documents found for indexing!")

    return documents

# Load and index documents
documents = load_documents()

if not documents:
    print("⚠️ No documents to index. Exiting...")
    exit()

# Store document chunks in Qdrant
for doc in documents:
    if not doc.get("text"):
        print(f"⚠️ Skipping document with missing text: {doc}")
        continue

    chunks = chunk_text(doc["text"])
    if not chunks:
        print(f"⚠️ Skipping empty chunk for document: {doc}")
        continue

    for chunk in chunks:
        index_document(chunk, doc["lang"])

print("✅ Documents indexed successfully from `data/` folder!")