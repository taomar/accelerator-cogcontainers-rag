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

# Define embedding sizes based on language-specific models
EMBEDDING_SIZES = {
    "en": 1024,  # English embeddings (bge-m3)
    "ar": 1024,   # Arabic embeddings (bge-m3)
}

# ================================
# 🔹 QDRANT COLLECTION SETUP
# ================================

for lang, vector_size in EMBEDDING_SIZES.items():
    collection_name = f"rag_docs_{lang[:2]}"  # 'rag_docs_ar' for Arabic, 'rag_docs_en' for English

    if client.collection_exists(collection_name):
        existing_collection_info = client.get_collection(collection_name)

        # Ensure existing collection has the correct vector size
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

# ================================
# 🔹 LANGUAGE DETECTION FUNCTION
# ================================

def detect_language(text):
    """Detects language using Azure AI Language API with error handling."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "ar" if detected_lang == "ar" else "en"

    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "en"  # Default to English if detection fails

# ================================
# 🔹 EMBEDDING GENERATION FUNCTION
# ================================

def generate_embedding(text, language):
    """Generates embeddings using different models for Arabic & English."""
    model_name = "bge-m3" if language == "ar" else "bge-m3"

    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # Ensure embeddings match Qdrant vector size
    expected_size = EMBEDDING_SIZES[language]
    if len(embedding) < expected_size:
        embedding = np.pad(embedding, (0, expected_size - len(embedding)), 'constant')
    elif len(embedding) > expected_size:
        embedding = embedding[:expected_size]  # Truncate if too large

    return embedding

# ================================
# 🔹 DOCUMENT INDEXING FUNCTION
# ================================

def index_document(text, lang):
    """Indexes a document into Qdrant after generating embeddings."""
    embedding = generate_embedding(text, lang)
    if embedding is None:
        print(f"⚠️ Skipping indexing due to embedding error for text: {text[:30]}...")
        return

    collection_name = f"rag_docs_{lang[:2]}"
    if not client.collection_exists(collection_name):
        print(f"⚠️ Collection `{collection_name}` not found! Skipping indexing...")
        return

    point = {
        "id": str(uuid.uuid4()),  # Unique UUID for each chunk
        "vector": embedding,  # ✅ FIXED: Directly use the list (no `.tolist()`)
        "payload": {"text": text, "language": lang}
    }

    client.upsert(collection_name=collection_name, points=[point])

# ================================
# 🔹 TEXT CHUNKING FUNCTION
# ================================

def chunk_text(text, chunk_size=200):
    """Splits text into smaller chunks for better retrieval performance."""
    if not text.strip():
        return []  # Return an empty list if text is empty
    return textwrap.wrap(text, chunk_size)

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

    if not documents:
        print("⚠️ No valid documents found for indexing!")

    return documents

# ================================
# 🔹 INDEXING DOCUMENTS INTO QDRANT
# ================================

documents = load_documents()

if not documents:
    print("⚠️ No documents to index. Exiting...")
    exit()

# Process each document and split into smaller chunks before indexing
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