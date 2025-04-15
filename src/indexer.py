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
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables
load_dotenv()

# ================================
# 🔹 CONFIGURATION & INITIAL SETUP
# ================================

# Configuration from environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
LANGUAGE_DETECTION_URL = os.getenv("LANGUAGE_DETECTION_URL", "http://localhost:5000/text/analytics/v3.1/languages")
LANGUAGE_API_KEY = os.getenv("LANGUAGE_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))

# Initialize Qdrant client
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)

# Define embedding sizes based on model
EMBEDDING_SIZE = 1024  # Both English & Arabic use bge-m3 (same dimension)

# ================================
# 🔹 QDRANT COLLECTION SETUP
# ================================

def create_collection_if_not_exists(collection_name: str) -> None:
    """Create a collection if it doesn't exist."""
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=EMBEDDING_SIZE, distance="Cosine"),
        )
        print(f"Created new collection '{collection_name}'")

# Create collections for both languages
for lang in ["en", "ar"]:
    collection_name = f"rag_docs_{lang}"
    create_collection_if_not_exists(collection_name)

print("✅ Qdrant collections are now correctly set up!")

# ================================
# 🔹 LANGUAGE DETECTION FUNCTION
# ================================

def detect_language(text: str) -> str:
    """Detects language using Azure AI Language API with robust error handling."""
    try:
        response = requests.post(
            LANGUAGE_DETECTION_URL,
            headers={
                "Content-Type": "application/json",
                "Ocp-Apim-Subscription-Key": LANGUAGE_API_KEY
            },
            json={
                "documents": [
                    {
                        "id": "1",
                        "text": text
                    }
                ]
            }
        )
        response.raise_for_status()
        result = response.json()
        return result["documents"][0]["detectedLanguage"]["iso6391Name"]
    except Exception as e:
        print(f"Error detecting language: {e}")
        return "unknown"

# ================================
# 🔹 EMBEDDING GENERATION FUNCTION
# ================================

def generate_embedding(text: str) -> List[float]:
    """Generates embeddings using bge-m3 for both Arabic & English."""
    try:
        response = ollama.embeddings(model="bge-m3", prompt=text)
        embedding = response["embedding"]

        # Ensure embedding size matches Qdrant vector size
        if len(embedding) < EMBEDDING_SIZE:
            embedding = np.pad(embedding, (0, EMBEDDING_SIZE - len(embedding)), 'constant', constant_values=0)
        elif len(embedding) > EMBEDDING_SIZE:
            embedding = embedding[:EMBEDDING_SIZE]  # Truncate if too large

        return embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# ================================
# 🔹 DOCUMENT PROCESSING FUNCTION
# ================================

def process_document(text: str, filename: str = None) -> List[Dict[str, Any]]:
    """Process a document and prepare it for indexing."""
    # Split text into chunks
    chunks = textwrap.wrap(text, CHUNK_SIZE) if text.strip() else []
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Detect language for each chunk
        language = detect_language(chunk)
        
        processed_chunks.append({
            "text": chunk,
            "metadata": {
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": filename or "unknown",
                "language": language
            }
        })
    
    return processed_chunks

# ================================
# 🔹 DOCUMENT INDEXING FUNCTION
# ================================

def index_document(text: str, filename: str = None) -> None:
    """Index a document into Qdrant."""
    # Process the document
    processed_chunks = process_document(text, filename)
    
    # Index in batches
    for i in range(0, len(processed_chunks), BATCH_SIZE):
        batch = processed_chunks[i:i + BATCH_SIZE]
        
        # Prepare points for batch upload
        points = []
        for chunk in batch:
            # Generate embedding
            embedding = generate_embedding(chunk["text"])
            if embedding is None:
                continue
            
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk["text"],
                    "metadata": chunk["metadata"]
                }
            ))
        
        # Upload batch to Qdrant
        if points:
            collection_name = f"rag_docs_{chunk['metadata']['language']}"
            client.upsert(
                collection_name=collection_name,
                points=points
            )
    
    print(f"Indexed {len(processed_chunks)} chunks from {filename or 'text'}")

# ================================
# 🔹 DOCUMENT LOADING FUNCTION
# ================================

def load_documents() -> List[Dict[str, Any]]:
    """Loads text, JSON, and CSV documents from the `data/` folder."""
    documents = []

    for file in os.listdir("data"):
        file_path = os.path.join("data", file)

        # Load text files
        if file.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
                if text.strip():
                    documents.append({"text": text, "filename": file})

        # Load JSON files
        elif file.endswith(".json"):
            with open(file_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                for doc in json_data:
                    if "text" in doc and doc["text"].strip():
                        documents.append({"text": doc["text"], "filename": file})

        # Load CSV files (assuming columns: text, lang)
        elif file.endswith(".csv"):
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "text" in row and row["text"].strip():
                        documents.append({"text": row["text"], "filename": file})

    return documents

# ================================
# 🔹 MAIN EXECUTION
# ================================

if __name__ == "__main__":
    documents = load_documents()

    if not documents:
        print("⚠️ No documents to index. Exiting...")
        exit()

    # Process each document
    total_docs = len(documents)
    for i, doc in enumerate(documents, start=1):
        print(f"📄 Processing document {i}/{total_docs}...")
        index_document(doc["text"], doc["filename"])

    print(f"✅ Successfully indexed {total_docs} documents from `data/` folder!")