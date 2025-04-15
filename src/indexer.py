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
from typing import List, Dict, Any, Optional

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

# Azure Language Service Configuration
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

if not AZURE_LANGUAGE_ENDPOINT or not AZURE_LANGUAGE_KEY:
    raise ValueError("Azure Language Service configuration missing. Please set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY environment variables.")

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

def create_collection_if_not_exists(client: QdrantClient, collection_name: str, vector_size: int = 1024) -> None:
    """Creates a collection if it doesn't exist."""
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance="Cosine")
        )
        print(f"Created new collection '{collection_name}'")

# Create collections for both languages
for lang in ["en", "ar"]:
    collection_name = f"rag_docs_{lang}"
    create_collection_if_not_exists(client, collection_name)

print("✅ Qdrant collections are now correctly set up!")

# ================================
# 🔹 LANGUAGE DETECTION FUNCTION
# ================================

def detect_language(text: str) -> str:
    """Detects the language of a given text using Azure Language Service."""
    try:
        # Remove trailing slash if present and add the correct path
        base_endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip('/')
        endpoint = f"{base_endpoint}/text/analytics/v3.1/languages"
        
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "documents": [{
                "id": "1",
                "text": text
            }]
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "documents" in result and result["documents"]:
            detected_lang = result["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "arabic" if detected_lang == "ar" else "english"
            
    except Exception as e:
        print(f"Error detecting language: {e}")
    
    return "english"  # Default to English if detection fails

# ================================
# 🔹 EMBEDDING GENERATION FUNCTION
# ================================

def generate_embedding(text: str, language: str) -> List[float]:
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

def extract_entities(text: str, language: str = "en") -> List[Dict[str, str]]:
    """Extract named entities from text using Azure Language Service."""
    try:
        base_endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip('/')
        endpoint = f"{base_endpoint}/text/analytics/v3.1/entities/recognition/general"
        
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        
        payload = {
            "documents": [{
                "id": "1",
                "text": text,
                "language": "ar" if language == "arabic" else "en"
            }]
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        response.raise_for_status()
        
        result = response.json()
        if "documents" in result and result["documents"]:
            return [{"text": entity["text"], "category": entity["category"]} 
                   for entity in result["documents"][0]["entities"]]
            
    except Exception as e:
        print(f"Error extracting entities: {e}")
    
    return []

def process_document(text: str, filename: str = None) -> List[Dict[str, Any]]:
    """Process a document and prepare it for indexing."""
    # Split text into chunks
    chunks = textwrap.wrap(text, CHUNK_SIZE) if text.strip() else []
    
    processed_chunks = []
    for i, chunk in enumerate(chunks):
        # Detect language for each chunk
        language = detect_language(chunk)
        
        # Extract entities from the chunk
        entities = extract_entities(chunk, language)
        
        # Group entities by category
        entities_by_category = {}
        for entity in entities:
            category = entity["category"]
            if category not in entities_by_category:
                entities_by_category[category] = []
            entities_by_category[category].append(entity["text"])
        
        processed_chunks.append({
            "text": chunk,
            "metadata": {
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": filename or "unknown",
                "language": language,
                "entities": entities_by_category  # Store categorized entities
            }
        })
    
    return processed_chunks

# ================================
# 🔹 DOCUMENT INDEXING FUNCTION
# ================================

def index_document(text: str, filename: str) -> None:
    """Index a document into Qdrant."""
    client = QdrantClient("localhost", port=6333)
    
    # Create collections for both languages if they don't exist
    create_collection_if_not_exists(client, "rag_docs_en")
    create_collection_if_not_exists(client, "rag_docs_ar")
    print("✅ Qdrant collections are now correctly set up!")
    
    # Process the document into chunks
    chunks = process_document(text, filename)
    
    # Prepare points for each language collection
    en_points = []
    ar_points = []
    
    for i, chunk in enumerate(chunks):
        language = chunk["metadata"]["language"]
        embedding = generate_embedding(chunk["text"], language)
        
        point = PointStruct(
            id=i,
            vector=embedding,
            payload={
                "text": chunk["text"],
                "metadata": chunk["metadata"]
            }
        )
        
        if language == "arabic":
            ar_points.append(point)
        else:
            en_points.append(point)
    
    # Batch upsert points for each language
    if en_points:
        client.upsert(collection_name="rag_docs_en", points=en_points)
    if ar_points:
        client.upsert(collection_name="rag_docs_ar", points=ar_points)

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