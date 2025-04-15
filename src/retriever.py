import os
import re
import requests
import ollama
import numpy as np
import nltk
import string
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from typing import List, Dict, Any

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data
nltk.download("punkt")

# ✅ Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# ✅ Define embedding sizes for models
EMBEDDING_SIZES = {
    "english": 1024,  # bge-m3 (Optimized for retrieval)
    "arabic": 1024,   # bge-m3 embeddings
}

# ✅ Azure AI Language Service Configuration
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

if not AZURE_LANGUAGE_ENDPOINT or not AZURE_LANGUAGE_KEY:
    raise ValueError("Azure Language Service configuration missing. Please set AZURE_LANGUAGE_ENDPOINT and AZURE_LANGUAGE_KEY environment variables.")

# -----------------------------------------------
# 🔹 Function: Detect Query Language
# -----------------------------------------------

def detect_language(text):
    """Detects the language of a given query using Azure Language Service."""
    
    # Remove trailing slash if present and add the correct path
    base_endpoint = AZURE_LANGUAGE_ENDPOINT.rstrip('/')
    endpoint = f"{base_endpoint}/text/analytics/v3.1/languages"
    
    print(f"Using endpoint: {endpoint}")  # Debug print
    
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

    try:
        print("Making request to Azure...")  # Debug print
        response = requests.post(endpoint, headers=headers, json=payload)
        print(f"Response status: {response.status_code}")  # Debug print
        response.raise_for_status()
        response_json = response.json()
        print(f"Response JSON: {response_json}")  # Debug print

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "arabic" if detected_lang == "ar" else "english"

    except requests.exceptions.RequestException as e:
        print(f"⚠️ Azure Language API error: {e}")
        print(f"Response content: {getattr(e.response, 'text', 'No response content')}")  # Debug print
    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "english"  # Default to English if detection fails

# -----------------------------------------------
# 🔹 Function: Generate Query Embeddings
# -----------------------------------------------

def generate_embedding(text, language):
    """Generates embeddings using different models for Arabic & English queries."""
    model_name = "bge-m3" if language == "arabic" else "bge-m3"
    
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # ✅ Fix: Ensure embedding size matches Qdrant expectations
    expected_size = EMBEDDING_SIZES[language]

    # 🔹 Ensure embedding has the correct size
    if len(embedding) < expected_size:
        embedding = np.pad(embedding, (0, expected_size - len(embedding)), 'constant')
    elif len(embedding) > expected_size:
        embedding = embedding[:expected_size]  # Truncate if larger

    return list(embedding)  # ✅ FIXED: Returning as list directly

# -----------------------------------------------
# 🔹 Function: Tokenize Text for BM25
# -----------------------------------------------

def tokenize_text(text, language):
    """Tokenizes input text for BM25 retrieval, handling Arabic separately."""
    if language == "arabic":
        return word_tokenize(text)  # Arabic tokenization (better for BM25)
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# -----------------------------------------------
# 🔹 Function: Extract Entities
# -----------------------------------------------

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

# -----------------------------------------------
# 🔹 Function: Calculate Entity Score
# -----------------------------------------------

def calculate_entity_score(query_entities: List[Dict[str, str]], doc_entities: Dict[str, List[str]]) -> float:
    """Calculate similarity score based on matching entities."""
    if not query_entities or not doc_entities:
        return 0.0
    
    score = 0.0
    for query_entity in query_entities:
        query_text = query_entity["text"].lower()
        query_category = query_entity["category"]
        
        # Check if the entity exists in the same category
        if query_category in doc_entities:
            doc_entities_in_category = [e.lower() for e in doc_entities[query_category]]
            if query_text in doc_entities_in_category:
                score += 1.0  # Direct match in same category
            else:
                # Check for partial matches
                for doc_entity in doc_entities_in_category:
                    if query_text in doc_entity or doc_entity in query_text:
                        score += 0.5  # Partial match
    
    return score / len(query_entities)  # Normalize score

# -----------------------------------------------
# 🔹 Function: Search Documents with Hybrid Retrieval
# -----------------------------------------------

def search_documents(query: str, language: str) -> List[Dict[str, Any]]:
    """
    Enhanced search using both vector similarity and entity matching.
    """
    # Extract entities from the query
    query_entities = extract_entities(query, language)
    print(f"\n🔍 Query Entities: {[f'{e['text']} ({e['category']})' for e in query_entities]}")
    
    # Generate query vector
    query_vector = generate_embedding(query, language)
    collection_name = "rag_docs_ar" if language == "arabic" else "rag_docs_en"

    if not client.collection_exists(collection_name):
        print(f"Collection '{collection_name}' not found")
        return []

    try:
        # Get initial results using vector search
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=20,  # Get more results initially for re-ranking
            with_payload=True,
            score_threshold=0.0
        )

        # Process and re-rank results
        enhanced_results = []
        for hit in vector_results:
            doc_text = hit.payload.get("text", "")
            doc_metadata = hit.payload.get("metadata", {})
            doc_entities = doc_metadata.get("entities", {})
            
            # Calculate entity matching score
            entity_score = calculate_entity_score(query_entities, doc_entities)
            
            # Combine vector similarity with entity score
            combined_score = (hit.score + entity_score) / 2 if entity_score > 0 else hit.score
            
            enhanced_results.append({
                "text": doc_text,
                "score": combined_score,
                "vector_score": hit.score,
                "entity_score": entity_score,
                "source": doc_metadata.get("source", "Unknown"),
                "chunk_id": doc_metadata.get("chunk_id", 0),
                "total_chunks": doc_metadata.get("total_chunks", 1),
                "language": doc_metadata.get("language", language),
                "matched_entities": doc_entities
            })

        # Sort by combined score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top 10 results
        return enhanced_results[:10]

    except Exception as e:
        print(f"Search error: {e}")
        return []

# -----------------------------------------------
# 🔹 Function: Clean AI Response & Apply Arabic Formatting
# -----------------------------------------------

def clean_ai_response(text, language):
    """Cleans AI-generated responses and ensures proper right-to-left (RTL) formatting for Arabic."""

    # Remove unwanted HTML tags
    text = re.sub(r'<.*?>', '', text)

    if language == "arabic":
        # ✅ Ensure proper Arabic bullets
        text = text.replace("•", "◼").replace("-", "◼")  
        text = text.replace("  *", "◼").replace("*", "◼")  

        # ✅ Convert numbers to Arabic numerals
        text = text.replace("1.", "١.").replace("2.", "٢.").replace("3.", "٣.").replace("4.", "٤.").replace("5.", "٥.")

        # ✅ Enforce strict right alignment and better spacing
        text = text.replace("\n", "<br>")  # Preserve new lines
        text = f'<div dir="rtl" style="text-align: right; direction: rtl; unicode-bidi: embed; font-size: 20px; line-height: 2.2; font-family: Arial, sans-serif;">{text}</div>'

    return text

# -----------------------------------------------
# 🔹 Function: Generate AI Response
# -----------------------------------------------

def generate_response(query, max_length=512, temperature=0.9, top_k=40, repetition_penalty=1.0):
    """Generates a response using the appropriate LLM model based on detected language."""
    
    language = detect_language(query)
    model_name = "gemma3:1b" if language == "arabic" else "phi4-mini:3.8b"

    if language == "arabic":
        prompt = f"""
        أنت مساعد ذكي متخصص باللغة العربية. يجب أن تجيب باللغة العربية الفصحى فقط.
        لا تستخدم أي كلمات إنجليزية أو رموز غير عربية.
        
        السؤال: {query}
        
        قواعد الإجابة:
        ١. استخدم اللغة العربية الفصحى فقط
        ٢. تجنب أي كلمات أجنبية أو رموز غير عربية
        ٣. استخدم المصطلحات العربية المعتمدة (مثل: الذكاء الاصطناعي بدلاً من AI)
        ٤. نظم الإجابة بشكل واضح ومرتب
        ٥. اكتب إجابة كاملة ومفصلة
        ٦. استخدم الأرقام العربية (١، ٢، ٣) بدلاً من الأرقام الإنجليزية
        """
    else:
        prompt = f"Answer the following question in clear, well-structured English:\n\n{query}"

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": temperature,
            "top_k": top_k,
            "max_length": max_length,
            "repetition_penalty": repetition_penalty,
            "stop": ["</s>", "user:", "assistant:"]  # Prevent model from continuing conversation
        }
    )

    # Clean and format the response
    response_text = response["message"]["content"]
    return clean_ai_response(response_text, language)