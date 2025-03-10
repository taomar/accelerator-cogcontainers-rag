from qdrant_client import QdrantClient
import ollama
import re
import numpy as np
import requests
import os
import nltk
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
import string

nltk.download("punkt")

# Initialize Qdrant client
client = QdrantClient("localhost", port=6333)

# Embedding sizes for models
EMBEDDING_SIZES = {
    "english": 1024,  # bge-m3 (Optimized for retrieval)
    "arabic": 1024,   # jaluma/arabert embeddings (Padded to match Qdrant)
}

# API Endpoints
AZURE_LANGUAGE_API_URL = "http://localhost:5000/text/analytics/v3.1/languages"

# Function to detect language using Azure AI Language API
def detect_language(text):
    """Detects query language using Azure AI Language API."""
    payload = {"documents": [{"id": "1", "text": text}]}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(AZURE_LANGUAGE_API_URL, json=payload, headers=headers)
        response_json = response.json()

        if "documents" in response_json and response_json["documents"]:
            detected_lang = response_json["documents"][0]["detectedLanguage"]["iso6391Name"]
            return "arabic" if detected_lang == "ar" else "english"

    except Exception as e:
        print(f"⚠️ Language detection error: {e}")

    return "english"  # Default to English if detection fails

# Function to generate embeddings
import numpy as np

def generate_embedding(text, language):
    """Generates embeddings using different models for Arabic & English."""
    model_name = "jaluma/arabert-all-nli-triplet-matryoshka" if language == "arabic" else "bge-m3"
    
    response = ollama.embeddings(model=model_name, prompt=text)
    embedding = response["embedding"]

    # ✅ Fix: Ensure embedding size matches Qdrant (Arabic → 768, English → 1024)
    expected_size = 768 if language == "arabic" else 1024

    # 🔹 Ensure embedding has the correct size
    if len(embedding) < expected_size:
        embedding = np.pad(embedding, (0, expected_size - len(embedding)), 'constant')
    elif len(embedding) > expected_size:
        embedding = embedding[:expected_size]  # Truncate if larger

    return list(embedding)  # ✅ FIXED: No `.tolist()`, returning list directly

# Function to tokenize text for BM25
def tokenize_text(text, language):
    """Tokenizes input text for BM25 retrieval, handling Arabic separately."""
    if language == "arabic":
        return word_tokenize(text)  # Arabic tokenization (better for BM25)
    return [word.lower() for word in word_tokenize(text) if word not in string.punctuation]

# Function to search documents with optimized retrieval
def search_documents(query, language):
    """Retrieves documents using hybrid search (Vector + BM25)."""
    query_vector = generate_embedding(query, language)
    collection_name = "rag_docs_ar" if language == "arabic" else "rag_docs_en"

    if not client.collection_exists(collection_name):
        print(f"🚨 Collection '{collection_name}' not found in Qdrant. Skipping retrieval.")
        return []

    print(f"🔎 Searching for query: {query} in collection: {collection_name}")

    retrieved_docs = []  # ✅ Initialize the list before use
    seen_texts = set()   # ✅ Track unique document texts

    try:
        vector_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=20,  # Retrieve more docs to improve ranking
            with_payload=True
        )

        # ✅ Populate retrieved_docs properly
        for hit in vector_results:
            doc_text = hit.payload["text"]
            if doc_text not in seen_texts:
                retrieved_docs.append({"text": doc_text, "score": hit.score})
                seen_texts.add(doc_text)

        print(f"🔹 Retrieved {len(retrieved_docs)} unique documents")

    except Exception as e:
        print(f"⚠️ Vector search error: {e}")
        return []  # ✅ Return an empty list if retrieval fails

    # ✅ Ensure retrieved_docs exists before filtering
    if retrieved_docs:
        try:
            corpus = [doc["text"] for doc in retrieved_docs]
            tokenized_corpus = [tokenize_text(doc, language) for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)

            query_tokens = tokenize_text(query, language)
            bm25_scores = bm25.get_scores(query_tokens)

            for idx, doc in enumerate(retrieved_docs):
                doc["bm25_score"] = bm25_scores[idx]

            # ✅ Boost BM25 for Arabic queries
            weight_vector = 0.5 if language == "english" else 1.2  
            retrieved_docs = sorted(retrieved_docs, key=lambda x: (x["bm25_score"] * weight_vector) + x["score"], reverse=True)

        except Exception as e:
            print(f"⚠️ BM25 search error: {e}")

    return retrieved_docs  # ✅ Return the list safely


# Function to clean AI responses and enforce proper Arabic formatting
def clean_ai_response(text, language):
    """Cleans AI response text and ensures proper right-to-left (RTL) formatting for Arabic."""

    # Remove unwanted HTML tags
    text = re.sub(r'<.*?>', '', text)

    if language == "arabic":
        # ✅ Ensure proper Arabic bullets (nested lists fix)
        text = text.replace("•", "◼").replace("-", "◼")  # Fix unordered bullets
        text = text.replace("  *", "◼").replace("*", "◼")  # Handle extra bullets

        # ✅ Convert numbers to Arabic numerals
        text = text.replace("1.", "١.").replace("2.", "٢.").replace("3.", "٣.").replace("4.", "٤.").replace("5.", "٥.")

        # ✅ Enforce strict right alignment and better spacing
        text = text.replace("\n", "<br>")  # Preserve new lines
        text = f'<div dir="rtl" style="text-align: right; direction: rtl; unicode-bidi: embed; font-size: 20px; line-height: 2.2; font-family: Arial, sans-serif;">{text}</div>'

    return text

# Function to generate AI response

def generate_response(query, max_length=512, temperature=0.9, top_k=40, repetition_penalty=1.0):
    """Generates a response using the appropriate LLM model based on detected language."""

    language = detect_language(query)
    model_name = "gemma2:2b" if language == "arabic" else "qwen2.5:0.5b"

    if language == "arabic":
        prompt = f"""
        جاوب على السؤال التالي باللغة العربية فقط:

        **السؤال:** {query}

        **الإجابة يجب أن تكون:**
        ◼ منظمة ومفصلة
        ◼ بدون تكرار غير ضروري
        ◼ لا تتوقف في منتصف الجملة، تأكد من إنهاء الفكرة بالكامل
        """
    else:
        prompt = f"Answer the following question in clear, well-structured English:\n\n{query}"

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature, "top_k": top_k, "max_length": max_length, "repetition_penalty": repetition_penalty}
    )

    return response["message"]["content"]