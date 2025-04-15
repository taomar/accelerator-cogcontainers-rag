import os
from functools import lru_cache
from typing import Dict, List, Optional, Union
import requests
from datetime import datetime
import json

# Load environment variables
AZURE_LANGUAGE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT", "https://edge-rag-lang.cognitiveservices.azure.com")
AZURE_LANGUAGE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

# Cache size for language detection results
LANGUAGE_CACHE_SIZE = 1000
ENTITY_CACHE_SIZE = 1000

@lru_cache(maxsize=LANGUAGE_CACHE_SIZE)
def detect_language(text: str) -> str:
    """
    Detect the language of the given text with caching.
    Returns 'english' or 'arabic' or 'unknown'.
    """
    if not text.strip():
        return "unknown"
        
    try:
        # Prepare the request
        url = f"{AZURE_LANGUAGE_ENDPOINT}/text/analytics/v3.1/languages"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        body = {
            "documents": [{
                "id": "1",
                "text": text[:500]  # Use first 500 chars for efficiency
            }]
        }
        
        # Make the request with timeout
        response = requests.post(url, headers=headers, json=body, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        detected = result["documents"][0]["detectedLanguage"]["iso6391Name"]
        
        # Map to supported languages
        if detected == "ar":
            return "arabic"
        elif detected == "en":
            return "english"
        else:
            return "unknown"
            
    except Exception as e:
        print(f"Language detection error: {e}")
        # Fallback to simple heuristic
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        return "arabic" if arabic_chars > len(text) * 0.5 else "english"

@lru_cache(maxsize=ENTITY_CACHE_SIZE)
def extract_entities(text: str, language: str) -> List[Dict[str, str]]:
    """
    Extract named entities from text with caching and optimized processing.
    """
    if not text.strip():
        return []
        
    try:
        # Prepare the request
        url = f"{AZURE_LANGUAGE_ENDPOINT}/text/analytics/v3.1/entities/recognition/general"
        headers = {
            "Ocp-Apim-Subscription-Key": AZURE_LANGUAGE_KEY,
            "Content-Type": "application/json"
        }
        body = {
            "documents": [{
                "id": "1",
                "text": text,
                "language": "ar" if language == "arabic" else "en"
            }]
        }
        
        # Make the request with timeout
        response = requests.post(url, headers=headers, json=body, timeout=5)
        response.raise_for_status()
        
        result = response.json()
        entities = []
        
        # Process entities with confidence threshold
        for entity in result["documents"][0]["entities"]:
            if entity.get("confidenceScore", 0) > 0.5:  # Filter low confidence entities
                entities.append({
                    "text": entity["text"],
                    "category": entity["category"],
                    "confidence": entity["confidenceScore"]
                })
        
        return entities
        
    except Exception as e:
        print(f"Entity extraction error: {e}")
        return []

def process_text_batch(texts: List[str]) -> List[Dict[str, Union[str, List[Dict[str, str]]]]]:
    """
    Process a batch of texts for language detection and entity extraction.
    """
    results = []
    for text in texts:
        language = detect_language(text)
        entities = extract_entities(text, language)
        results.append({
            "text": text,
            "language": language,
            "entities": entities,
            "timestamp": datetime.utcnow().isoformat()
        })
    return results 