import os
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Azure AI Configuration
AZURE_ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_LANGUAGE_KEY")

def extract_entities(text: str) -> dict:
    """Extract named entities from text using Azure Language Service."""
    
    # Remove trailing slash if present
    base_endpoint = AZURE_ENDPOINT.rstrip('/')
    endpoint = f"{base_endpoint}/text/analytics/v3.1/entities/recognition/general"
    
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Content-Type": "application/json"
    }
    
    payload = {
        "documents": [{
            "id": "1",
            "text": text,
            "language": "en"  # or "ar" for Arabic
        }]
    }

    print(f"Making request to: {endpoint}")
    response = requests.post(endpoint, headers=headers, json=payload)
    print(f"Response status: {response.status_code}")
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.text}")
        return None

# Test cases
test_texts = [
    "Microsoft and OpenAI are working on artificial intelligence in Seattle.",
    "Google announced new AI features in their cloud platform.",
    "Amazon's Jeff Bezos visited their new office in Dubai.",
    "هيئة الاتصالات وتقنية المعلومات تطلق مبادرة جديدة في الرياض"  # Arabic test
]

print("🔍 Testing Azure Named Entity Recognition\n")

for text in test_texts:
    print(f"Text: {text}")
    result = extract_entities(text)
    if result and "documents" in result:
        entities = result["documents"][0]["entities"]
        print("\nDetected Entities:")
        for entity in entities:
            print(f"- {entity['text']} ({entity['category']})")
    print("\n" + "="*50 + "\n") 