import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from retriever import detect_language, generate_embedding, search_documents, generate_response

# ✅ Test Language Detection
def test_detect_language():
    assert detect_language("ما هو الذكاء الاصطناعي؟") == "arabic"
    assert detect_language("What is artificial intelligence?") == "english"

# ✅ Test Embedding Generation
def test_generate_embedding():
    arabic_embedding = generate_embedding("ما هو الذكاء الاصطناعي؟", "arabic")
    english_embedding = generate_embedding("What is artificial intelligence?", "english")

    assert isinstance(arabic_embedding, list) and len(arabic_embedding) > 0
    assert isinstance(english_embedding, list) and len(english_embedding) > 0

# ✅ Test Document Retrieval
def test_search_documents():
    results = search_documents("What is AI?", "english")
    assert isinstance(results, list)

# ✅ Test AI Response Generation
def test_generate_response():
    response_en = generate_response("What is artificial intelligence?")
    response_ar = generate_response("ما هو الذكاء الاصطناعي؟")

    assert isinstance(response_en, str) and len(response_en) > 0
    assert isinstance(response_ar, str) and len(response_ar) > 0