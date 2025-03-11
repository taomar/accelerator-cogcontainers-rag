import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from indexer import detect_language, generate_embedding, index_document, client

# ✅ Test Language Detection
def test_detect_language():
    assert detect_language("ما هي الفوائد الرئيسية للذكاء الاصطناعي؟") == "ar"
    assert detect_language("What are the benefits of AI?") == "en"

# ✅ Test Embedding Generation
def test_generate_embedding():
    arabic_embedding = generate_embedding("ما هي الفوائد الرئيسية للذكاء الاصطناعي؟", "ar")
    english_embedding = generate_embedding("What are the benefits of AI?", "en")

    assert isinstance(arabic_embedding, list) and len(arabic_embedding) > 0
    assert isinstance(english_embedding, list) and len(english_embedding) > 0

# ✅ Test Qdrant Collection
def test_qdrant_collections():
    """Test if Qdrant collections exist with correct names."""
    collections = client.get_collections()
    collection_names = [col.name for col in collections.collections]  # ✅ Fix here
    assert "rag_docs_en" in collection_names
    assert "rag_docs_ar" in collection_names

# ✅ Test Document Indexing
def test_index_document():
    index_document("Artificial intelligence is transforming industries.", "en")
    index_document("الذكاء الاصطناعي يحدث ثورة في الصناعات.", "ar")

    results_en = client.scroll("rag_docs_en", limit=1)
    results_ar = client.scroll("rag_docs_ar", limit=1)

    assert len(results_en) > 0
    assert len(results_ar) > 0