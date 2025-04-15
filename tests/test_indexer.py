import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
from indexer import detect_language, generate_embedding, index_document, client
from src.indexer import index_document

# Test Language Detection
def test_detect_language():
    assert detect_language("ما هي الفوائد الرئيسية للذكاء الاصطناعي؟") == "arabic"
    assert detect_language("What are the benefits of AI?") == "english"

# Test Document Indexing
def test_index_document():
    # Test English document
    english_text = '''
    Microsoft and OpenAI announced a groundbreaking $10 billion partnership in artificial intelligence today. 
    The collaboration, which takes place in Seattle, will focus on developing new AI technologies.
    '''
    
    # Test Arabic document
    arabic_text = '''
    أعلنت شركة مايكروسوفت عن افتتاح مركز جديد للذكاء الاصطناعي في دبي.
    يهدف المركز إلى تطوير حلول الذكاء الاصطناعي المخصصة للمنطقة العربية.
    '''
    
    # Index both documents
    english_result = index_document(english_text, 'test_english.txt')
    arabic_result = index_document(arabic_text, 'test_arabic.txt')
    
    assert english_result['status'] == 'success'
    assert arabic_result['status'] == 'success'
    assert english_result['chunks_processed'] > 0
    assert arabic_result['chunks_processed'] > 0

from indexer import process_document, index_document

# Test document with named entities
test_doc = """
Microsoft and OpenAI announced a major partnership in artificial intelligence today. 
The collaboration, which takes place in Seattle, will focus on developing new AI technologies.
CEO Satya Nadella emphasized the importance of responsible AI development.

In related news, Google's DeepMind team in London has also made significant progress in AI research.
The team, led by Demis Hassabis, published groundbreaking results in Nature journal.

Amazon's AWS division, based in Seattle, launched new machine learning services.
Jeff Bezos praised the innovation during his visit to their Dubai office.
"""

print("🔍 Testing Document Processing with NER\n")

# Process the document
chunks = process_document(test_doc, "test_doc.txt")

# Display results
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i + 1}/{len(chunks)}:")
    print("-" * 40)
    print("Text:", chunk["text"])
    print("\nMetadata:")
    print("- Language:", chunk["metadata"]["language"])
    print("\nDetected Entities:")
    for category, entities in chunk["metadata"]["entities"].items():
        print(f"- {category}: {', '.join(entities)}")
    print("-" * 40)

# Test indexing
print("\n📝 Testing Document Indexing\n")
index_document(test_doc, "test_doc.txt") 