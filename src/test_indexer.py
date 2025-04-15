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