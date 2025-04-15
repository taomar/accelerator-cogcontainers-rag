from retriever import search_documents

# Test queries with named entities
test_queries = [
    ("What did Microsoft and OpenAI announce?", "english"),
    ("Tell me about Jeff Bezos and Amazon", "english"),
    ("What's happening in Seattle?", "english"),
    ("ماذا يحدث في دبي؟", "arabic"),  # What's happening in Dubai?
    ("من هو ساتيا ناديلا؟", "arabic")  # Who is Satya Nadella?
]

print("🔍 Testing Enhanced Retriever with NER\n")

for query, language in test_queries:
    print(f"\nQuery: {query}")
    print(f"Language: {language}")
    print("-" * 50)
    
    results = search_documents(query, language)
    
    if results:
        print(f"\nFound {len(results)} results:")
        for i, result in enumerate(results[:3], 1):  # Show top 3 results
            print(f"\n{i}. Score: {result['score']:.3f}")
            print(f"   Vector Score: {result['vector_score']:.3f}")
            print(f"   Entity Score: {result['entity_score']:.3f}")
            print(f"   Text: {result['text'][:200]}...")
            print("\n   Matched Entities:")
            for category, entities in result["matched_entities"].items():
                print(f"   - {category}: {', '.join(entities)}")
    else:
        print("No results found.")
    
    print("\n" + "="*50) 