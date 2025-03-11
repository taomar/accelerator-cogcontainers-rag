from src.retriever import search_documents, detect_language

# Sample Queries
queries = [
    "How is AI improving diagnostics in healthcare?",  # Indexed Query (English)
    "كيف يمكن للذكاء الاصطناعي تحسين التشخيص في الرعاية الصحية؟",  # Indexed Query (Arabic)
    "What are the ethical concerns of AI in medicine?",  # Non-Indexed Query (English)
    "ما هي المخاوف الأخلاقية حول استخدام الذكاء الاصطناعي في الطب؟"  # Non-Indexed Query (Arabic)
]

print("\n🔍 Running Retriever Tests...\n")

for query in queries:
    detected_language = detect_language(query)
    retrieved_docs = search_documents(query, detected_language)

    print(f"📌 Query: {query}")
    print(f"🌍 Detected Language: {detected_language}")
    
    if retrieved_docs:
        print("📄 Retrieved Documents:")
        for idx, doc in enumerate(retrieved_docs, start=1):
            print(f"{idx}. {doc['text']} (Score: {doc['score']:.2f})")
    else:
        print("⚠️ No relevant documents found!")

    print("\n" + "="*80 + "\n")