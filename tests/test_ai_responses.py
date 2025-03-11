import os
import json
from src.retriever import search_documents, generate_response, detect_language

# Ensure results folder exists
os.makedirs("results", exist_ok=True)

# 📌 Test queries (Indexed & Non-Indexed)
TEST_QUERIES = [
    # ✅ Indexed Queries (Should use retrieved documents)
    {"query": "How is AI improving diagnostics in healthcare?", "expected_lang": "english"},
    {"query": "كيف يمكن للذكاء الاصطناعي تحسين التشخيص في الرعاية الصحية؟", "expected_lang": "arabic"},

    # ❌ Non-Indexed Queries (Should rely mostly on LLM)
    {"query": "What are the future trends in AI healthcare?", "expected_lang": "english"},
    {"query": "ما هي الاتجاهات المستقبلية للذكاء الاصطناعي في الرعاية الصحية؟", "expected_lang": "arabic"},
]

# 📂 Store results
results = []

print("\n🔍 Running AI Response Tests...\n")

for test in TEST_QUERIES:
    query_text = test["query"]
    expected_lang = test["expected_lang"]

    # 🌍 Detect Language
    detected_lang = detect_language(query_text)
    assert detected_lang == expected_lang, f"❌ Mismatch! Expected {expected_lang}, detected {detected_lang}"

    # 🔎 Retrieve Documents
    retrieved_docs = search_documents(query_text, detected_lang)
    retrieved_texts = [doc["text"] for doc in retrieved_docs]

    # 🤖 Generate AI Response
    ai_response = generate_response(query_text)

    # 📝 Save Test Result
    result = {
        "query": query_text,
        "detected_language": detected_lang,
        "retrieved_docs": retrieved_texts,
        "ai_response": ai_response,
    }
    results.append(result)

    print(f"✅ Query: {query_text[:50]}...")
    print(f"   📄 Retrieved Docs: {len(retrieved_docs)}")
    print(f"   🤖 AI Response: {ai_response[:100]}...\n")

# 📌 Save to JSON for manual review
output_file = "results/ai_responses.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=4)

print(f"\n📂 AI responses saved to `{output_file}` for review.")