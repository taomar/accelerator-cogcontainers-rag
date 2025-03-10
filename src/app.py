import streamlit as st
from retriever import generate_response, search_documents, detect_language
from indexer import index_document
import ollama

# 🎨 Streamlit UI Setup
st.set_page_config(page_title="AI-Powered RAG System", layout="wide")

# 🌍 Header
st.markdown("<h1 style='text-align: center;'>🔍 AI-Powered RAG System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Azure AI Containers to run this RAG system offline or on premise and enhance accuracy, retrieval, and insights.</p>", unsafe_allow_html=True)


# 🏗️ First Row: Upload, Query Input, and Generation Parameters
col_upload, col_query, col_params = st.columns([1.2, 1.5, 1.3])

with col_upload:
    st.subheader("📂 Upload a Document")
    uploaded_file = st.file_uploader("Upload (TXT, JSON, CSV)", type=["txt", "json", "csv"])

    if uploaded_file:
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        file_contents = uploaded_file.read().decode("utf-8")
        lang = detect_language(file_contents[:200])
        index_document(file_contents)
        st.info(f"📄 Indexed in `{lang}` language!")

    # 📥 Sample Download
    sample_text = "Sample English text.\n\nيمكنك أيضًا اختبار مستند باللغة العربية."
    st.download_button("📥 Download Sample", sample_text, file_name="sample_doc.txt")

with col_query:
    st.subheader("💬 Query Input")
    language_option = st.selectbox("🌍 Language:", ["Auto-Detect", "Arabic", "English"])
    query_text = st.text_area("Enter your query:", placeholder="Type your question here...")

    # 🔎 Search & Generate Button
    if st.button("🚀 Search & Generate"):
        if query_text:
            language = detect_language(query_text) if language_option == "Auto-Detect" else language_option.lower()
            retrieved_docs = search_documents(query_text, language)

            st.session_state["query_text"] = query_text
            st.session_state["retrieved_docs"] = retrieved_docs
            st.session_state["language"] = language

with col_params:
    st.subheader("🎛️ Generation Parameters")
    max_length = st.slider("Max Length", 50, 1024, 256)
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
    top_k = st.slider("Top-k", 1, 100, 50)
    repetition_penalty = st.slider("Repetition Penalty", 0.0, 2.0, 1.2)

    st.session_state["generation_params"] = {
        "max_length": max_length,
        "temperature": temperature,
        "top_k": top_k,
        "repetition_penalty": repetition_penalty,
    }

# 🏗️ Second Row: Retrieved Docs & AI Response
col_retrieved, col_ai = st.columns(2)

with col_retrieved:
    st.subheader("📄 Retrieved Documents")
    retrieved_docs = st.session_state.get("retrieved_docs", [])

    if retrieved_docs:
        for idx, doc in enumerate(retrieved_docs):
            st.markdown(f"**{idx + 1}.** {doc['text']} _(Score: {doc['score']:.2f})_")
    else:
        st.warning("⚠️ No relevant documents found!")

with col_ai:
    st.subheader("🤖 AI Response")
    if "query_text" in st.session_state:
        ai_response = generate_response(
            st.session_state["query_text"],
            max_length=st.session_state["generation_params"]["max_length"],
            temperature=st.session_state["generation_params"]["temperature"],
            top_k=st.session_state["generation_params"]["top_k"],
            repetition_penalty=st.session_state["generation_params"]["repetition_penalty"]
        )

        st.markdown(f"<div class='ai-response'><h4>🔹 AI Response:</h4><p>{ai_response}</p></div>", unsafe_allow_html=True)

# 🏗️ Third Row: Example Prompts & System Information
col_prompts, col_info = st.columns(2)

with col_prompts:
    st.subheader("📝 Example Prompts")
    example_prompts = [
        ("What are the key benefits of artificial intelligence in healthcare?", True),
        ("ما هي الفوائد الرئيسية للذكاء الاصطناعي في الرعاية الصحية؟", True),
        ("How did AI influence ancient civilizations?", False),
        ("كيف أثر الذكاء الاصطناعي على الحضارات القديمة؟", False),
    ]
    for prompt, is_indexed in example_prompts:
        indexed_label = "✅ Indexed" if is_indexed else "❌ Not Indexed"
        st.markdown(f"**{indexed_label}:** {prompt}")

with col_info:
    st.subheader("ℹ️ System Information")
    st.markdown("""
    - **Retrieval Method**: Hybrid (Vector Search + BM25)  
    - **Vector Database**: [Qdrant](https://qdrant.tech/)  
    - **Embedding Models**: `bge-m3` for English & Arabic  
    - **LLM Models**: `gemma2:2b` (Arabic) & `qwen2.5:0.5b` (English)  
    - **Azure AI Containers**: better accuracy, security, and usability 
    """)