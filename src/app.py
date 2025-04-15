import streamlit as st
from retriever import generate_response, search_documents, detect_language
from indexer import index_document
import ollama
import os
from dotenv import load_dotenv
from indexer import load_documents

# Load environment variables
load_dotenv()

# 🎨 Streamlit UI Setup
st.set_page_config(
    page_title="Edge RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🌍 Header
st.markdown("<h1 style='text-align: center;'>🔍 Edge RAG Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Azure AI Containers to run this RAG system offline or on premise and enhance accuracy, retrieval, and insights.</p>", unsafe_allow_html=True)

# Initialize session state
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = None

# Custom CSS for better UI
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        margin: 5px 0;
    }
    .stTextInput>div>div>input {
        font-size: 16px;
    }
    .stMarkdown {
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Hide the map container and duplicate title */
    .element-container:has(iframe),
    .element-container:has(> div > .stMarkdown:first-child h1) {
        display: none !important;
    }
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
    }
    /* Adjust title spacing */
    h1 {
        margin-bottom: 1rem !important;
    }
    /* Hide the map container */
    [data-testid="stDecoration"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("📄 Document Management")
    
    # Document Upload Section
    st.subheader("Upload Documents")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['txt', 'pdf', 'docx', 'json', 'csv'],
        help="Supported formats: TXT, PDF, DOCX, JSON, CSV"
    )
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            try:
                # Save uploaded file temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # Index the document
                index_document(temp_path)
                st.success("✅ Document indexed successfully!")
                st.session_state.documents_indexed = True
                
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                st.error(f"❌ Error indexing document: {str(e)}")

    # Load Documents Button
    if st.button("📂 Load Sample Documents", help="Load pre-indexed sample documents about AI in healthcare"):
        with st.spinner("Loading documents..."):
            try:
                load_documents()
                st.success("✅ Documents loaded successfully!")
                st.session_state.documents_indexed = True
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")

    # Models Information
    st.title("🤖 System Information")
    st.markdown("""
    ### Models
    - **Embedding**: `bge-m3` (English & Arabic)
    - **Response Generation**: 
        - English: `gemma3:1b`
        - Arabic: `phi4-mini:3.8b`
    - **Database**: Qdrant + BM25

    ### Azure AI Services
    - **Language Detection**: Azure Language Service
    - **Named Entity Recognition**: Azure NER Service
    - **Content Safety**: Azure Content Safety
    - **Document Intelligence**: Azure Document Intelligence
    """)

# Main Content

# Search Input
query = st.text_input(
    "Ask a question about AI:",
    placeholder="e.g., What are the benefits of AI in healthcare?",
    key="search_input"
)

# Search Button
if st.button("🔍 Search", use_container_width=True):
    if query:
        with st.spinner("Searching through documents..."):
            try:
                # Detect language automatically
                language = detect_language(query)
                
                # Show detected language with appropriate emoji
                lang_emoji = "🇬🇧" if language == "english" else "🇸🇦"
                lang_display = "English" if language == "english" else "Arabic"
                st.info(f"{lang_emoji} Detected Language: {lang_display}")
                
                # Search for relevant documents
                results = search_documents(query, language)
                
                if results:
                    # Generate AI response first
                    with st.spinner("Generating response..."):
                        response = generate_response(query, results)
                        st.subheader("🤖 AI Response")
                        st.write(response)
                    
                    # Display sources below the response
                    st.subheader("📚 Sources")
                    for i, doc in enumerate(results, 1):
                        source_name = doc.get('source', 'Unknown').split('/')[-1] if doc.get('source') else 'Unknown'
                        lang_emoji = "🇬🇧" if doc.get('language') == "english" else "🇸🇦"
                        lang_display = doc.get('language', 'unknown').capitalize()
                        
                        with st.expander(f"Source {i} (Relevance: {doc['score']:.2f})"):
                            st.write(f"**Document:** {source_name}")
                            st.write(f"**Language:** {lang_emoji} {lang_display}")
                            st.write("**Relevant Content:**")
                            st.markdown(f"```\n{doc.get('text', 'No content available')}\n```")
                else:
                    st.warning("No relevant documents found. Try rephrasing your question or loading more documents.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a question to search.")

# Example Prompts
st.subheader("💡 Example Questions")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**English:**")
    st.markdown("- What are the key benefits of AI in healthcare?")
    st.markdown("- How does AI improve disease diagnostics?")

with col2:
    st.markdown("**Arabic:**")
    st.markdown("- ما هي فوائد الذكاء الاصطناعي في الرعاية الصحية؟")
    st.markdown("- كيف يساعد الذكاء الاصطناعي في تشخيص الأمراض؟")