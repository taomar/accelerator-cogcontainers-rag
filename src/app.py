import streamlit as st
from retriever import generate_response, search_documents, detect_language
from indexer import index_document, load_documents
import ollama
import os
import nltk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_app():
    """Initialize app dependencies and configurations."""
    # Download NLTK data if not already present
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    
    # Initialize Qdrant collections
    try:
        load_documents()  # This will set up collections if they don't exist
    except Exception as e:
        st.error(f"Error initializing collections: {str(e)}")

# 🎨 Streamlit UI Setup
st.set_page_config(
    page_title="Edge RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Run setup
setup_app()

# 🌍 Header
st.markdown("<h1 style='text-align: center;'>🔍 Edge RAG Search</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Powered by Azure AI Services to enhance accuracy, retrieval, and insights.</p>", unsafe_allow_html=True)

# Initialize session state
if 'documents_indexed' not in st.session_state:
    st.session_state.documents_indexed = False
if 'last_query' not in st.session_state:
    st.session_state.last_query = None
if 'is_loading' not in st.session_state:
    st.session_state.is_loading = False

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
    if st.button("📂 Load Sample Documents", help="Load pre-indexed sample documents"):
        with st.spinner("Loading documents..."):
            try:
                load_documents()
                st.success("✅ Documents loaded successfully!")
                st.session_state.documents_indexed = True
            except Exception as e:
                st.error(f"❌ Error loading documents: {str(e)}")
    
    # Tech Stack Information
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🔍 **Vector DB**: 
        - Qdrant
    - 🧠 **Embeddings**: 
        - bge-m3
    - 🤖 **LLM**: 
        - Arabic: gemma3:1b
        - English: phi4-mini:3.8b
    - ⚡ **Azure AI Disconnected Containers**: 
        - Language Detection 
        - Named Entity Recognition (NER) 
        - Text Analytics 
    """)
    
    # Language Support
    st.markdown("### 🌍 Language Support")
    st.markdown("""
    - **English** 🇺🇸
    - **Arabic** 🇦🇪
    """)

# Main Content

# Search Input
query = st.text_input(
    "Ask a question:",
    placeholder="e.g., What did Microsoft and OpenAI announce?",
    key="search_input"
)

@st.cache_resource
def get_ollama_model():
    """Get cached Ollama model instance."""
    return ollama

@st.cache_data
def cached_search(query: str, language: str):
    """Cache search results to prevent recomputation."""
    return search_documents(query, language)

@st.cache_data
def cached_response(query: str, results: list):
    """Cache AI responses to prevent recomputation."""
    return generate_response(query, results)

# Search Button
if st.button("🔍 Search", use_container_width=True, disabled=st.session_state.is_loading):
    if query:
        st.session_state.is_loading = True
        try:
            with st.spinner("🔍 Searching through documents..."):
                # Detect language automatically
                language = detect_language(query)
                
                # Show detected language with appropriate emoji
                lang_emoji = "🇺🇸" if language == "english" else "🇦🇪"
                lang_display = "English" if language == "english" else "Arabic"
                st.info(f"{lang_emoji} Detected Language: {lang_display}")
                
                # Search for relevant documents
                results = cached_search(query, language)
                
                if results:
                    with st.spinner("🤖 Generating response..."):
                        response = cached_response(query, results)
                        st.subheader("🤖 AI Response")
                        
                        # Add RTL support for Arabic responses
                        if language == "arabic":
                            st.markdown(f"""
                            <div dir="rtl" style="text-align: right; font-family: 'Arial', sans-serif; line-height: 1.8;">
                                {response}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.write(response)
                        
                    # Display sources below the response
                    st.subheader("📚 Sources")
                    for i, doc in enumerate(results, 1):
                        source_name = doc.get('source', 'Unknown').split('/')[-1] if doc.get('source') else 'Unknown'
                        lang_emoji = "🇺🇸" if doc.get('language') == "english" else "🇦🇪"
                        lang_display = doc.get('language', 'unknown').capitalize()
                        
                        # Add RTL support for Arabic source content
                        with st.expander(f"Source {i} (Relevance: {doc['score']:.2f})"):
                            st.write(f"**Document:** {source_name}")
                            st.write(f"**Language:** {lang_emoji} {lang_display}")
                            st.write("**Relevant Content:**")
                            
                            if doc.get('language') == "arabic":
                                st.markdown(f"""
                                <div dir="rtl" style="text-align: right; font-family: 'Arial', sans-serif; line-height: 1.8;">
                                    {doc.get('text', 'No content available')}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"```\n{doc.get('text', 'No content available')}\n```")
                                
                            if 'entities' in doc['matched_entities']:
                                st.write("**Named Entities:**")
                                for category, entities in doc['matched_entities'].items():
                                    if doc.get('language') == "arabic":
                                        st.markdown(f"""
                                        <div dir="rtl" style="text-align: right;">
                                            - {category}: {', '.join(entities)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.write(f"- {category}: {', '.join(entities)}")
                else:
                    st.warning("No relevant documents found. Try rephrasing your question or loading more documents.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            st.session_state.is_loading = False
    else:
        st.warning("Please enter a question to search.")

# Example Prompts Section at the bottom
st.markdown("---")
st.markdown("### 📝 Example Prompts for Testing")
st.code("""
# English Prompts:
What did Microsoft and OpenAI announce recently?
Tell me about AI developments in Seattle
What is Jeff Bezos's role in AI innovation?
Compare AI initiatives of different tech companies
What is the Microsoft and G42 partnership in the UAE about?

# Arabic Prompts:
ما هي مشاريع الذكاء الاصطناعي في دبي؟
ماذا أعلنت شركة مايكروسوفت في المنطقة العربية؟
ما هي خطط جيف بيزوس في أبوظبي؟
كيف تساهم الشركات التقنية في تطوير الذكاء الاصطناعي في السعودية؟
ما هي تفاصيل شراكة مايكروسوفت وG42 في الإمارات؟
""", language="markdown")