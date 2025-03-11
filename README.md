# 🧠 AI-Powered RAG System 🔍  
**Deployable Offline | On-Premise | Any Cloud**  

This project showcases a **Retrieval-Augmented Generation (RAG) system** designed for **flexibility, scalability, and real-world deployment**. 

Powered by **Azure AI Containers**, it ensures **high-performance retrieval, accuracy, and security**—whether running **fully offline, on-premise, or in any cloud environment**.

## **Why This Matters?**

**Hybrid Retrieval**: Combines **Qdrant (vector search)**, **BM25 (keyword matching)**, and **Ollama embeddings** to enhance ranking and accuracy.  

**Multilingual AI**: Supports **Arabic & English** with optimized retrieval for short queries and complex prompts.  

**Azure AI Integration**: Leverages **Azure AI Containers** for ensuring **better accuracy, security, and usability**—even in offline environments.  

**On-Premise Ready**: Designed for **full offline deployment**, making it ideal for **customers needing secure, cloud-independent AI solutions**.  

---

## 🚀 How Azure AI Containers can Enhance Offline RAG System

Azure AI Containers enable **advanced AI capabilities** while keeping the system **fully offline and on-premise-ready**. 

These services enhance **document processing, query understanding, and response generation**, making the system **more accurate, secure, and scalable**.

### **1️ Improving Document Processing & Indexing**
- **Azure AI Vision - Read** → Extracts text from scanned documents & images, making PDFs and handwritten content searchable.  
- **Document Intelligence** → Processes structured documents (invoices, contracts) before indexing, improving retrieval in legal and enterprise use cases.  

### **2️ Enhancing Query Understanding & Retrieval**
-  **Azure AI Language** → Detects **query language**
- **Conversational Language Understanding (CLU)** → Classifies **query intent** (e.g., **search vs. summarization**) for smarter responses.  

### **3️ Enhancing AI-Generated Responses**
- **Sentiment Analysis** → Adjusts AI response tone (formal/casual).  
- **Text Translation** → Enables **cross-language retrieval** (Arabic query → English documents).  
- **Neural Text-to-Speech** → Converts AI responses into voice for **future chatbot integrations**.  

### **4️ Ensuring Content Safety & Compliance**  
- **Azure AI Content Safety** → Scans both **text and images** for **violence, hate speech, self-harm, and explicit content**, ensuring **AI-generated responses and retrieved documents comply with safety standards**.  

🔹 **These integrations ensure the RAG system remains fully functional in offline environments while benefiting from enterprise-grade AI.**  

📌 **Learn More**: [Azure AI Containers](https://learn.microsoft.com/en-us/azure/ai-services/cognitive-services-container-support)  
📌 **Azure AI Content Safety**: [Content Safety Containers (Preview)](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/how-to/containers/container-overview)  

---

## 🏗 Enhanced User Flow with Azure AI Containers  

| **Step**                     | **Tool Used**                 | **Description** |
|------------------------------|------------------------------|----------------|
| **1. User enters a query**    | Streamlit UI                 | Provides an input field for users to enter queries. |
| **2. Detect query language**  | **Azure AI Language (Offline Container)**  | Determines whether the query is in Arabic or English. |
| **3. Generate query embedding** | Ollama (`bge-m3`) | Converts the query into a numerical vector representation. |
| **4. Retrieve relevant documents** | Qdrant (Vector DB)       | Performs a **hybrid search**: **vector similarity search** (embeddings) + **BM25 keyword match**. |
| **5. Rank retrieved documents** | BM25 (Rank-BM25) + `bge-m3`  | Ranks results based on keyword relevance and vector similarity. |
| **6. Extract named entities (Optional)** | **Azure AI NER (Offline Container)** | Identifies key entities in the query to improve retrieval precision. |
| **7. Apply OCR for document parsing** | **Azure Document Intelligence (Offline Container)** | Extracts text from scanned PDFs, images, or structured documents to improve knowledge base ingestion. |
| **8. Generate an AI response** | Ollama (`Qwen/Gemma`) | Uses an **LLM to generate an answer** using the top-ranked documents as context. |
| **9. Apply content safety filters** | **Azure AI Content Safety (Offline Container)** | Ensures the **AI-generated response follows safety guidelines**, filtering out harmful or inappropriate content. |
| **10. Display response**       | Streamlit UI                 | Shows retrieved documents, scores, and final AI response. |

---
## 🛠️ **Setup & Installation**  

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-repo/edge-rag.git
cd edge-rag
```

### **2️⃣ Set Up Python Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate   # Windows
```

### **3️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
docker compose up -d
```

### **4️⃣ Start Qdrant (Vector Database)**
Make sure **Docker** is installed, then run:
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### **5️⃣ Install & Run Ollama**
Follow Ollama installation from [Ollama's official website](https://ollama.com). Then, pull the required models:
```bash
ollama serve 
ollama pull qwen2.5:0.5b
ollama pull gemma2:2b
ollama pull bge-m3
```

### **6️⃣ Run Azure AI Containers for Language Detection & NER**
#### **Language Detection**
```bash
docker run --rm -it --platform linux/amd64 -p 5000:5000 --memory 6g --cpus 2 \
  mcr.microsoft.com/azure-cognitive-services/textanalytics/language \
  Eula=accept \
  Billing="$AZURE_LANGUAGE_BILLING_URL" \
  ApiKey="$AZURE_LANGUAGE_API_KEY"
```

```
curl -X POST "http://localhost:5000/text/analytics/v3.1/languages" \
     -H "Content-Type: application/json" \
     -d '{
          "documents": [
            {"id": "1", "text": "Hello, how are you?"},
            {"id": "2", "text": "مرحبا كيف حالك؟"}
          ]
        }'
```

### **7️⃣ Prepare & Index Documents**
Store your dataset inside the `data/` folder, then run:
```bash
python src/index_documents.py
```

Verify if the Qdrant Collections Exist
```
curl -X GET "http://localhost:6333/collections"
```

Clean Qdrant Collections if needed. 
```
curl -X DELETE "http://localhost:6333/collections/rag_docs_en"
curl -X DELETE "http://localhost:6333/collections/rag_docs_ar"
```

### **8️⃣ Start the Streamlit UI**
```bash
streamlit run src/streamlit_app.py
```

### **9️⃣ Test Queries**
Open your browser at `http://localhost:8501` and enter any query.  
Examples:  
- **English:** `"What is artificial intelligence?"`  
- **Arabic:** `"ما هو الذكاء الاصطناعي؟"`

The system will:
1. **Detect query language using Azure AI**
2. **Perform Named Entity Recognition (NER)**
3. **Retrieve relevant documents from Qdrant**
4. **Rank results using BM25 + embedding similarity + reranking**
5. **Generate an AI response using Ollama**

---

## 📌 **Addressing Arabic Language Challenges**

- 1️⃣ Challenge: Arabic Ranker Models**

📌 **Problem:** Many ranker models struggle to reconstruct answers when the supporting information is scattered across multiple chunks.  
✅ **Solution:** We integrate **BM25 + bge-m3 reranker**, which improves the ranking of relevant Arabic documents based on **semantic similarity and keyword matching**.

-  2️⃣ Challenge: Arabic Embedding Models**

📌 **Problem:** Single-word Arabic queries sometimes fail to retrieve results, even when relevant content exists in the knowledge base.  
✅ **Solution:** We use a **hybrid search approach**, combining:
   - **Vector search (Ollama embeddings)**
   - **BM25 keyword matching**
   - **Reranking using bge-m3**
   This ensures better retrieval even for **short Arabic queries**.


- 3️⃣ Mitigating those Issues** 

📌 Our current implementation mitigates these issues with:
  - Hybrid Search (BM25 + Vectors)
  - Re-ranking (bge-m3)
  - Named Entity Recognition (NER)
  - LLM Context Expansion

- 4️⃣ Future Improvements

🟢 **Experiment with specialized Arabic embedding models** (e.g., Arabic-trained versions of BGE or ARABERT).  
🟢 **Optimize BM25 weights for Arabic vs. English separately** to fine-tune ranking balance.  
🟢 **Extend Named Entity Recognition (NER) to improve keyword-based lookup**.  
🟢 **Benchmark different Arabic language models for better retrieval performance**.

---

Enjoy building your **production-ready RAG system**! 🏗️🔥  
