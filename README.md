# 📊 Financial AI RAG Chatbot

An industry-standard, high-performance Financial RAG (Retrieval-Augmented Generation) Assistant designed to analyze complex financial reports (e.g., Tesla 10-K filings) and provide accurate, context-grounded answers in real-time.

---

## 🌟 Key Features

- **Document Agnostic:** Upload any financial PDF report and analyze it dynamically on the fly.
- **Strict Hallucination Prevention:** Powered by a financial-analyst system prompt that bounds answers strictly to the uploaded document context.
- **Dynamic API Model Discovery:** Automatically queries the Groq API for available production LLMs, ensuring zero downtime from model deprecations.
- **Optimized Caching:** Utilizes Streamlit's `@st.cache_resource` to process and index documents into FAISS vector database only once per upload.
- **Modern LCEL Architecture:** Built entirely using LangChain Expression Language (LCEL) for high execution speed and clean modularity.

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Orchestration:** [LangChain](https://www.langchain.com/) (LCEL)
- **LLM Inference:** [Groq API](https://groq.com/) (Ultra-low latency inference)
- **Embeddings Model:** [HuggingFace](https://huggingface.co/) (`sentence-transformers/all-MiniLM-L6-v2`)
- **Vector Database:** [FAISS](https://github.com/facebookresearch/faiss) (Facebook AI Similarity Search)
- **PDF Parser:** `PyPDF`

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have Python 3.10+ installed on your system.

### 2. Installation

Clone this repository and navigate to the project directory:
```bash
git clone [https://github.com/YOUR_GITHUB_USERNAME/Financial_RAG_Chatbot.git](https://github.com/YOUR_GITHUB_USERNAME/Financial_RAG_Chatbot.git)
cd Financial_RAG_Chatbot