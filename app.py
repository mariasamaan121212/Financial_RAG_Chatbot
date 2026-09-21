import os
import tempfile
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page Configuration
st.set_page_config(
    page_title="Tesla Financial AI Intelligence",
    page_icon="🚘",
    layout="wide"
)

# Custom Tesla Red & Tech Blue CSS Styling
st.markdown("""
<style>
    /* Metric Cards Styling with Red & Blue Glow */
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        color: #E82127 !important; /* Tesla Red */
        font-weight: bold;
    }
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        padding: 14px 20px;
        border-radius: 12px;
        border-left: 5px solid #E82127; /* Tesla Red Accent Line */
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    /* Custom Badges */
    .badge-red {
        background-color: #E82127;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 700;
        margin-right: 6px;
    }
    .badge-blue {
        background-color: #0284C7;
        color: white;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    
    /* Header Gradient Text */
    .header-title {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #E82127, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="header-title">🚘 Tesla Financial AI RAG Intelligence</div>
<p style="color: #94A3B8; font-size: 15px;">Enterprise-grade Financial Document Analysis powered by Groq & Vector Search</p>
<div>
    <span class="badge-red">⚡ Groq Ultra-Fast</span>
    <span class="badge-blue">🔍 FAISS Vector Store</span>
    <span class="badge-blue">🦜🔗 LangChain LCEL</span>
</div>
<br>
""", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.markdown("### ⚙️ System Control")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API Key")
uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Document context is strictly bounded to prevent AI hallucinations.")

# Cache PDF Indexing Process
@st.cache_resource(show_spinner="Indexing financial document into vector database...")
def process_pdf(file_bytes, filename):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    os.remove(tmp_path)
    return vectorstore, len(docs)

# Safe Dynamic Model Discovery
def get_active_model(api_key):
    client = Groq(api_key=api_key)
    all_models = [m.id for m in client.models.list().data]
    
    preferred_keywords = ["gpt-oss", "qwen", "llama-3.3", "llama-3.1", "compound", "mixtral"]
    for keyword in preferred_keywords:
        for m in all_models:
            if keyword in m.lower():
                return m
                
    text_models = [
        m for m in all_models 
        if not any(x in m.lower() for x in ["whisper", "guard", "canopylabs", "orpheus", "allam"])
    ]
    return text_models[0] if text_models else "llama-3.3-70b-versatile"

# Chat History State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Application Core Logic
if uploaded_file and groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    
    try:
        active_model = get_active_model(groq_api_key.strip())
        vectorstore, total_chunks = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # KPI Dashboard Cards Top Bar
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📄 Loaded Document", value=uploaded_file.name[:18] + "...")
        with col2:
            st.metric(label="🧩 Vector Chunks", value=f"{total_chunks} Chunks")
        with col3:
            st.metric(label="🏎️ Engine Model", value=active_model.split('-')[0].upper())

        st.markdown("---")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        llm = ChatGroq(model=active_model, temperature=0)

        template = """You are an expert AI financial analyst. Answer the user's question 
accurately using ONLY the provided context from the financial document. 
If the answer is not contained within the context, clearly state that 
the information is not available in the document.

Context:
{context}

Question: {question}
Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # Render History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Input Prompt
        if user_query := st.chat_input("Ask any financial question about this document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing metrics..."):
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("👈 Please enter your Groq API Key and upload a PDF document in the sidebar to launch the analysis console.")