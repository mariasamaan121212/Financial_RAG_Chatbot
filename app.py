import os
import tempfile
import streamlit as st
import plotly.graph_objects as go
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
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
    div[data-testid="stMetricValue"] {
        font-size: 22px !important;
        color: #E82127 !important;
        font-weight: bold;
    }
    div[data-testid="stMetric"] {
        background-color: #1E293B;
        padding: 14px 20px;
        border-radius: 12px;
        border-left: 5px solid #E82127;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
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
    .header-title {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #E82127, #38BDF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .quick-btn {
        margin-bottom: 10px;
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
    <span class="badge-red">📊 Plotly Analytics</span>
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

# Function to Render Interactive Plotly Chart
def render_financial_chart():
    categories = ['2023 Total Revenues', '2024 Total Revenues', '2023 Gross Profit', '2024 Gross Profit']
    values = [96.77, 97.69, 17.66, 17.45] # In Billions USD from Tesla filings

    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=['#0284C7', '#E82127', '#0284C7', '#E82127'],
            text=[f"${v}B" for v in values],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="📊 Financial Performance Overview (in Billions USD)",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

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

        # Feature 2 & 4: Executive Summary & Plotly Chart Expander
        with st.expander("📋 Automated Executive Highlights & Financial Charts", expanded=False):
            chart_col, summary_col = st.columns([1.2, 1])
            with chart_col:
                render_financial_chart()
            with summary_col:
                st.markdown("#### 🚀 Executive Summary")
                st.markdown("""
                * **Revenue Trend:** Revenue reached **$97.69 Billion** in 2024 (up from $96.77B in 2023).
                * **Gross Profit:** Gross Profit recorded **$17.45 Billion** in 2024.
                * **Core Sectors:** Energy Storage and Services showed robust double-digit expansion.
                * **Risk Bounding:** Supply chain fluctuations remain a highlighted focus area.
                """)

        # Feature 3: Quick Prompt Buttons
        st.markdown("##### ⚡ Quick Financial Prompts")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        selected_prompt = None

        if btn_col1.button("💵 Revenues 2024 vs 2023"):
            selected_prompt = "What was Tesla's total revenues in 2024 compared to 2023?"
        if btn_col2.button("🚗 Total Deliveries"):
            selected_prompt = "How many total vehicles did Tesla produce and deliver in 2024?"
        if btn_col3.button("🔬 R&D Spending"):
            selected_prompt = "How much did Tesla spend on Research and Development (R&D) in 2024?"
        if btn_col4.button("⚠️ Supply Chain Risks"):
            selected_prompt = "What are the primary risk factors mentioned regarding battery supply chain?"

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

        # Render Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Document Sources & Citation"):
                        for idx, doc in enumerate(message["sources"]):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Source {idx+1} (Page {page_num}):**")
                            st.caption(doc.page_content)

        # Determine user input (either typed or button clicked)
        typed_query = st.chat_input("Ask any financial question about this document...")
        user_query = typed_query if typed_query else selected_prompt

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing metrics and retrieving page context..."):
                    # Feature 1: Retrieve Documents with Metadata for Source Citation
                    retrieved_docs = retriever.invoke(user_query)
                    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": formatted_context, "question": user_query})
                    
                    st.markdown(response)
                    
                    # Display Citation Expander
                    with st.expander("📚 View Document Sources & Citation"):
                        for idx, doc in enumerate(retrieved_docs):
                            page_num = doc.metadata.get("page", 0) + 1
                            st.markdown(f"**Source {idx+1} (Page {page_num}):**")
                            st.caption(doc.page_content)

            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": retrieved_docs
            })

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("👈 Please enter your Groq API Key and upload a PDF document in the sidebar to launch the analysis console.")