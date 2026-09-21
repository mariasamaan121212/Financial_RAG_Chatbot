import os
import tempfile
import json
import re
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
    page_title="Universal Document AI Intelligence",
    page_icon="📊",
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
</style>
""", unsafe_allow_html=True)

# Hero Header
st.markdown("""
<div class="header-title">📊 Universal AI Document Intelligence Engine</div>
<p style="color: #94A3B8; font-size: 15px;">Enterprise-grade Multi-Document Analytics powered by Groq & Dynamic Vector Search</p>
<div>
    <span class="badge-red">⚡ Groq Ultra-Fast</span>
    <span class="badge-blue">🔍 FAISS Vector Store</span>
    <span class="badge-blue">🦜🔗 Dynamic RAG</span>
    <span class="badge-red">📊 Plotly Dynamic Charts</span>
</div>
<br>
""", unsafe_allow_html=True)

# Sidebar Setup
st.sidebar.markdown("### ⚙️ System Control")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Enter your Groq API Key")
uploaded_file = st.sidebar.file_uploader("Upload Any PDF Document", type=["pdf"])

st.sidebar.markdown("---")
st.sidebar.caption("🔒 Document context is strictly bounded to prevent AI hallucinations.")

# Cache PDF Indexing Process
@st.cache_resource(show_spinner="Indexing document into vector database...")
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

# Dynamic Analysis Function for Any Uploaded Document

def generate_dynamic_insights(_vectorstore, model_name):
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke("summary key metrics statistics numbers results revenue profit performance highlights")
    context = "\n\n".join(doc.page_content for doc in docs)

    llm = ChatGroq(model=model_name, temperature=0)

    prompt = f"""
    Analyze the provided document context and produce a JSON response with two keys:
    1. "summary": List of 4 concise bullet points summarizing key findings.
    2. "metrics": List of up to 4 key numerical data points found in the document for plotting on a bar chart. Each item must be an object with "label" (string) and "value" (number float/int). Convert billion/million values to standard float numbers (e.g. 97.69).

    STRICT REQUIREMENT: Respond strictly with valid JSON only. Format:
    {{
      "summary": ["Bullet 1", "Bullet 2", "Bullet 3", "Bullet 4"],
      "metrics": [
        {{"label": "Metric Name 1", "value": 100.5}},
        {{"label": "Metric Name 2", "value": 85.2}}
      ]
    }}

    Document Context:
    {context}
    """

    res = llm.invoke(prompt)
    
    try:
        json_match = re.search(r'\{.*\}', res.content, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception:
        pass
        
    return {
        "summary": [
            "Document successfully indexed.",
            "Ready for context-aware Q&A.",
            "Ask specific questions using the chat console.",
            "Source citations available for all generated answers."
        ],
        "metrics": []
    }

# Render Dynamic Plotly Chart
def render_dynamic_chart(metrics_data):
    if not metrics_data:
        st.info("ℹ️ No specific numerical metrics auto-extracted for charting. You can query numbers directly in chat!")
        return

    labels = [m["label"] for m in metrics_data]
    values = [m["value"] for m in metrics_data]

    fig = go.Figure(data=[
        go.Bar(
            x=labels,
            y=values,
            marker_color=['#0284C7', '#E82127', '#38BDF8', '#F59E0B'][:len(values)],
            text=[f"{v}" for v in values],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="📊 Extracted Key Document Metrics",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#F8FAFC'),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320
    )
    st.plotly_chart(fig, use_container_width=True)

# Main Application Execution
if uploaded_file and groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    
    try:
        active_model = get_active_model(groq_api_key.strip())
        vectorstore, total_chunks = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Top KPI Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(label="📄 Loaded Document", value=uploaded_file.name[:18] + "...")
        with col2:
            st.metric(label="🧩 Vector Chunks", value=f"{total_chunks} Chunks")
        with col3:
            st.metric(label="🏎️ Engine Model", value=active_model.split('-')[0].upper())

        st.markdown("---")

        # Extract Dynamic Insights for ANY Uploaded File
        insights = generate_dynamic_insights(vectorstore, active_model)

        # Render Dynamic Executive Summary & Chart Expander
        with st.expander("📋 Automated Executive Highlights & Dynamic Analytics", expanded=True):
            chart_col, summary_col = st.columns([1.2, 1])
            with chart_col:
                render_dynamic_chart(insights.get("metrics", []))
            with summary_col:
                st.markdown("#### 🚀 Executive Summary")
                for bullet in insights.get("summary", []):
                    st.markdown(f"* {bullet}")

        # Universal Quick Prompt Buttons
        st.markdown("##### ⚡ Quick Document Prompts")
        btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
        selected_prompt = None

        if btn_col1.button("📋 Executive Summary"):
            selected_prompt = "Provide a comprehensive summary of the main points in this document."
        if btn_col2.button("🔑 Key Highlights"):
            selected_prompt = "What are the key findings, metrics, and highlights in this document?"
        if btn_col3.button("📊 Financial & Stat Figures"):
            selected_prompt = "List all major financial or statistical figures mentioned in this document."
        if btn_col4.button("⚠️ Risk Factors & Challenges"):
            selected_prompt = "What are the main risks, challenges, or limitations discussed in this file?"

        llm = ChatGroq(model=active_model, temperature=0)

        template = """You are an expert AI document analyst. Answer the user's question 
accurately using ONLY the provided context from the document. 
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

        # Query handling
        typed_query = st.chat_input("Ask any question about this document...")
        user_query = typed_query if typed_query else selected_prompt

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing document context..."):
                    retrieved_docs = retriever.invoke(user_query)
                    formatted_context = "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    chain = prompt | llm | StrOutputParser()
                    response = chain.invoke({"context": formatted_context, "question": user_query})
                    
                    st.markdown(response)
                    
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
    st.info("👈 Please enter your Groq API Key and upload any PDF document in the sidebar to launch the analysis console.")