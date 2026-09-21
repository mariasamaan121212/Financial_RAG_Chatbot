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
    page_title="Financial AI Assistant",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart RAG Financial Document Chatbot")
st.caption("Powered by Groq, LangChain (LCEL), and FAISS Vector Database")

# Sidebar Configuration
st.sidebar.header("Configuration")
groq_api_key = st.sidebar.text_input("Groq API Key", type="password")
uploaded_file = st.sidebar.file_uploader("Upload Financial PDF", type=["pdf"])

# Cache Embeddings & Vector Store Initialization
@st.cache_resource(show_spinner="Processing document and building vector database...")
def process_pdf(file_bytes, filename):
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(file_bytes)
        tmp_path = tmp_file.name

    # Load PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    # Split Document
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(pages)

    # Generate Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Cleanup temp file
    os.remove(tmp_path)
    return vectorstore

# Function to get available active model
# Function to get available active LLM model safely
def get_active_model(api_key):
    client = Groq(api_key=api_key)
    all_models = [m.id for m in client.models.list().data]
    
    # Preferred production text LLM models
    preferred_keywords = ["gpt-oss", "qwen", "llama-3.3", "llama-3.1", "compound", "mixtral"]
    
    for keyword in preferred_keywords:
        for m in all_models:
            if keyword in m.lower():
                return m
                
    # Fallback filtering to exclude audio/guard/special terms models
    text_models = [
        m for m in all_models 
        if not any(x in m.lower() for x in ["whisper", "guard", "canopylabs", "orpheus", "allam"])
    ]
    return text_models[0] if text_models else "llama-3.3-70b-versatile"

# Chat History Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main Chat Logic
if uploaded_file and groq_api_key:
    os.environ["GROQ_API_KEY"] = groq_api_key.strip()
    
    try:
        # Build or fetch Vector Store
        vectorstore = process_pdf(uploaded_file.getvalue(), uploaded_file.name)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Helper function for formatting context
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        # Setup Model and Prompt
        active_model = get_active_model(groq_api_key.strip())
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

        # Build LCEL Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        # User Query Input
        if user_query := st.chat_input("Ask any financial question about this document..."):
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Analyzing document..."):
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

else:
    st.info("👈 Please enter your Groq API Key and upload a PDF document in the sidebar to begin.")