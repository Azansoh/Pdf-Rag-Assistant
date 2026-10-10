import os
import hashlib
import tempfile

import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📚",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #f7f9fc;
    }

    .main-header {
        padding: 25px 0 10px 0;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .main-subtitle {
        font-size: 18px;
        color: #6b7280;
        margin-bottom: 25px;
    }

    .info-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin-bottom: 15px;
    }

    .user-message {
        background: #e8f0fe;
        padding: 14px 18px;
        border-radius: 12px;
        margin: 10px 0;
    }

    .ai-message {
        background: white;
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        margin: 10px 0;
    }

    .source-box {
        background: #f9fafb;
        border-left: 4px solid #4f46e5;
        padding: 10px 15px;
        margin: 8px 0;
        border-radius: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

if "uploaded_file_hash" not in st.session_state:
    st.session_state.uploaded_file_hash = None

if "active_chroma_directory" not in st.session_state:
    st.session_state.active_chroma_directory = None


# ============================================================
# LOAD EMBEDDING MODEL ONCE
# ============================================================

@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ============================================================
# LOAD GROQ MODEL ONCE
# ============================================================

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-title">📚 PDF RAG Assistant</div>
        <div class="main-subtitle">
            Upload any PDF and ask questions using Retrieval-Augmented Generation.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Upload Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload any PDF document that you want to chat with."
    )

    st.divider()

    st.subheader("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk Size",
        min_value=500,
        max_value=2000,
        value=1000,
        step=100
    )

    chunk_overlap = st.slider(
        "Chunk Overlap",
        min_value=50,
        max_value=500,
        value=200,
        step=50
    )

    top_k = st.slider(
        "Retrieved Documents",
        min_value=1,
        max_value=10,
        value=4
    )

    st.divider()

    if st.session_state.uploaded_file_name:
        st.success(
            f"Current document:\n\n"
            f"**{st.session_state.uploaded_file_name}**"
        )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.success("Chat history cleared.")
        st.rerun()


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    is_new_document = (
        file_hash != st.session_state.uploaded_file_hash
    )

    if is_new_document:

        temp_dir = tempfile.mkdtemp()

        try:

            with st.spinner("📚 Processing your PDF..."):

                pdf_path = os.path.join(
                    temp_dir,
                    uploaded_file.name
                )

                with open(pdf_path, "wb") as f:
                    f.write(file_bytes)

                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                if not documents:
                    raise ValueError(
                        "The PDF does not contain readable content."
                    )

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                chunks = splitter.split_documents(documents)

                if not chunks:
                    raise ValueError(
                        "No text chunks could be created from the PDF."
                    )

                embedding_model = get_embedding_model()

                chroma_root = "./chroma_db"

                os.makedirs(
                    chroma_root,
                    exist_ok=True
                )

                chroma_directory = os.path.join(
                    chroma_root,
                    file_hash
                )

                if os.path.exists(chroma_directory):

                    vectorstore = Chroma(
                        persist_directory=chroma_directory,
                        embedding_function=embedding_model
                    )

                else:

                    vectorstore = Chroma.from_documents(
                        documents=chunks,
                        embedding=embedding_model,
                        persist_directory=chroma_directory
                    )

                retriever = vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": top_k,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                st.session_state.vectorstore = vectorstore
                st.session_state.retriever = retriever
                st.session_state.uploaded_file_name = uploaded_file.name
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.active_chroma_directory = chroma_directory
                st.session_state.chat_history = []

            st.success(
                f"✅ {uploaded_file.name} processed successfully!"
            )

            st.info(
                f"📄 Pages: {len(documents)}  |  "
                f"🧩 Chunks: {len(chunks)}"
            )

        except Exception as e:

            st.error(
                f"❌ Error while processing PDF:\n\n{str(e)}"
            )

        finally:

            try:
                if os.path.exists(temp_dir):
                    import shutil
                    shutil.rmtree(temp_dir)
            except Exception:
                pass


# ============================================================
# MAIN CONTENT
# ============================================================

if st.session_state.retriever is None:

    st.info(
        "👈 Upload a PDF from the sidebar to start chatting."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            ### 📤 Upload

            Upload any PDF document from your computer.
            """
        )

    with col2:
        st.markdown(
            """
            ### 🔎 Retrieve

            Relevant document chunks are retrieved from ChromaDB.
            """
        )

    with col3:
        st.markdown(
            """
            ### 🤖 Ask

            Groq AI generates an answer using the retrieved context.
            """
        )

else:

    st.markdown(
        f"""
        <div class="info-card">
            <strong>📄 Active Document:</strong>
            {st.session_state.uploaded_file_name}
        </div>
        """,
        unsafe_allow_html=True
    )

    for message in st.session_state.chat_history:

        if message["role"] == "user":

            st.markdown(
                f"""
                <div class="user-message">
                    <strong>👤 You</strong><br><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="ai-message">
                    <strong>🤖 AI Assistant</strong><br><br>
                    {message["content"]}
                </div>
                """,
                unsafe_allow_html=True
            )

            if message.get("sources"):

                with st.expander("📚 View Sources"):

                    for source in message["sources"]:

                        st.markdown(
                            f"""
                            <div class="source-box">
                                📄 Page {source["page"]}<br>
                                <small>
                                {source["content"]}
                                </small>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    query = st.chat_input(
        "Ask a question about your PDF..."
    )

    if query:

        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        try:

            with st.spinner("🔍 Searching the document..."):

                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": top_k,
                        "fetch_k": 10,
                        "lambda_mult": 0.5
                    }
                )

                docs = retriever.invoke(query)

            context = "\n\n".join(
                [doc.page_content for doc in docs]
            )

            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
                        You are a helpful PDF question-answering assistant.

                        Answer the user's question using ONLY the
                        provided context.

                        If the answer is not present in the context,
                        say exactly:

                        "I couldn't find the answer in the document."

                        Do not make up information.
                        """
                    ),
                    (
                        "human",
                        """
                        Context:

                        {context}

                        Question:

                        {question}
                        """
                    )
                ]
            )

            llm = get_llm()

            final_prompt = prompt.invoke(
                {
                    "context": context,
                    "question": query
                }
            )

            with st.spinner("🤖 Generating answer..."):

                response = llm.invoke(final_prompt)
                answer = response.content

            sources = []

            for doc in docs:

                page_number = (
                    doc.metadata.get("page", 0) + 1
                )

                sources.append(
                    {
                        "page": page_number,
                        "content": doc.page_content[:500]
                    }
                )

            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                }
            )

            st.rerun()

        except Exception as e:

            st.error(
                f"❌ Error generating response:\n\n{str(e)}"
            )