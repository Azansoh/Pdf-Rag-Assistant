📚 PDF RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows usersto upload PDF documents and ask questions about their content through aninteractive Streamlit chat interface.

The application uses Mistral AI for answer generation, HuggingFace Sentence Transformers for embeddings, and ChromaDB as thevector database for semantic document retrieval.

🚀 Project Overview

The PDF RAG Assistant follows a complete RAG pipeline:

                    ┌──────────────────────┐
                    │     Upload PDF       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Load PDF          │
                    │    PyPDFLoader        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Split Documents    │
                    │ RecursiveCharacter   │
                    │    TextSplitter      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Generate Embeddings  │
                    │ all-MiniLM-L6-v2     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      ChromaDB        │
                    │   Vector Database    │
                    └──────────┬───────────┘
                               │
                         User Question
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Retriever       │
                    │       MMR Search     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Mistral AI       │
                    │ mistral-small-latest│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Generated Answer   │
                    └──────────────────────┘

✨ Features

📄 Upload any PDF document

🔍 Ask questions about the uploaded document

🤖 Generate answers using Mistral AI

🧠 Semantic search using vector embeddings

🗃️ ChromaDB vector database

🔗 Retrieval-Augmented Generation architecture

💬 Interactive Streamlit chat interface

📚 Display retrieved source pages

⚙️ Adjustable chunk size

⚙️ Adjustable chunk overlap

🎯 Adjustable number of retrieved documents (Top K)

🔄 Supports switching between different PDF documents

🛡️ Answers are restricted to the retrieved document context

🚫 Avoids making up information when the answer is not present inthe document

💾 Creates a separate ChromaDB directory for each different PDF

🔐 Uses environment variables for the Mistral API key

🛠️ Technologies Used

Technology                  Purpose

Python                  Core programming languageStreamlit               Web-based user interfaceLangChain               RAG application frameworkMistral AI              Large Language Model for answer generationChromaDB                Vector databaseHugging Face            Embedding model integrationSentence Transformers   Text embeddingsPyPDF                   PDF document loadingpython-dotenv           Environment variable management

🧠 Models Used

LLM

The application uses:

mistral-small-latest

Mistral receives the user's question together with the relevantretrieved document context and generates the final answer.

Embedding Model

The project uses:

sentence-transformers/all-MiniLM-L6-v2

This model converts document chunks and queries into vectorrepresentations that can be compared semantically.

📂 Project Structure

A typical project structure is:

PDF-RAG-Assistant/
│
├── UImain.py
├── create_database.py
├── requirements.txt
├── .env
├── README.md
│
└── chroma_db/
    ├── <pdf-hash-1>/
    ├── <pdf-hash-2>/
    └── ...

Main Files

UImain.py

The main Streamlit application.

It handles:

PDF uploading

PDF processing

Text splitting

Embedding generation

ChromaDB creation/loading

Document retrieval

Mistral AI response generation

Chat history

Source display

create_database.py

The original standalone script used to load a PDF, split it into chunks,generate embeddings, and create a ChromaDB database.

The Streamlit version performs this process automatically when a PDF isuploaded, so create_database.py is not required to run the UIapplication.

requirements.txt

Contains the Python dependencies required by the project.

.env

Stores the Mistral API key securely.

chroma_db/

Stores the generated vector database data.

🔄 How the RAG Pipeline Works

1. Upload PDF

The user uploads a PDF through Streamlit.

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

2. Load PDF

The uploaded document is temporarily saved and loaded usingPyPDFLoader.

3. Split Text

Large documents are divided into smaller chunks using:

RecursiveCharacterTextSplitter

Default settings:

Chunk Size:     1000
Chunk Overlap:  200

The UI allows these values to be adjusted.

4. Generate Embeddings

Each document chunk is converted into a numerical vector using:

sentence-transformers/all-MiniLM-L6-v2

5. Store in ChromaDB

The embeddings and document chunks are stored in ChromaDB.

Each different PDF receives its own database directory based on a hashof its content.

This prevents conflicts when switching from one PDF to another.

6. Retrieve Relevant Information

When the user asks a question, the application retrieves the mostrelevant chunks from ChromaDB.

The project uses MMR (Maximal Marginal Relevance) retrieval.

Default settings:

k = 4
fetch_k = 10
lambda_mult = 0.5

7. Build Context

The retrieved chunks are combined into a context:

Context:
<retrieved document content>

Question:
<user question>

8. Generate Answer

The context and question are passed to Mistral AI.

The system prompt instructs the model to use only the provided context.

If the answer is not present in the retrieved document content, theapplication asks the model to respond:

I couldn't find the answer in the document.

9. Display Sources

The application also displays the pages and excerpts used as retrievalsources.

🎨 Streamlit Interface

The interface contains:

Sidebar

PDF uploader

Chunk Size control

Chunk Overlap control

Retrieved Documents (Top K) control

Clear Chat button

Current document information

Main Area

Application title

Active document

Chat history

User questions

AI answers

Retrieved source information

PDF question input

⚙️ Installation

1. Clone the Repository

git clone <your-repository-url>

Move into the project directory:

cd PDF-RAG-Assistant

2. Create a Virtual Environment

Windows:

python -m venv venv

Activate it:

venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt

🔑 Environment Variables

Create a .env file in the project root:

MISTRAL_API_KEY=your_mistral_api_key_here

Do not commit your .env file to GitHub.

Add this to .gitignore:

.env
venv/
__pycache__/
chroma_db/

▶️ Run the Application

Start the Streamlit application with:

streamlit run UImain.py

The terminal will provide a local URL similar to:

http://localhost:8501

Open the URL in your browser.

💬 Example Questions

After uploading a PDF, you can ask questions such as:

What is this document about?

What are the main concepts discussed in the document?

Explain the definition given in the document.

What are the advantages mentioned in the document?

Summarize the second page.

What does the document say about artificial intelligence?

🧪 Example RAG Test

For example, if the uploaded PDF contains information about ArtificialIntelligence, you could ask:

Question:

What is Artificial Intelligence?

The system:

Converts the question into an embedding.

Searches ChromaDB.

Retrieves relevant chunks.

Places those chunks into the prompt.

Sends the prompt to Mistral AI.

Generates an answer.

Displays the answer and retrieved sources.

📌 Important Design Decision: Multiple PDFs

The application is designed so that different PDFs do not overwrite theactive ChromaDB data.

Instead, a hash is generated from the uploaded PDF:

PDF
 ↓
MD5 Hash
 ↓
Unique ChromaDB Directory

Example:

chroma_db/
│
├── 4f8c2.../
│   └── Chroma data for PDF 1
│
└── 91a73.../
    └── Chroma data for PDF 2

This approach also avoids the Windows WinError 32 problem that canoccur when attempting to delete ChromaDB files while they are stillbeing used by the running application.

🔐 Security

The Mistral API key should be stored in .env rather than directlyinside the Python source code.

Example:

MISTRAL_API_KEY=your_mistral_api_key_here

Never upload your .env file to a public GitHub repository.

If an API key is accidentally exposed, revoke it and generate a new one.

🚫 Limitations

The application currently focuses on one active PDF at a time.

The quality of answers depends on the quality and text extraction ofthe uploaded PDF.

Scanned PDFs containing images without an OCR text layer may notproduce useful text.

Very large PDFs may require more processing time and memory.

The model is instructed to answer from retrieved context, soinformation that is not successfully retrieved may not be availableto the model.

🔮 Future Improvements

Possible future features include:

📚 Multiple PDFs in a single knowledge base

🔎 Advanced metadata filtering

🧾 Better source citations

📑 Page previews

🖼️ OCR support for scanned PDFs

🧠 Conversation-aware retrieval

💾 Persistent chat history

👥 User authentication

☁️ Cloud vector databases

🌐 Deployment to Streamlit Community Cloud

📊 RAG evaluation and retrieval metrics

📝 Conversation export

🗂️ Document management dashboard

📖 Learning Objectives

This project demonstrates practical implementation of:

Retrieval-Augmented Generation

Large Language Models

Vector databases

Semantic search

Text embeddings

Document chunking

Prompt engineering

LangChain

Mistral AI

ChromaDB

Streamlit

PDF processing

Environment variable management

👨‍💻 Author

Muhammad Azan

A Generative AI / RAG project built to understand the complete workflowof connecting documents, embeddings, vector databases, retrieval, andLLM-based answer generation.

⭐ Project Summary

PDF RAG Assistant transforms static PDF documents into aninteractive question-answering system.

Instead of manually searching through a long document, users can uploada PDF and ask natural-language questions. The application retrievesrelevant information from the document and uses Mistral AI to generatean answer based on that retrieved context.

PDF
 ↓
Document Loader
 ↓
Text Splitter
 ↓
Embeddings
 ↓
ChromaDB
 ↓
Retriever
 ↓
Relevant Context
 ↓
Mistral AI
 ↓
Answer + Sources

If you find this project useful, consider giving the repository a ⭐.