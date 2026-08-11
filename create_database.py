from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

print("1. Loading PDF...")
loader = PyPDFLoader("document loader/Deeplearning.pdf")
doc = loader.load()

print(f"Loaded {len(doc)} pages.")

print("2. Splitting text...")
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(doc)
print(f"Created {len(chunks)} text chunks.")

print("3. Loading Embedding Model...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

print("4. Saving to Chroma DB...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"
)

print("✅ Success! Database created and saved to ./chroma_db")