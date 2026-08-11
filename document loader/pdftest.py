from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter


data = PyPDFLoader("document loader/study_material_for_rag.pdf")

doc = data.load()



splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(doc)

print(chunks[0].page_content)