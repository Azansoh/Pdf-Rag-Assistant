import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

print("Loading saved database...")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
)

# FIXED: Use Groq LLM with Llama 3 model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

# Prompt Template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use ONLY the provided context to answer the question. If the answer is not in the context, say 'I couldn't find the answer in the document.'"),
    ("human",
     """
     Context: 
     {context}

     Question:
     {question}
     """
    )
])

while True:
    query = input("\nEnter your question (or press 0 to exit): ")
    if query == '0':
        print("Exiting the program.")
        break

    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.invoke({"context": context, "question": query})

    response = llm.invoke(final_prompt)
    print(f"\nAI: {response.content}\n")