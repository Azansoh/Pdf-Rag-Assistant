from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document

from dotenv import load_dotenv
load_dotenv()


docs=[
    Document(page_content="This is the first document.",metadata={"source":"AI_book"}),
    Document(page_content="This is the second document.",metadata={"source":"Datascience_book"}),
    Document(page_content="This is the third document.",metadata={"source":"DL_book"})
]





