# Инициализация
from langchain_community.chat_models.gigachat import GigaChat
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# устанавливаем локаль
import locale

locale.setlocale(locale.LC_ALL , ('ru', 'utf-8'))
print(locale.getlocale())

data_file = "./data/bzd.docx"

load_dotenv()
token = os.getenv("token")

llm = GigaChat(
    credentials=token ,
    scope="GIGACHAT_API_PERS" ,
    verify_ssl_certs=False)

question = "Как создать акт работ"
print(f"Запрос пользователя: {question}")
# print(llm([HumanMessage(content=question)]).content[0:600])

# Загрузка документов
from langchain_community.document_loaders import Docx2txtLoader , TextLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)

loader = Docx2txtLoader(data_file)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000 ,
    chunk_overlap=200 ,
)
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")

# Эмбеддинги
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GigaChatEmbeddings

embeddings = GigaChatEmbeddings(
    one_by_one_mode=True ,
    _debug_delay=0.005 ,
    credentials=token ,
    scope="GIGACHAT_API_PERS" ,
    verify_ssl_certs=False
)

db = Chroma.from_documents(
    documents ,
    embeddings ,
    client_settings=Settings(anonymized_telemetry=True) ,
)

while (True):
    question = input("User: ")

    # Поиск по базе данных
    docs = db.similarity_search(question , k=4)
    print(f"top docs: {len(docs)}")

    # QnA цепочка
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm , retriever=db.as_retriever())
    print(qa_chain({"query": question}))
