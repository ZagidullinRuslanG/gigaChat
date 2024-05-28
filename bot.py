import asyncio
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.utils.markdown import hbold

# Инициализация
from langchain_community.chat_models.gigachat import GigaChat
# Загрузка документов
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter
)
# Эмбеддинги
from chromadb.config import Settings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.gigachat import GigaChatEmbeddings

token = 'NmNhYjhmZGEtNmFmNi00MDAwLTg5NWMtNWRiZDJjYjRjN2E1OjUxMzM1Yzk1LWUzY2EtNDJjZC1iMTRjLWJlMzllM2JjYTYyYQ=='

llm = GigaChat(
    credentials=token,
    scope="GIGACHAT_API_CORP",
    verify_ssl_certs=False)

loader = Docx2txtLoader("C:\\Users\\zagid\\Downloads\\bzd.docx")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")

embeddings = GigaChatEmbeddings(
    one_by_one_mode=True,
    _debug_delay=0.005,
    credentials=token,
    scope="GIGACHAT_API_CORP",
    verify_ssl_certs=False
)

db = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=True),
)

# Включаем логирование, чтобы не пропустить важные сообщения
logging.basicConfig(level=logging.INFO)
# Объект бота
bot = Bot(token="6927307400:AAEvTaaQBDiouhNEiveDBkjcccLnPssUU8Q")
# Диспетчер
dp = Dispatcher()

# Хэндлер на команду /start
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    await message.answer(f"Hello, {hbold(message.from_user.full_name)}!")

@dp.message()
async def giga_handler(message: types.Message) -> None:
    try:
        question = message.text

        # Поиск по базе данных
        docs = db.similarity_search(question, k=4)
        print(f"top docs: {len(docs)}")

        # QnA цепочка
        from langchain.chains import RetrievalQA

        qa_chain = RetrievalQA.from_chain_type(llm, retriever=db.as_retriever())
        answer = qa_chain({"query": question})
        print(answer)

        await message.answer(answer['result'])
        # await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        await message.answer("Что-то пошло не так!")

# Запуск процесса поллинга новых апдейтов
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())