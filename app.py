from fastapi import FastAPI, HTTPException, BackgroundTasks
import aiohttp
import asyncio
import os
import re
import logging
import json
import time
import hashlib
import feedparser
from dotenv import load_dotenv
from yandex_cloud_ml_sdk import YCloudML
from concurrent.futures import ThreadPoolExecutor

# Настройка логгера
logging.basicConfig(
    filename='bot.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
load_dotenv()

app = FastAPI()

executor = ThreadPoolExecutor(max_workers=30)
session = None

YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_IAM_TOKEN = os.getenv("YC_IAM_TOKEN")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")

class TimedCache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self._cache = {}

    def get(self, key):
        entry = self._cache.get(key)
        if entry and (time.time() - entry['timestamp']) < self.ttl:
            return entry['data']
        return None

    def set(self, key, data):
        self._cache[key] = {'timestamp': time.time(), 'data': data}

request_cache = TimedCache()

@app.on_event("startup")
async def startup_event():
    global session
    session = aiohttp.ClientSession()
    if not all([YC_FOLDER_ID, YC_IAM_TOKEN]):
        raise RuntimeError("Yandex Cloud credentials missing in .env")

@app.on_event("shutdown")
async def shutdown_event():
    await session.close()

sdk = YCloudML(folder_id=YC_FOLDER_ID, auth=YC_IAM_TOKEN)
model = sdk.models.completions("yandexgpt")

def generate_cache_key(question: str) -> str:
    return hashlib.sha256(question.encode()).hexdigest()

async def ask_yandex_llm(question: str) -> tuple:
    """Исправленный запрос с обработкой JSON ошибок"""
    cache_key = generate_cache_key(question)
    if cached := request_cache.get(cache_key):
        return cached

    system_prompt = """Ты эксперт по Университету ИТМО. Отвечай ТОЛЬКО в формате JSON:
    {"answer": номер_ответа | null, "reasoning": "объяснение с источником"}"""

    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            lambda: model.configure(
                temperature=0.1,
                max_tokens=150
            ).run_deferred([
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": question}
            ]).wait()
        )

        if not result.alternatives:
            raise ValueError("Empty response from model")

        response_text = result.alternatives[0].text.strip()
        
        # Улучшенный поиск JSON
        json_match = re.search(r'(?s)\{.*\}', response_text)
        if not json_match:
            raise json.JSONDecodeError("JSON not found", doc=response_text, pos=0)

        answer_data = json.loads(json_match.group())
        
        # Валидация структуры
        if "answer" not in answer_data or "reasoning" not in answer_data:
            raise KeyError("Invalid JSON structure")

        result = (answer_data.get('answer'), answer_data.get('reasoning', ''))
        request_cache.set(cache_key, result)
        return result

    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"JSON Error: {str(e)} | Response: {response_text}", exc_info=True)
        return None, "Ошибка формата ответа"
    except Exception as e:
        logging.error(f"LLM Error: {str(e)}", exc_info=True)
        return None, "Ошибка обработки запроса"

async def search_web(query: str) -> list:
    cache_key = f"search_{generate_cache_key(query)}"
    if cached := request_cache.get(cache_key):
        return cached

    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": f"{query} site:itmo.ru",
                "key": os.getenv("GOOGLE_API_KEY"),
                "cx": SEARCH_ENGINE_ID,
                "num": 3
            },
            timeout=8
        ) as response:
            data = await response.json()
            links = [item["link"] for item in data.get("items", [])]
            request_cache.set(cache_key, links)
            return links
    except Exception as e:
        logging.error(f"Search Error: {str(e)}")
        return []

async def get_relevant_news() -> list:
    """Получение новостей с улучшенной обработкой ошибок и таймаутами"""
    cache_key = "news_cache"
    if cached := request_cache.get(cache_key):
        return cached

    try:
        async with session.get("https://news.itmo.ru/ru/rss/", timeout=10) as response:
            response.raise_for_status()  # Проверка HTTP-статуса
            text = await response.text()
            
            loop = asyncio.get_event_loop()
            feed = await loop.run_in_executor(
                executor,
                lambda: feedparser.parse(text)
            )
            
            if feed.get('bozo', 0) == 1:
                error = feed.bozo_exception
                raise ValueError(f"RSS parse error: {error}")
                
            entries = getattr(feed, 'entries', [])[:2]
            news = [entry.link for entry in entries if hasattr(entry, 'link')]
            
            if len(entries) > len(news):
                logging.warning(f"Missing links in {len(entries)-len(news)} news entries")
            
            request_cache.set(cache_key, news)
            return news

    except asyncio.TimeoutError:
        logging.warning("News fetch timeout")
        return []
    except aiohttp.ClientError as e:
        logging.error(f"Connection error: {str(e)}")
        return []
    except Exception as e:
        logging.error(f"News Error: {str(e)}", exc_info=True)
        return []

@app.post("/api/request")
async def handle_request(data: dict, background_tasks: BackgroundTasks):
    start_time = time.time()
    
    try:
        query = data["query"]
        request_id = data["id"]
    except KeyError:
        raise HTTPException(status_code=400, detail="Invalid request format")

    try:
        llm_task = asyncio.create_task(ask_yandex_llm(query))
        print('готово')
        search_task = asyncio.create_task(search_web(query))
        news_task = asyncio.create_task(get_relevant_news())

        answer, reasoning = await asyncio.wait_for(llm_task, timeout=30)
        search_results = await asyncio.wait_for(search_task, timeout=10)
        news_results = await asyncio.wait_for(news_task, timeout=10)

        has_options = re.search(r'\n\s*\d+[\.\)]', query) is not None

        background_tasks.add_task(
            logging.info,
            f"Processed request {request_id} in {time.time() - start_time:.2f}s"
        )

        return {
            "id": request_id,
            "answer": answer if has_options else None,
            "reasoning": reasoning,
            "sources": (search_results)[:3]
        }

    except asyncio.TimeoutError:
        logging.error("Request processing timeout")
        raise HTTPException(status_code=504, detail="Request timeout")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
import asyncio
import json
import re
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from yandex_cloud_ml_sdk import YCloudML
from typing import List, Optional
import aiohttp

# Инициализация FastAPI приложения
app = FastAPI()

# Инициализация Yandex Cloud SDK
sdk = YCloudML(
    folder_id="b1gco55q3lsml89erp6k",
    auth="AQVNxrWwNrAnFxkHXKKIVjeDJ_DhowEXE0Eg86EA",
)

# Инициализация сессии для асинхронных запросов
session = aiohttp.ClientSession()

# Модель данных для запроса
class QueryRequest(BaseModel):
    query: str
    id: int

# Модель данных для ответа
class QueryResponse(BaseModel):
    id: int
    answer: Optional[int] = None
    reasoning: str
    sources: List[str] = []

# Функция для поиска релевантных ссылок
async def search_web(query: str) -> list:
    # cache_key = f"search_{hash(query)}"
    # Для кэша можно использовать библиотеку, но для простоты предположим, что кеширование здесь будет не реализовано
    # Если кэширование нужно, можно использовать какой-либо кэш, например, Redis или встроенный кеш в FastAPI

    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": f"{query} site:itmo.ru",
                "key": os.getenv("GOOGLE_API_KEY"),
                "cx": 'f008fcde4857c4450',
                "num": 3
            },
            timeout=20
        ) as response:
            data = await response.json()
            links = [item["link"] for item in data.get("items", [])]
            return links
    except Exception as e:
        logging.error(f"Search Error: {str(e)}")
        return []

# Функция для генерации ответа
async def generate_answer(query: str, request_id: int) -> QueryResponse:
    # Определение вопроса и вариантов ответов
    match = re.match(r"(.*?)(\n\d\..*)", query, re.DOTALL)
    if not match:
        raise HTTPException(status_code=400, detail="Невалидный формат вопроса")

    question_text = match.group(1).strip()
    options_text = match.group(2).strip().split("\n")

    # Выделение вариантов ответов
    options = [option.split(". ", 1)[1] for option in options_text]

    # Формируем пронумерованный список вариантов с исходными номерами
    numbered_options = "\n".join([f"{i}. {option}" for i, option in enumerate(options, start=1)])

    # Жёстко заданный формат промпта
    messages = [
        {"role": "system", "text": "Предоставь информацию об Университете ИТМО"},
        {"role": "user", "text": f"Вопрос: {question_text}\nВарианты:\n{numbered_options}\n\nПожалуйста, ответь в следующем формате:\n1. Правильный ответ: (только номер варианта ответа)\n2. Объяснение: (пояснение без цифр и ссылок). Пример:\n 1. Правильный ответ: 2\n2. Объяснение: Так сказано на основной странице Университета ИТМО"},
    ]

    result = sdk.models.completions("yandexgpt-lite").configure(temperature=0.2).run(messages)
    print(result)

    # Разделим результат на две части: правильный ответ и объяснение
    result_text = result[0].text.strip()

    # Попробуем извлечь правильный ответ (цифру) и пояснение
    try:
        correct_answer_text = re.search(r"Правильный ответ:\s*(\d+)", result_text).group(1)
        reasoning = re.search(r"Объяснение:\s*(.*)", result_text).group(1).strip()
    except AttributeError:
        raise HTTPException(status_code=400, detail="Невалидный формат ответа от модели")

    # Преобразуем correct_answer_text в целое число
    answer_index = int(correct_answer_text)

    # Получим релевантные ссылки
    sources = await search_web(query)  # Асинхронный поиск

    # Формируем структуру JSON ответа
    response = QueryResponse(
        id=request_id,  # Используем id из запроса
        answer=answer_index,  # Просто используем правильный ответ как цифру
        reasoning=reasoning,  # Пояснение без цифр
        sources=sources,  # Добавляем релевантные ссылки
    )

    return response

# Роут для обработки POST-запросов
@app.post("/api/request", response_model=QueryResponse)
async def handle_request(request: QueryRequest):
    query = request.query
    response = await generate_answer(query, request.id)  # Передаем request.id в функцию
    return response
