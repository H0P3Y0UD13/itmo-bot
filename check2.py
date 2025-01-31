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
# import Bea
import feedparser  # Для RSS новостей

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
    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": f"{query} site:itmo.ru",
                "key": os.getenv("GOOGLE_API_KEY"),
                "cx": 'f008fcde4857c4450',  # Используем ваш Custom Search ID
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

# Функция для получения последних новостей из RSS
async def get_itmo_news() -> list:
    news_links = []
    try:
        feed_url = "https://news.itmo.ru/rss/"
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:3]:  # Берем 3 последние новости
            news_links.append(entry.link)
    except Exception as e:
        logging.error(f"RSS Fetch Error: {str(e)}")
    return news_links

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

    # Жёстко заданный формат промпта с добавлением релевантных ссылок
    links = await search_web(query)  # Получаем релевантные ссылки
    news = await get_itmo_news()  # Получаем последние новости

    # Добавляем ссылки в контекст
    context = f"Вопрос: {question_text}\nВарианты:\n{numbered_options}\n\nРелевантные ссылки:\n"
    context += "\n".join(links + news)  # Сначала ссылки из поиска, потом новости

    messages = [
        {"role": "system",
         "text": "Предоставь информацию об Университете ИТМО"
        },
        {"role": "user",
         "text": f"{context}\n\nПожалуйста, ответь в следующем формате:\n1. Правильный ответ: (только номер варианта ответа)\n2. Объяснение: (пояснение без цифр и ссылок). Пример:\n 1. Правильный ответ: 2\n2. Объяснение: Так сказано на основной странице Университета ИТМО"
         },
    ]

    result = sdk.models.completions("yandexgpt-lite").configure(temperature=0.2).run(messages)
    print(result)

    # Разделим результат на две части: правильный ответ и объяснение
    result_text = result[0].text.strip()
    print(result_text)
    # Попробуем извлечь правильный ответ (цифру) и пояснение
    try:
        correct_answer_text = re.search(r"Правильный ответ:\s*(\d+)", result_text).group(1)
        reasoning = re.search(r"Объяснение:\s*(.*)", result_text).group(1).strip()
    except AttributeError:
        raise HTTPException(status_code=400, detail="Невалидный формат ответа от модели")

    # Преобразуем correct_answer_text в целое число
    answer_index = int(correct_answer_text)

    # Формируем структуру JSON ответа
    response = QueryResponse(
        id=request_id,  # Используем id из запроса
        answer=answer_index,  # Просто используем правильный ответ как цифру
        reasoning=reasoning,  # Пояснение без цифр
        sources=links + news,  # Добавляем релевантные ссылки и новости
    )

    return response

# Роут для обработки POST-запросов
@app.post("/api/request", response_model=QueryResponse)
async def handle_request(request: QueryRequest):
    query = request.query
    response = await generate_answer(query, request.id)  # Передаем request.id в функцию
    return response
