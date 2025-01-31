import asyncio
import json
import re
import logging
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
import feedparser
from urllib.parse import urlparse
from dotenv import load_dotenv

load_dotenv()

YC_FOLDER_ID = os.getenv("YC_FOLDER_ID")
YC_IAM_TOKEN = os.getenv("YC_IAM_TOKEN")
SEARCH_ENGINE_ID = os.getenv("SEARCH_ENGINE_ID")
GOOGLE_API_KEY = os.getenv("SEARCH_ENGINE_ID")

app = FastAPI()
session = aiohttp.ClientSession()

class QueryRequest(BaseModel):
    query: str
    id: int

class QueryResponse(BaseModel):
    id: int
    answer: Optional[int] = None
    reasoning: str
    sources: List[str] = []

def is_valid_itmo_link(url: str) -> bool:
    """Проверяет принадлежность ссылки к доменам ITMO"""
    domains = ['itmo.ru', 'news.itmo.ru']
    parsed = urlparse(url)
    return any(parsed.netloc.endswith(domain) for domain in domains)

async def search_web(query: str) -> list:
    try:
        async with session.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "q": f"{query} site:itmo.ru",
                "key": GOOGLE_API_KEY,
                "cx": SEARCH_ENGINE_ID,
                "num": 5,
                # "sort": "date"  # Приоритет новым материалам
            },
            timeout=20
        ) as response:
            data = await response.json()
            return [item["link"] for item in data.get("items", []) if is_valid_itmo_link(item["link"])]
    except Exception as e:
        logging.error(f"Search Error: {str(e)}")
        return []

async def get_itmo_news() -> list:
    try:
        feed = feedparser.parse("https://news.itmo.ru/rss/")
        return [entry.link for entry in feed.entries[:5]]
    except Exception as e:
        logging.error(f"RSS Error: {str(e)}")
        return []

def format_context(links: list, news: list) -> str:
    unique_sources = list(set(links + news))
    return "Релевантные источники ITMO:\n" + "\n".join(
        f"[Источник {i+1}] {url}" 
        for i, url in enumerate(unique_sources[:8])  # Ограничение контекста
    )

async def generate_answer(query: str, request_id: int) -> QueryResponse:
    match = re.match(r"(.*?)(\n\d\..*)", query, re.DOTALL)
    if not match:
        raise HTTPException(400, "Invalid question format")

    question, options = match.group(1).strip(), match.group(2).strip()
    options_list = [opt.split(". ", 1)[1] for opt in options.split("\n")]

    # Параллельный сбор информации
    links, news = await asyncio.gather(
        search_web(question),
        get_itmo_news()
    )
    
    context = f"""Вопрос: {question}
 Варианты ответов:
 {''.join(f'{i}. {opt}\n' for i, opt in enumerate(options_list, 1))}

 {format_context(links, news)}
 """

    messages = [
        {
            "role": "system",
            "text": """Ты официальный помощник Университета ИТМО. Отвечай ТОЛЬКО на основе предоставленной информации. 
Если ответа нет в источниках, то ищи сам ответ. Формат ответа:
1. Правильный ответ: ( СТРОГО ! Только номер варианта ответа!)
2. Объяснение: (пояснение без цифр и ссылок)
Пример:\n 1. Правильный ответ: 2\n2. Объяснение: Так сказано на основной странице Университета ИТМО"""
        },
        {
            "role": "user", 
            "text": context
        }
    ]

    try:
        # Формируем запрос к YandexGPT API
        headers = {
            "Authorization": f"Api-Key {YC_IAM_TOKEN}",
            "x-folder-id": YC_FOLDER_ID,
            "Content-Type": "application/json"
        }

        body = {
            "modelUri": f"gpt://{YC_FOLDER_ID}/yandexgpt-lite",
            "completionOptions": {
                "stream": False,
                "temperature": 0.2,
                "maxTokens": 500
            },
            "messages": messages
        }

        async with session.post(
            "https://llm.api.cloud.yandex.net/foundationModels/v1/completion",
            headers=headers,
            json=body
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise HTTPException(response.status, f"API Error: {error_text}")
            
            result = await response.json()
            result_text = result['result']['alternatives'][0]['message']['text']
            print(result_text)

            # Парсинг ответа
            answer_match = re.search(
                r"Правильный ответ:\s*(\d+)", 
                result_text, 
                re.IGNORECASE | re.MULTILINE
            )
            
            reasoning_match = re.search(
                r"Объяснение:\s*(.*?)(?=\n\s*\d+\.|\n\n|$)", 
                result_text, 
                re.DOTALL | re.IGNORECASE
            )

            if not answer_match or not reasoning_match:
                logging.error(f"Invalid response format: {result_text}")
                return QueryResponse(
                    id=request_id,
                    reasoning="Не удалось обработать ответ системы",
                    sources=[]
                )

            return QueryResponse(
                id=request_id,
                answer=int(answer_match.group(1)),
                reasoning=reasoning_match.group(1).strip(),
                sources=list(set(links + news))[:3]
            )

    except Exception as e:
        logging.error(f"Generation Error: {str(e)}", exc_info=True)
        return QueryResponse(
            id=request_id,
            reasoning="Ошибка при обработке запроса",
            sources=[]
        )

@app.post("/api/request", response_model=QueryResponse)
async def handle_request(request: QueryRequest):
    return await generate_answer(request.query, request.id)

@app.on_event("shutdown")
async def shutdown():
    await session.close()

if __name__ == "__main__":
    app.run(app, host="0.0.0.0", port=8080)