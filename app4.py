from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import aiohttp
import asyncio
import numpy as np
import json
import os
import glob
import re
import base64
from bs4 import BeautifulSoup

# === OpenAI Proxy Config ===
EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = os.environ.get("API_KEY")


# === Jina API Config ===
JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    images: Optional[List[str]] = None

class AnswerResponse(BaseModel):
    answer: str
    links: List[dict]

async def get_openai_embedding_async(text: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": EMBEDDING_MODEL
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(EMBEDDING_URL, headers=headers, json=payload, timeout=15) as response:
            if response.status != 200:
                raise Exception(f"Embedding API error: {response.status} - {await response.text()}")
            data = await response.json()
            return data["data"][0]["embedding"]

async def get_jina_image_embedding_async(base64_image: str):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": base64_image,
        "model": "jina-clip-v2"
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(JINA_EMBEDDING_URL, headers=headers, json=payload, timeout=15) as response:
            if response.status != 200:
                raise Exception(f"Jina API error: {response.status} - {await response.text()}")
            data = await response.json()
            return data["data"][0]["embedding"]

async def get_image_embedding_from_url_async(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                image_data = base64.b64encode(await response.read()).decode('utf-8')
                return await get_jina_image_embedding_async(image_data)
    except Exception:
        return None

def get_cached_posts():
    cache_file = 'cached_posts.json'
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print("Cached file is corrupted or empty.")
    raise FileNotFoundError("cached_posts.json not found.")

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def compute_similarity_async(post, question_embedding):
    score = cosine_similarity(question_embedding, post['text_embedding'])
    return (score, post)

async def semantic_search_async(question, posts, image_embedding=None, top_k_text=10):
    question_embedding = await get_openai_embedding_async(question)
    tasks = [compute_similarity_async(post, question_embedding) for post in posts]
    results = await asyncio.gather(*tasks)
    top_text_results = sorted(results, key=lambda x: x[0], reverse=True)[:top_k_text]

    if image_embedding is None:
        return top_text_results[:3], question_embedding

    refined_results = []
    for score, post in top_text_results:
        image_embeds = []
        for url in post.get("images", []):
            emb = await get_image_embedding_from_url_async(url)
            if emb is not None:
                image_embeds.append(emb)

        image_score = max([cosine_similarity(image_embedding, emb) for emb in image_embeds], default=0.0)
        combined_score = (score + image_score) / 2
        if question.lower() in post['content'].lower():
            combined_score += 0.5
        refined_results.append((combined_score, post))

    top_results = sorted(refined_results, key=lambda x: x[0], reverse=True)[:3]
    return top_results, question_embedding

async def find_best_markdown_match_async(question_embedding, folder_path="markdown_files", threshold=0.50):
    best_match = None
    best_score = -1

    for md_file in glob.glob(os.path.join(folder_path, "*.md")):
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()

        match = re.search(r'^---\s*(.*?)\s*---', content, re.DOTALL)
        if not match:
            continue

        front_matter = match.group(1)
        title_match = re.search(r'title:\s*"(.*?)"', front_matter)
        url_match = re.search(r'original_url:\s*"(.*?)"', front_matter)

        if not title_match or not url_match:
            continue

        title = title_match.group(1)
        original_url = url_match.group(1)

        try:
            title_embedding = await get_openai_embedding_async(title)
            score = cosine_similarity(question_embedding, title_embedding)
            if score >= best_score:
                best_score = score
                best_match = {"url": original_url, "text": "refer above article for more details"}
        except Exception as e:
            print(f"Error embedding markdown title: {e}")
            continue

    return best_match if best_score >= threshold else None

@app.post("/api/", response_model=AnswerResponse)
async def answer_question(request: QuestionRequest):
    image_embeddings = []

    if request.image:
        for item in request.image:
            try:
                emb = await get_image_embedding_from_url_async(item) if item.startswith("http") else await get_jina_image_embedding_async(item)
                if emb is not None:
                    image_embeddings.append(emb)
            except Exception as e:
                print(f"Error processing image: {e}")

    image_embedding = np.mean(image_embeddings, axis=0) if image_embeddings else None
    all_post_contents = get_cached_posts()
    if not all_post_contents:
        raise HTTPException(status_code=404, detail="No posts found to search.")

    top_results, question_embedding = await semantic_search_async(request.question, all_post_contents, image_embedding=image_embedding)
    if not top_results:
        return AnswerResponse(answer="No relevant posts found.", links=[])

    answer = top_results[0][1]['content']
    links = [{"url": result[1]['post_url'], "text": result[1]['content']} for result in top_results]

    md_match = await find_best_markdown_match_async(question_embedding)
    if md_match:
        links.append(md_match)

    return AnswerResponse(answer=answer, links=links)
