from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import numpy as np
import json
import os
import glob
import re
import base64
from io import BytesIO
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from rapidfuzz import fuzz
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from rapidfuzz import fuzz
import string



EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = os.environ.get("API_KEY")
#API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZHMyMDAwMTE2QGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.zMwXMjQzRY5qReAa3jvzKD9lyPw0MZm2dbm-5tSfuW0"
JINA_API_KEY = "jina_ea7a5633e1434426b44c98fe0f0abdc3b1WqqCxKuougEsch7W2i0-CElX_J"
#JINA_API_KEY = os.environ.get("JINA_API_KEY")
JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str
    attachments: Optional[List[str]] = None

class AnswerResponse(BaseModel):
    answer: str
    links: List[dict]

def get_openai_embedding(text: str):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": EMBEDDING_MODEL
    }
    response = requests.post(EMBEDDING_URL, headers=headers, data=json.dumps(payload), timeout=15)
    if response.status_code != 200:
        raise Exception(f"Embedding API error: {response.status_code} - {response.text}")
    return response.json()["data"][0]["embedding"]

def get_jina_image_embedding(base64_image: str):
    headers = {
        "Authorization": f"Bearer {JINA_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": base64_image,
        "model": "jina-clip-v2"
    }
    response = requests.post(JINA_EMBEDDING_URL, headers=headers, json=payload, timeout=15)
    if response.status_code != 200:
        raise Exception(f"Jina API error: {response.status_code} - {response.text}")
    return response.json()["data"][0]["embedding"]

def get_image_embedding_from_url(url: str):
    if not url or not url.startswith("http"):
        print(f"[Image Skipped] Invalid URL: {url}")
        return None
    try:
        head = requests.head(url, timeout=5)
        if head.status_code != 200 or "image" not in head.headers.get("Content-Type", ""):
            print(f"[Image Skipped] Not a valid image: {url}")
            return None

        payload = {
            "model": "jina-clip-v2",
            "input": [{"image": url}]
        }
        response = requests.post(
            "https://api.jina.ai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {JINA_API_KEY}",
                "Content-Type": "application/json"
            },
            json=payload
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]
    except Exception as e:
        print(f"[Image Error] {url} - {e}")
        return None


def get_cached_posts():
    cache_file = 'cached_emb.json'
    source_file = 'post_dump.json'

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            try:
                data = json.load(f)
                if data:
                    return data
            except json.JSONDecodeError:
                print("Cached file is corrupted ")

    if not os.path.exists(source_file):
        raise FileNotFoundError("post_dump.json.")

    with open(source_file, 'r') as f:
        raw_data = json.load(f)
    raw_posts = raw_data.get("post_stream", {}).get("posts", [])
    all_post_contents = []
    for i, post in enumerate(raw_posts):
        try:
            soup = BeautifulSoup(post["cooked"], "html.parser")
            text = soup.get_text()
            images = post.get("images", [])

            text_embedding = get_openai_embedding(text)
            all_post_contents.append({
                "post_number": post["post_number"],
                "created_at": post["created_at"],
                "content": text,
                "images": images,
                "post_url": post["post_url"],
                "text_embedding": text_embedding,
                "image_embeddings": []
            })
        except Exception as e:
            continue

    with open(cache_file, 'w') as f:
        json.dump(all_post_contents, f)

    return all_post_contents

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def semantic_search(question, posts, image_embedding=None, top_k_text=10):
    question_embedding = get_openai_embedding(question)

    text_scores = [
        cosine_similarity(question_embedding, post['text_embedding'])
        for post in posts
    ]

    text_ranked = sorted(
        zip(text_scores, posts),
        key=lambda x: x[0],
        reverse=True
    )

    top_text_results = text_ranked[:top_k_text]

    if image_embedding is None:
        return top_text_results[:3]

    refined_results = []
    for score, post in top_text_results:
        image_embeds = []
        for url in post.get('images', []):
            emb = get_image_embedding_from_url(url)
            if emb is not None:
                image_embeds.append(emb)

        if image_embeds:
            sims = [cosine_similarity(image_embedding, emb) for emb in image_embeds]
            image_score = max(sims)
        else:
            image_score = 0.0

        combined_score = (score + image_score) / 2
        if question.lower() in post['content'].lower():
            combined_score += 0.5

        refined_results.append((combined_score, post))

    top_results = sorted(refined_results, key=lambda x: x[0], reverse=True)[:3]
    return top_results




nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)



def find_best_markdown_match(question, folder_path="markdown_files", threshold=30):
    best_match = None
    best_score = 0
    processed_question = preprocess(question)

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
        processed_title = preprocess(title)

        # Combine multiple fuzzy scores
        score1 = fuzz.token_set_ratio(processed_question, processed_title)
        score2 = fuzz.partial_ratio(processed_question, processed_title)
        score = max(score1, score2)

        if score > best_score:
            best_score = score
            best_match = {"url": original_url, "text": f"refer to: {title}"}

    if best_score >= threshold:
        return best_match
    return None



@app.post("/api/", response_model=AnswerResponse)
def answer_question(request: QuestionRequest):
    image_embeddings = []

    if request.attachments:
        for item in request.attachments:
            try:
                if item.startswith("http://") or item.startswith("https://"):
                    emb = get_image_embedding_from_url(item)
                else:
                    emb = get_jina_image_embedding(item)
                if emb is not None:
                    image_embeddings.append(emb)
            except Exception as e:
                print(f": {e}")

    image_embedding = None
    if image_embeddings:
        image_embedding = np.mean(image_embeddings, axis=0)

    all_post_contents = get_cached_posts()
    if not all_post_contents:
        raise HTTPException(status_code=404, detail="No posts.")

    top_results = semantic_search(request.question, all_post_contents, image_embedding=image_embedding)
    if not top_results:
        return AnswerResponse(answer="No posts.", links=[])

    answer = top_results[0][1]['content']
    links = [{
        "url": result[1]['post_url'],
        "text": result[1]['content']
    } for result in top_results]

    md_match = find_best_markdown_match(request.question)

    if md_match:
        links.append(md_match)

    return AnswerResponse(answer=answer, links=links)
