from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from transformers import CLIPProcessor, CLIPModel
import torch
import base64
from io import BytesIO
from PIL import Image
import json
import os
import glob
import re

# === OpenAI Proxy Config ===
EMBEDDING_URL = "https://aiproxy.sanand.workers.dev/openai/v1/embeddings"
EMBEDDING_MODEL = "text-embedding-3-small"
API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjI0ZHMyMDAwMTE2QGRzLnN0dWR5LmlpdG0uYWMuaW4ifQ.zMwXMjQzRY5qReAa3jvzKD9lyPw0MZm2dbm-5tSfuW0"

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str
    attachments: Optional[List[str]] = None

class AnswerResponse(BaseModel):
    answer: str
    links: List[dict]

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
    embedding = response.json()["data"][0]["embedding"]
    return torch.tensor(embedding)

def get_image_embedding(base64_image: str):
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features[0]
    except Exception:
        return None

def get_image_embedding_from_url(url: str):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt")
        with torch.no_grad():
            image_features = clip_model.get_image_features(**inputs)
        return image_features[0]
    except Exception:
        return None

def get_cached_posts():
    cache_file = 'cached_posts.json'
    source_file = 'post_dump.json'

    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            try:
                data = json.load(f)
                if data:
                    return data
            except json.JSONDecodeError:
                print("Cached file is corrupted or empty. Rebuilding cache...")

    if not os.path.exists(source_file):
        raise FileNotFoundError("post_dump.json not found.")

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
                "text_embedding": text_embedding.tolist(),
                "image_embeddings": []
            })
        except Exception as e:
            print(f"Error processing post {i+1}: {e}")
            continue

    with open(cache_file, 'w') as f:
        json.dump(all_post_contents, f)

    return all_post_contents

def cosine_similarity(a, b):
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def semantic_search(question, posts, image_embedding=None, top_k_text=10):
    question_embedding = get_openai_embedding(question).tolist()
    question_tensor = torch.tensor(question_embedding)

    text_scores = [
        cosine_similarity(question_tensor, torch.tensor(post['text_embedding']))
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

def find_best_markdown_match(question_embedding, folder_path="markdown_files", threshold=0.50):
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
            title_embedding = get_openai_embedding(title)
            score = cosine_similarity(question_embedding, title_embedding)
            if score > best_score:
                best_score = score
                best_match = {"url": original_url, "text": "refer above article for more details"}
        except Exception as e:
            print(f"Error embedding markdown title: {e}")
            continue

    if best_score >= threshold:
        return best_match
    return None

@app.post("/api/", response_model=AnswerResponse)
def answer_question(request: QuestionRequest):
    image_embeddings = []

    if request.attachments:
        for base64_file in request.attachments:
            emb = get_image_embedding(base64_file)
            if emb is not None:
                image_embeddings.append(emb)

    image_embedding = None
    if image_embeddings:
        image_embedding = torch.stack(image_embeddings).mean(dim=0)

    all_post_contents = get_cached_posts()
    if not all_post_contents:
        raise HTTPException(status_code=404, detail="No posts found to search.")

    top_results = semantic_search(request.question, all_post_contents, image_embedding=image_embedding)
    if not top_results:
        return AnswerResponse(answer="No relevant posts found.", links=[])

    answer = top_results[0][1]['content']
    links = [{
        "url": result[1]['post_url'],
        "text": result[1]['content']
    } for result in top_results]

    question_embedding = get_openai_embedding(request.question)
    md_match = find_best_markdown_match(question_embedding)
    if md_match:
        links.append(md_match)

    return AnswerResponse(answer=answer, links=links)
