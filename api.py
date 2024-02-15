from typing import Annotated
from fastapi import FastAPI, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import urllib.request as libreq
import feedparser

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

index = faiss.read_index("arxiv_abstracts.index")
metadata = json.load(open('abstracts_metadata.json'))
dois = list(metadata.keys())
model = SentenceTransformer('hkunlp/instructor-xl')

async def embed_text(sentence):
    instruction = "Represent the arXiv query for retrieving supporting documents:"
    embeddings = model.encode([[instruction,sentence]])
    return embeddings

async def search_index(query, k):
    if (len(query.shape) == 1):
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
    query = np.array(query, dtype=np.float32)
    D, I = index.search(query, k)
    return I[0].tolist()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/search", response_class=HTMLResponse)
async def search(search: Annotated[str, Form(), None], request: Request):
    query = await embed_text(search)
    results = await search_index(query, 10)
    results = [f"https://arxiv.org/abs/{dois[i]}" for i in results]
    return templates.TemplateResponse("search_results.html", {"results": results, "request": request})