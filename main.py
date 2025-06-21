from fastapi import FastAPI, Request
from pydantic import BaseModel
import fitz  # PyMuPDF
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

# Config
PDF_PATH = "siddique_family.pdf"
GROQ_API_KEY = ""
GROQ_MODEL = "llama3-8b-8192"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

app = FastAPI()

# Load PDF
def read_pdf(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Chunk text
def split_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Prepare embeddings and FAISS index once at startup
text = read_pdf(PDF_PATH)
chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
embedder = SentenceTransformer("all-MiniLM-L6-v2")
chunk_vectors = embedder.encode(chunks)
dimension = chunk_vectors.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(chunk_vectors))

# Request schema
class Query(BaseModel):
    question: str

# Groq completion
def query_groq(context, question):
    prompt = f"""Use the following context to answer the question.

Context:
{context}

Question: {question}
Answer:"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# FastAPI endpoint
@app.post("/ask")
async def ask_question(q: Query):
    question_vector = embedder.encode([q.question])
    D, I = faiss_index.search(np.array(question_vector), k=3)
    relevant_chunks = "\n\n".join([chunks[i] for i in I[0]])
    answer = query_groq(relevant_chunks, q.question)
    return {"question": q.question, "answer": answer}
