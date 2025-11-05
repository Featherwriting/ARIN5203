# src/rag_retriever.py
import os
import json
import torch
import numpy as np
from typing import List, Dict, Any
import faiss
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
import jsonlines

class LocalRAG:
    def __init__(self, kb_dir: str, embedder_name: str = "all-MiniLM-L6-v2"):
        self.kb_dir = kb_dir
        self.embedder = SentenceTransformer(embedder_name)
        self.documents = []
        self.embeddings = None
        self.index = None
        self._load_and_index()

    def _load_and_index(self):
        print(f"[RAG] Loading knowledge base from: {self.kb_dir}")
        for file in os.listdir(self.kb_dir):
            path = os.path.join(self.kb_dir, file)
            if file.endswith(".txt"):
                with open(path, "r", encoding="utf-8") as f:
                    self._add_text(f.read(), file)
            elif file.endswith(".pdf"):
                reader = PdfReader(path)
                text = "\n".join([p.extract_text() or "" for p in reader.pages])
                self._add_text(text, file)
            elif file.endswith(".jsonl"):
                with jsonlines.open(path) as reader:
                    for obj in reader:
                        if "text" in obj:
                            self._add_text(obj["text"], f"{file}#jsonl")

        if not self.documents:
            print("[RAG] Warning: No documents loaded.")
            return

        print(f"[RAG] Encoding {len(self.documents)} chunks...")
        texts = [d["text"] for d in self.documents]
        self.embeddings = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        print(f"[RAG] Index ready: {self.index.ntotal} vectors")

    def _add_text(self, text: str, source: str, chunk_size: int = 300):
        words = text.split()
        for i in range(0, len(words), chunk_size - 50):
            chunk = " ".join(words[i:i + chunk_size])
            self.documents.append({"text": chunk, "source": source})

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        if self.index is None or len(self.documents) == 0:
            return []
        q_vec = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        scores, idxs = self.index.search(q_vec, top_k)
        return [
            {**self.documents[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0]) if i >= 0
        ]