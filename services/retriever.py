from typing import List, Dict, Any, Optional
import numpy as np
import joblib

import torch
from transformers import AutoTokenizer, AutoModel


class TfidfRetriever:
    def __init__(self, tfidf_path: str, meta_path: str):
        self.vectorizer = joblib.load(tfidf_path)
        meta = joblib.load(meta_path)
        self.X = meta["X"]               # sparse matrix [n_chunks, n_features]
        self.chunks = meta["chunks"]     # list of dicts

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = (question or "").strip()
        if not q:
            return []

        q_vec = self.vectorizer.transform([q])  # [1, n_features]
        scores = (self.X @ q_vec.T).toarray().ravel()  # [n_chunks]
        if scores.size == 0:
            return []

        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"score": float(scores[idx]), "chunk": self.chunks[idx], "chunk_id": int(idx)}
            for idx in top_idx
        ]


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class EmbeddingRetriever:
    """
    Semantic retrieval по заранее посчитанным эмбеддингам чанков.
    Эмбеддинги должны быть L2-нормализованы -> cosine = dot.
    """
    def __init__(self, emb_path: str, meta_path: str, model_name: str):
        meta = joblib.load(meta_path)
        self.chunks = meta["chunks"]

        pack = joblib.load(emb_path)
        self.embeddings = pack["embeddings"].astype(np.float32)  # [n_chunks, dim], normalized
        self.model_name = pack.get("model_name", model_name)

        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        # e5 использует префиксы query:/passage: (это заметно улучшает качество)
        self.use_e5_prefix = "e5" in model_name.lower()

    @torch.inference_mode()
    def _embed_query(self, text: str) -> np.ndarray:
        t = text.strip()
        if self.use_e5_prefix:
            t = "query: " + t

        enc = self.tokenizer(
            t,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        )
        out = self.model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled[0].cpu().numpy().astype(np.float32)

    def retrieve(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        q = (question or "").strip()
        if not q:
            return []

        q_emb = self._embed_query(q)  # [dim]
        scores = self.embeddings @ q_emb  # [n_chunks] cosine sim (because normalized)
        top_idx = np.argsort(scores)[::-1][:top_k]

        return [
            {"score": float(scores[idx]), "chunk": self.chunks[idx], "chunk_id": int(idx)}
            for idx in top_idx
        ]
