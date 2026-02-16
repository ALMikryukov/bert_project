import json
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer

from services.pdf_extract import extract_pages_text
from services.chunking import chunk_text

PROJECT_DIR = Path(".")
PDF_PATH = PROJECT_DIR / "manual_vacum.pdf"
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
TFIDF_PATH = DATA_DIR / "tfidf.joblib"
META_PATH = DATA_DIR / "chunk_meta.joblib"
EMB_PATH = DATA_DIR / "embeddings.joblib"

# Можно менять модель эмбеддингов здесь или через env EMB_MODEL_NAME
EMB_MODEL_NAME = "intfloat/multilingual-e5-small"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


@torch.inference_mode()
def embed_texts(texts, model_name: str, batch_size: int = 32, max_length: int = 256):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    use_e5_prefix = "e5" in model_name.lower()

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        if use_e5_prefix:
            batch = ["passage: " + t for t in batch]

        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        out = model(**enc)
        pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        all_vecs.append(pooled.cpu().numpy().astype(np.float32))

    return np.vstack(all_vecs)


def main():
    if not PDF_PATH.exists():
        raise FileNotFoundError(f"Не найден файл: {PDF_PATH.resolve()}")

    pages = extract_pages_text(str(PDF_PATH))

    chunks = []
    for p in pages:
        page_num = p["page"]
        text = p["text"]

        for ch in chunk_text(
            text,
            target_chars=900,
            overlap_sents=2,
            min_len=200,
        ):
            chunks.append({
                "doc_id": "manual",
                "page": page_num,
                "text": ch
            })

    # jsonl для отладки
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    corpus = [c["text"] for c in chunks]

    # TF-IDF (оставляем как fallback)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=80_000
    )
    X = vectorizer.fit_transform(corpus)
    joblib.dump(vectorizer, TFIDF_PATH)

    # Embeddings (semantic)
    embeddings = embed_texts(corpus, model_name=EMB_MODEL_NAME, batch_size=32, max_length=256)
    joblib.dump({"embeddings": embeddings, "model_name": EMB_MODEL_NAME}, EMB_PATH)

    # Meta (chunks + tfidf-matrix)
    joblib.dump({"X": X, "chunks": chunks}, META_PATH)

    print("Готово!")
    print(f"chunks: {len(chunks)}")
    print(f"saved: {CHUNKS_PATH}, {TFIDF_PATH}, {META_PATH}, {EMB_PATH}")


if __name__ == "__main__":
    main()
