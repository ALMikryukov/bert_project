import os
from pydantic import BaseModel, Field


class Settings(BaseModel):
    # ===== QA (extractive) =====
    # Ставим базовую русскую QA (быстрее, чем large)
    QA_MODEL_NAME: str = Field(
        default=os.getenv("QA_MODEL_NAME", "MilyaShams/rubert-russian-qa-sberquad")
    )

    HF_TOKEN: str | None = Field(
        default_factory=lambda: os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )

    MIN_ANSWER_SCORE: float = Field(default=float(os.getenv("MIN_ANSWER_SCORE", "0.25")))

    # ограничиваем контекст, чтобы QA не захлёбывалась
    MAX_CONTEXT_CHARS: int = Field(default=int(os.getenv("MAX_CONTEXT_CHARS", "1600")))
    MAX_SEQ_LEN: int = Field(default=int(os.getenv("MAX_SEQ_LEN", "384")))
    DOC_STRIDE: int = Field(default=int(os.getenv("DOC_STRIDE", "128")))
    MAX_QUESTION_LEN: int = Field(default=int(os.getenv("MAX_QUESTION_LEN", "64")))

    # ===== Retrieval =====
    RETRIEVER_TYPE: str = Field(default=os.getenv("RETRIEVER_TYPE", "emb"))  # "tfidf" | "emb"
    TOP_K: int = Field(default=int(os.getenv("TOP_K", "3")))
    MIN_RETRIEVAL_SCORE: float = Field(default=float(os.getenv("MIN_RETRIEVAL_SCORE", "0.10")))

    # TF-IDF artifacts
    TFIDF_PATH: str = Field(default=os.getenv("TFIDF_PATH", "data/tfidf.joblib"))
    META_PATH: str = Field(default=os.getenv("META_PATH", "data/chunk_meta.joblib"))

    # Embedding artifacts
    EMB_MODEL_NAME: str = Field(
        default=os.getenv("EMB_MODEL_NAME", "intfloat/multilingual-e5-small")
    )
    EMB_PATH: str = Field(default=os.getenv("EMB_PATH", "data/embeddings.joblib"))


settings = Settings()
