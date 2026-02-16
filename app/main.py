from fastapi import FastAPI, HTTPException
from app.schemas import AskRequest, AskResponse
from app.settings import settings
from services.retriever import TfidfRetriever, EmbeddingRetriever
from services.qa import ExtractiveQA

app = FastAPI(
    title="Documentation QA (manual)",
    description="Отвечает на вопросы, основываясь ТОЛЬКО на предоставленной документации (manual.pdf).",
    version="0.2.0",
)

retriever = None
qa = None


@app.on_event("startup")
def startup():
    global retriever, qa

    if settings.HF_TOKEN is None:
        print("WARNING: HF_TOKEN не задан. Если модель приватная/gated, скачивание упадёт.")

    # Выбор ретривера
    if settings.RETRIEVER_TYPE.lower() == "tfidf":
        retriever = TfidfRetriever(settings.TFIDF_PATH, settings.META_PATH)
        print("Retriever: TF-IDF")
    else:
        retriever = EmbeddingRetriever(
            emb_path=settings.EMB_PATH,
            meta_path=settings.META_PATH,
            model_name=settings.EMB_MODEL_NAME,
        )
        print("Retriever: Embeddings")

    qa = ExtractiveQA(
        model_name=settings.QA_MODEL_NAME,
        hf_token=settings.HF_TOKEN,
        max_seq_len=settings.MAX_SEQ_LEN,
        doc_stride=settings.DOC_STRIDE,
        max_question_len=settings.MAX_QUESTION_LEN,
        max_context_chars=settings.MAX_CONTEXT_CHARS,
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Пустой вопрос")

    candidates = retriever.retrieve(question, top_k=settings.TOP_K)
    if not candidates:
        return AskResponse(answer="Не нашёл релевантных фрагментов в документации.", found=False)

    # (1) Фильтр по retrieval score: если даже лучший кандидат слабый — не гоняем QA
    best_retrieval = float(candidates[0]["score"])
    if best_retrieval < settings.MIN_RETRIEVAL_SCORE:
        return AskResponse(
            answer="Не нашёл релевантный фрагмент в документации (низкая уверенность поиска).",
            found=False,
            scores={"retrieval_score": best_retrieval},
        )

    result = qa.answer_best(
        question=question,
        candidates=candidates,
        min_answer_score=settings.MIN_ANSWER_SCORE,
    )

    if result.get("found"):
        return AskResponse(
            answer=result["answer"],
            found=True,
            source=result["source"],
            scores=result.get("scores"),
        )

    return AskResponse(answer=result["answer"], found=False)
