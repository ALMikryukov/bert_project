from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline


class ExtractiveQA:
    def __init__(
        self,
        model_name: str,
        hf_token: Optional[str] = None,
        max_seq_len: int = 384,
        doc_stride: int = 128,
        max_question_len: int = 64,
        max_context_chars: int = 1600,
    ):
        self.model_name = model_name
        self.max_seq_len = int(max_seq_len)
        self.doc_stride = int(doc_stride)
        self.max_question_len = int(max_question_len)
        self.max_context_chars = int(max_context_chars)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=False,
            token=hf_token,
        )
        self.model = AutoModelForQuestionAnswering.from_pretrained(
            model_name,
            token=hf_token,
        )

        self.qa = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
        )

        self.last_skipped_chunks = 0

    def _safe_context(self, context: str) -> str:
        context = (context or "").strip()
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars]
        return context

    def answer_best(
        self,
        question: str,
        candidates: List[Dict[str, Any]],
        min_answer_score: float = 0.25,
    ) -> Dict[str, Any]:
        best = None
        skipped = 0

        for c in candidates:
            context = self._safe_context(c["chunk"]["text"])
            try:
                out = self.qa(
                    question=question,
                    context=context,
                    max_seq_len=self.max_seq_len,
                    doc_stride=self.doc_stride,
                    max_question_len=self.max_question_len,
                    handle_impossible_answer=True,
                )
            except Exception:
                skipped += 1
                continue

            answer = (out.get("answer") or "").strip()
            qa_score = float(out.get("score", 0.0))

            merged = {
                "answer": answer,
                "qa_score": qa_score,
                "retrieval_score": float(c["score"]),
                "chunk_id": int(c["chunk_id"]),
                "doc_id": c["chunk"]["doc_id"],
                "page": c["chunk"]["page"],
            }

            if best is None or merged["qa_score"] > best["qa_score"]:
                best = merged

            # ранняя остановка: если модель уже уверена — не тратим время
            if best["qa_score"] >= 0.70:
                break

        self.last_skipped_chunks = skipped

        if best is None or best["qa_score"] < float(min_answer_score) or len(best["answer"]) < 2:
            return {
                "answer": "Не нашёл ответ в предоставленной документации.",
                "found": False,
                "debug": {"model": self.model_name, "skipped_chunks": skipped, "candidates": len(candidates)},
            }

        return {
            "answer": best["answer"],
            "found": True,
            "source": {"doc_id": best["doc_id"], "page": best["page"], "chunk_id": best["chunk_id"]},
            "scores": {"qa_score": best["qa_score"], "retrieval_score": best["retrieval_score"]},
            "debug": {"model": self.model_name, "skipped_chunks": skipped, "candidates": len(candidates)},
        }
