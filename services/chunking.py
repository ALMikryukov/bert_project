import re
from typing import List


_SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def _split_paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in (text or "").split("\n\n")]
    return [p for p in parts if len(p) > 0]


def _split_sentences(paragraph: str) -> List[str]:
    # очень простой сплит по концам предложений
    sents = [s.strip() for s in _SENT_SPLIT.split(paragraph.strip())]
    return [s for s in sents if len(s) > 0]


def chunk_text(
    text: str,
    target_chars: int = 900,
    overlap_sents: int = 2,
    min_len: int = 200,
) -> List[str]:
    """
    Чанкинг по абзацам и предложениям:
    - держим чанки "смысловыми"
    - overlap делаем по предложениям
    """
    text = (text or "").strip()
    if len(text) < min_len:
        return []

    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    chunks: List[str] = []
    cur: List[str] = []

    def flush():
        nonlocal cur
        if not cur:
            return
        chunk = " ".join(cur).strip()
        if len(chunk) >= min_len:
            chunks.append(chunk)
        cur = []

    for p in paragraphs:
        sents = _split_sentences(p)
        if not sents:
            continue

        for sent in sents:
            # если добавление предложения переполнит — сбрасываем
            prospective = (" ".join(cur + [sent])).strip()
            if len(prospective) > target_chars and cur:
                flush()
                # overlap: берём хвост последних N предложений из прошлого чанка
                if overlap_sents > 0 and chunks:
                    tail = _split_sentences(chunks[-1])[-overlap_sents:]
                    cur = tail.copy()
            cur.append(sent)

    flush()
    return chunks
