import re
import fitz  # pymupdf


def normalize_pdf_text(text: str) -> str:
    text = (text or "").replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Склеиваем переносы слов: "управле-\nния" -> "управления"
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

    # Нормализуем слишком частые переносы
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Сохраняем абзацы: двойные переносы оставляем как разделители
    placeholder = " <PARA_BREAK> "
    text = text.replace("\n\n", placeholder)

    # Обычные переносы строк превращаем в пробел (чтобы не ломать предложения)
    text = re.sub(r"\s*\n\s*", " ", text)

    # Восстанавливаем абзацы
    text = text.replace(placeholder, "\n\n")

    # Сжимаем пробелы
    text = re.sub(r"[ \t]+", " ", text)
    # Сжимаем пробелы вокруг абзацев
    text = re.sub(r" *\n\n *", "\n\n", text)

    return text.strip()


def extract_pages_text(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text") or ""
        text = normalize_pdf_text(text)
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages
