"""Small, beginner-friendly Retrieval-Augmented Generation (RAG) helper.

This does not fine-tune/train the language model. It extracts PDF text, splits it
into chunks, embeds those chunks, retrieves relevant chunks, and asks FLAN-T5
to answer using the retrieved context.
"""
from io import BytesIO
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "google/flan-t5-small"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract selectable text from an uploaded PDF."""
    reader = PdfReader(BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(
            "No selectable text was found. This may be a scanned PDF; run OCR first."
        )
    return text


def split_text(text: str, chunk_size: int = 700, overlap: int = 120) -> list[str]:
    """Split text into overlapping character chunks for this teaching demo."""
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


class SimpleRAG:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(GENERATION_MODEL).to(self.device)
        self.chunks = []
        self.vectors = None

    def index_pdf(self, pdf_bytes: bytes) -> int:
        text = extract_pdf_text(pdf_bytes)
        self.chunks = split_text(text)
        if not self.chunks:
            raise ValueError("The PDF did not contain usable text.")
        self.vectors = self.embedder.encode(
            self.chunks, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False
        )
        return len(self.chunks)

    def ask(self, question: str, top_k: int = 3) -> tuple[str, list[str]]:
        if self.vectors is None or not self.chunks:
            raise ValueError("Upload and index a PDF first.")
        q_vector = self.embedder.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.vectors @ q_vector
        best_ids = np.argsort(scores)[::-1][:top_k]
        retrieved = [self.chunks[int(i)] for i in best_ids]
        context = "\n\n".join(retrieved)
        prompt = (
            "Answer the question using only the context below. "
            "If the context does not contain the answer, say "
            "'I could not find that in the uploaded PDF.'\n\n"
            f"Context: {context}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=160, num_beams=2, early_stopping=True
            )
        answer = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()
        return answer, retrieved
