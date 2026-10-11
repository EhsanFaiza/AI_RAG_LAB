import os
import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚")
st.title("📚 PDF RAG Lab")
st.write("Upload a PDF, build a searchable index, and ask questions about its content.")

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    pages = []
    for page_no, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append((page_no, text))
    return pages

def make_chunks(pages, chunk_size=900, overlap=150):
    chunks = []
    for page_no, text in pages:
        start = 0
        while start < len(text):
            part = text[start:start + chunk_size].strip()
            if part:
                chunks.append({"page": page_no, "text": part})
            start += max(1, chunk_size - overlap)
    return chunks

def retrieve(question, chunks, vectorizer, matrix, top_k=4):
    q_vec = vectorizer.transform([question])
    scores = cosine_similarity(q_vec, matrix).ravel()
    best = scores.argsort()[::-1][:top_k]
    return [(chunks[i], float(scores[i])) for i in best if scores[i] > 0]

def generate_answer(question, retrieved):
    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}" for item, _ in retrieved
    )
    token = os.getenv("HF_TOKEN", "").strip()
    try:
        token = token or st.secrets.get("HF_TOKEN", "").strip()
    except Exception:
        pass

    if token and context:
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=token)
            prompt = (
                "Answer the question using only the context below. "
                "If the answer is not in the context, say you could not find it in the PDF. "
                "Cite page numbers when possible.\n\n"
                f"CONTEXT:\n{context}\n\nQUESTION: {question}\nANSWER:"
            )
            result = client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=350,
                temperature=0.2,
            )
            return result.choices[0].message.content, context, "Hugging Face model"
        except Exception as exc:
            st.warning(f"Model API unavailable; showing retrieved PDF evidence instead. Details: {exc}")

    if not retrieved:
        return "I couldn't find a relevant passage in the indexed PDF. Try different keywords or check whether the PDF contains selectable text.", context, "Retrieval only"
    excerpts = "\n\n".join(
        f"**Page {item['page']}** (similarity {score:.2f})\n\n{item['text']}"
        for item, score in retrieved
    )
    return "No model API is configured, so here are the most relevant passages from your PDF:\n\n" + excerpts, context, "Retrieval only"

uploaded = st.file_uploader("Upload a text-based PDF", type=["pdf"])

if uploaded:
    if st.button("Build / rebuild PDF index", type="primary"):
        with st.spinner("Reading PDF and building the index..."):
            pages = extract_pdf(uploaded)
            chunks = make_chunks(pages)
            if not chunks:
                st.error("No selectable text was found. This may be a scanned PDF; run OCR first.")
                st.stop()
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            matrix = vectorizer.fit_transform([c["text"] for c in chunks])
            st.session_state["rag_data"] = {
                "chunks": chunks, "vectorizer": vectorizer, "matrix": matrix,
                "filename": uploaded.name
            }
        st.success(f"Indexed {len(pages)} pages into {len(chunks)} chunks from {uploaded.name}.")

if "rag_data" in st.session_state:
    data = st.session_state["rag_data"]
    st.caption(f"Current index: {data['filename']}")
    question = st.text_input("Ask a question about your PDF")
    if st.button("Ask"):
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            hits = retrieve(question, data["chunks"], data["vectorizer"], data["matrix"])
            answer, context, mode = generate_answer(question, hits)
            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Mode: {mode}")
            if hits:
                with st.expander("Retrieved evidence"):
                    for item, score in hits:
                        st.markdown(f"**Page {item['page']} · similarity {score:.2f}**")
                        st.write(item["text"])
                        st.divider()

with st.expander("For the instructor: how this RAG demo works"):
    st.markdown(
        "1. **Load:** extract text from the uploaded PDF.  \n"
        "2. **Chunk:** split the text into overlapping passages.  \n"
        "3. **Index:** create a TF-IDF search index.  \n"
        "4. **Retrieve:** find passages related to the question.  \n"
        "5. **Generate (optional):** if `HF_TOKEN` is configured, ask a Hugging Face hosted instruction model to answer from the retrieved context.  \n\n"
        "**Teaching note:** RAG indexes/retrieves document content; it does not train or fine-tune the language model."
    )
