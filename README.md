# PDF RAG Lab (free-tier friendly)

A beginner-friendly PDF question-answering lab for students. The app extracts PDF text, splits it into chunks, indexes those chunks, retrieves relevant passages, and optionally uses a hosted Hugging Face instruction model to generate an answer.

## Important about “free unlimited GPT”
There is no reliable promise of unlimited free hosted GPT/model tokens. Hugging Face access and model availability can have rate limits, quotas, or change over time. This app works without an API token in **retrieval-only mode** and displays relevant evidence from the PDF. Add a free Hugging Face token to enable optional generated answers; this is subject to the provider's current free-tier limits.

## Files
- `app.py` — Streamlit UI and RAG workflow
- `requirements.txt` — packages installed by Streamlit Community Cloud

## Run locally
1. Install Python 3.10+.
2. Open a terminal in this folder.
3. Run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. Upload a text-based PDF and click **Build / rebuild PDF index**.
5. Ask a question.

## Optional: enable hosted generation
1. Create a Hugging Face account and create a read token, subject to the service's current terms/limits.
2. For local testing, set an environment variable named `HF_TOKEN` before launching Streamlit.
   - Windows PowerShell: `$env:HF_TOKEN="hf_your_token_here"`
   - macOS/Linux: `export HF_TOKEN="hf_your_token_here"`
3. On Streamlit Community Cloud, open your app's **Settings → Secrets** and add:
   ```toml
   HF_TOKEN = "hf_your_token_here"
   ```
4. Never commit tokens or passwords to GitHub.

## Upload to GitHub
1. Sign in to GitHub and create a new **public** repository, e.g. `pdf-rag-lab`.
2. Upload `app.py`, `requirements.txt`, and `README.md` to the repository root.
3. Commit the files.

## Deploy to Streamlit Community Cloud
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose your repository, branch (`main`), and main file path (`app.py`).
4. Click **Deploy**.
5. If enabling generated answers, set `HF_TOKEN` under app Settings → Secrets. Otherwise, no API key is required.
6. Test by uploading a text-based PDF and asking a question.

## Teaching notes / limitations
- This demo does not fine-tune or train a model. It builds a temporary document index for retrieval.
- The index lives in the active Streamlit session; re-upload and rebuild after a session resets.
- Scanned/image-only PDFs need OCR before their text can be indexed.
- TF-IDF is deliberately used to keep this lab light and easy to understand. For production RAG, consider dense embeddings, persistent vector storage, source validation, and access controls.
