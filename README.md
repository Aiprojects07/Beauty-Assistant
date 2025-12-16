# Beauty Assistant (Streamlit)

A Streamlit app that answers beauty product questions with retrieval, ranking and note-taking memory. The core logic lives in `product_tools_optimized.py`, and the app UI is `streamlit_app.py`.

## Local Setup

- Python: 3.9+
- Create and activate a virtual environment, then install requirements:

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

- Create a `.env` file (not committed, see `.gitignore`) with your API keys:

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
PINECONE_API_KEY=...
PINECONE_INDEX=your_index_name
PINECONE_NAMESPACE=optional_namespace
COHERE_API_KEY=optional
MEMORY_SESSION_ID=cli_session
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
COHERE_RERANK_MODEL=rerank-v3.5
```

- Run the app:

```bash
streamlit run streamlit_app.py
```

## Project Structure

- `streamlit_app.py` — Streamlit UI entry point (chat UI with streaming)
- `product_tools_optimized.py` — Core product QnA, memory tool, Pinecone/OpenAI/Anthropic integration
- `requirements.txt` — Dependencies for Streamlit Cloud
- `memories/` — Session-specific notes and files (ignored by git)
- `.env` — Local secrets (ignored by git)

## Deploy to Streamlit Cloud

1. Push this project to a GitHub repository (see next section).  
2. In Streamlit Cloud, create a new app and point it to your repo.
3. Set the main file path to `streamlit_app.py`.
4. Add the same environment variables in Streamlit Cloud (Settings → Secrets):
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `PINECONE_API_KEY`
   - `PINECONE_INDEX` (or `PINECONE_INDEX_NAME`)
   - `PINECONE_NAMESPACE` (optional)
   - `COHERE_API_KEY` (optional)
   - `MEMORY_SESSION_ID` (optional, defaults to `cli_session`)
   - `OPENAI_EMBEDDING_MODEL` (optional)
   - `COHERE_RERANK_MODEL` (optional)

Streamlit Cloud will install packages from `requirements.txt` automatically.

## Push to GitHub

If you don’t have a Git repo yet, run these commands in the project root:

```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: Beauty Assistant Streamlit app"
```

Create a GitHub repository (via GitHub UI), then add it as a remote and push:

```bash
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

Note: `.gitignore` already excludes `venv/`, `.env`, and local `memories/` so you won’t push secrets or local session files.

## Troubleshooting

- If you see SSL/LibreSSL warnings from `urllib3`, they are warnings and can usually be ignored locally.
- If Pinecone or Anthropic/OpenAI calls fail, confirm keys and index names in `.env` or Streamlit Cloud secrets.
- If the app cannot find `Layer_2_prompt.txt`, ensure it exists in the repo root (or set `QNA_PROMPT_PATH`).
