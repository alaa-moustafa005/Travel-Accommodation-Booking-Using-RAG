# Safari — AI-Powered Hotel Search (RAG Chatbot)

Safari is a Retrieval-Augmented Generation (RAG) chatbot that answers
natural-language hotel search questions using **live, real hotel listings**
rather than a static or hallucinated dataset. Built as a graduation project
for the DEPI Data Engineering & AI track.

**Live demo:** https://huggingface.co/spaces/omaaarrrr/safarii

---

## What it does

Ask something like *"Find me a luxury hotel in Paris with breakfast and a
pool"*, and Safari:

1. Identifies which city you're asking about
2. Scrapes live, current hotel listings for that city (if not already cached)
3. Embeds and stores those listings in a vector database
4. Retrieves the most relevant listings for your specific question
5. Generates a grounded answer using only the retrieved listings — never
   inventing hotel names, prices, or amenities

---

## Architecture

```
User (browser)
      │
      ▼
┌─────────────────────────────┐
│  FastAPI backend (main.py)  │
│  serves both the API        │
│  and the frontend UI        │
└──────────────┬──────────────┘
               │
     ┌─────────┼──────────────┐
     ▼         ▼               ▼
 SerpAPI    NVIDIA NIM     ChromaDB
 (live      (LLM: city     (vector store,
 hotel      extraction +   MiniLM
 data)      RAG answers)   embeddings)
```

| Layer | Technology |
|---|---|
| Backend | FastAPI (single process serves API + frontend) |
| LLM | NVIDIA NIM — `meta/llama-3.1-8b-instruct` |
| Orchestration | LangChain (LCEL) |
| Vector store | ChromaDB |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Live hotel data | SerpAPI (Google Hotels engine) |
| Frontend | Vanilla HTML/CSS/JS (`static/index.html`) |
| Deployment | Hugging Face Spaces (Docker) |

---

## Key design decisions

- **Grounded generation, not hallucination.** The system prompt explicitly
  instructs the model to answer only from retrieved context and say so
  clearly when data is insufficient, rather than inventing hotel details.
- **City-aware retrieval.** Before answering, the city in the question is
  identified and used to filter vector search results — so a question
  about Oslo never returns Cairo hotels.
- **Cost- and latency-aware city extraction.** A local list of ~180 known
  cities is checked with regex first; the LLM is only called for city
  extraction when a message doesn't match anything on that list. This
  roughly halves LLM calls per question.
- **Cache-aware scraping.** Hotel data for a city is only re-scraped if
  it's missing or older than 14 days, avoiding redundant SerpAPI calls.
- **Resilience.** Every LLM call is wrapped in a rate limiter plus
  exponential-backoff retry, and repeated identical questions are served
  from an in-memory cache instead of re-calling the LLM.

---

## Project structure

```
.
├── app.py                # FastAPI web layer — routes, CORS, serves frontend
├── rag.py                 # RAG pipeline — scraping, vector store, LLM calls
├── static/
│   └── index.html          # Frontend chat UI
├── Dockerfile
├── requirements.txt        # Python dependencies
├── .env.example             # Template for local secrets (copy to .env)
├── .gitignore
├── .dockerignore
└── README.md
```

`app.py` owns HTTP concerns only — routes, request/response models, CORS,
serving the frontend. `rag.py` owns everything AI/retrieval-related: city
extraction, live SerpAPI scraping, ChromaDB storage and retrieval, and the
final grounded-answer generation via NVIDIA NIM. `app.py` imports a single
function, `rag_answer()`, from `rag.py` — the two files have no other
coupling, so the RAG pipeline can be tested or reused independently of the
web server.

---

## Running locally

**1. Clone and install dependencies**

```bash
git clone <your-repo-url>
cd safari
pip install -r requirements.txt
```

**2. Set up your secrets**

```bash
cp .env.example .env
```

Then edit `.env` and fill in:
- `NVIDIA_API_KEY` — free key from [build.nvidia.com](https://build.nvidia.com)
  (open any model page, e.g. `meta/llama-3.1-8b-instruct`, click **Get API Key**)
- `SERPAPI_KEY` — from [serpapi.com](https://serpapi.com/manage-api-key)

**3. Run the server**

```bash
uvicorn app:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

---

## Deploying to Hugging Face Spaces

1. Create a new Space (SDK: **Docker**)
2. Push this repo's contents to the Space's git remote
3. In the Space's **Settings → Variables and secrets**, add:
   - `NVIDIA_API_KEY`
   - `SERPAPI_KEY`
4. The Space builds and starts automatically — no `.env` file needed in
   production; Space secrets are injected as environment variables directly.

---

## Known limitations

- Hugging Face's free tier puts the Space to sleep after a period of
  inactivity; the first request after waking may take a few seconds.
- ChromaDB storage on the free tier is ephemeral (wiped on
  restart/rebuild) — acceptable here since hotel data is re-scraped
  on demand rather than requiring long-term persistence.
- SerpAPI's free tier has a monthly search quota; heavy testing can
  consume it faster than expected.

---

## Author

Built by Omar as a graduation project for the DEPI Data Engineering & AI
track.
