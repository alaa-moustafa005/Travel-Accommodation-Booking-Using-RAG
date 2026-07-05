"""
Travel Accommodation RAG System — Azure-deployable FastAPI backend.

Converted from Travel_RAG_Final.ipynb:
  - Colab `userdata.get()` -> os.environ (Azure App Service Application Settings)
  - Removed ngrok tunnel (App Service gives you a permanent public URL)
  - Removed !pip installs (use requirements.txt instead)
  - Removed Gradio UI (the static/index.html page is now the UI, served by this same app)
  - persist_directory moved to /home/hotel_vector_db_langchain, which is the
    one path on Azure App Service (Linux) that persists across restarts/redeploys
  - Swapped 'gemini-2.5-flash-lite' -> 'gemini-2.0-flash-lite' (the former isn't
    publicly available yet)

Nothing about the RAG logic itself (chunking, retrieval, prompts, rate limiting,
city-cache freshness) was changed.
"""

import os
import re
import time
import threading
from collections import deque
from datetime import datetime, timedelta

import serpapi
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ════════════════════════════════════════════════
# Config / secrets — read from Azure App Service
# "Configuration -> Application settings", NOT hardcoded
# ════════════════════════════════════════════════
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

if not GEMINI_API_KEY or not SERPAPI_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY and SERPAPI_KEY must be set as Space secrets "
        "(Settings -> Variables and secrets, on your Hugging Face Space)."
    )

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# On Hugging Face Spaces free tier, local disk is ephemeral (wiped on
# restart/rebuild) but writable during the life of the running container.
# That's fine here since ensure_city_ready() re-scrapes on demand anyway.
VECTOR_DB_PATH = os.environ.get("VECTOR_DB_PATH", "/app/hotel_vector_db_langchain")
os.makedirs(VECTOR_DB_PATH, exist_ok=True)

REFRESH_DAYS = 14

# ════════════════════════════════════════════════
# Embeddings & vector store
# ════════════════════════════════════════════════
print("Loading embedding model...")
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    collection_name="hotels_langchain",
    embedding_function=embedding_function,
    persist_directory=VECTOR_DB_PATH,
)

city_last_scraped = {}


class RateLimiter:
    def __init__(self, max_calls: int, period_seconds: float = 60.0):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = deque()
        self._lock = threading.Lock()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            while self.calls and now - self.calls[0] > self.period:
                self.calls.popleft()
            if len(self.calls) >= self.max_calls:
                sleep_time = self.period - (now - self.calls[0]) + 0.05
                if sleep_time > 0:
                    time.sleep(sleep_time)
                now = time.monotonic()
                while self.calls and now - self.calls[0] > self.period:
                    self.calls.popleft()
            self.calls.append(time.monotonic())


gemini_rate_limiter = RateLimiter(max_calls=12, period_seconds=60.0)


def _normalize_city(city: str) -> str:
    return city.strip().lower()


def is_city_fresh(city: str) -> bool:
    city_key = _normalize_city(city)
    if city_key not in city_last_scraped:
        return False
    age = datetime.now() - city_last_scraped[city_key]
    return age < timedelta(days=REFRESH_DAYS)


# ════════════════════════════════════════════════
# Scraping + LangChain Document creation
# ════════════════════════════════════════════════
def scrape_hotels_for_city(city: str, check_in: str = "2026-08-01", check_out: str = "2026-08-03") -> list:
    print(f"Scraping live hotel data for {city}...")
    try:
        serp_client = serpapi.Client(api_key=SERPAPI_KEY)
        results = serp_client.search({
            "engine": "google_hotels",
            "q": f"Hotels in {city}",
            "check_in_date": check_in,
            "check_out_date": check_out,
            "currency": "USD",
            "hl": "en",
            "gl": "us",
        })
        properties = results.get("properties", [])
        if not properties:
            print(f"No hotels found for {city}")
            return []

        hotels = []
        for h in properties[:20]:
            hotels.append({
                "name": h.get("name", "Unknown"),
                "description": h.get("description", ""),
                "price_per_night": h.get("rate_per_night", {}).get("extracted_lowest", 0),
                "rating": h.get("overall_rating", 0.0),
                "reviews": h.get("reviews", 0),
                "amenities": h.get("amenities", []),
                "link": h.get("link", ""),
                "city": city,
            })
        print(f"Scraped {len(hotels)} hotels for {city}")
        return hotels

    except Exception as e:
        print(f"Scraping failed for {city}: {e}")
        return []


def hotels_to_langchain_documents(hotels: list) -> list:
    documents = []
    for h in hotels:
        amenities_str = ", ".join(h["amenities"]) if h["amenities"] else "No listed amenities"
        content = (
            f"Hotel: {h['name']} in {h['city']}. "
            f"Price: ${h['price_per_night']} per night. "
            f"Rating: {h['rating']} out of 5 based on {h['reviews']} reviews. "
            f"Amenities: {amenities_str}. "
            f"Description: {h['description']}"
        )
        documents.append(Document(
            page_content=content,
            metadata={
                "city": h["city"].strip().lower(),
                "name": h["name"],
                "price_per_night": h["price_per_night"],
                "rating": h["rating"],
                "link": h["link"],
            },
        ))
    return documents


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""],
)


# ════════════════════════════════════════════════
# Ingestion pipeline (cache-aware)
# ════════════════════════════════════════════════
def ingest_city(city: str) -> int:
    city_key = _normalize_city(city)
    hotels = scrape_hotels_for_city(city)
    if not hotels:
        return 0

    try:
        existing = vectorstore.get(where={"city": city_key})
        if existing["ids"]:
            vectorstore.delete(ids=existing["ids"])
    except Exception:
        pass

    raw_documents = hotels_to_langchain_documents(hotels)
    split_documents = text_splitter.split_documents(raw_documents)
    vectorstore.add_documents(split_documents)

    city_last_scraped[city_key] = datetime.now()
    print(f"Ingested {len(hotels)} hotels ({len(split_documents)} chunks) for {city}")
    return len(hotels)


def ensure_city_ready(city: str) -> int:
    if is_city_fresh(city):
        existing = vectorstore.get(where={"city": _normalize_city(city)})
        return len(existing["ids"])
    return ingest_city(city)


# ════════════════════════════════════════════════
# City extraction (local match first, Gemini fallback)
# ════════════════════════════════════════════════
city_extractor_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)

city_extraction_prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "Extract ONLY the city name the user is asking about. "
        "Respond with JUST the city name, nothing else — no country, "
        "no punctuation, no explanation. If a country is mentioned but "
        "no specific city, respond with that country's capital city. "
        "If no location is mentioned, respond with exactly: NONE"
    )),
    ("human", "{message}"),
])

city_extraction_chain = city_extraction_prompt | city_extractor_llm | StrOutputParser()

_city_extract_cache = {}

KNOWN_CITIES = [
    "cairo", "alexandria", "giza", "luxor", "aswan", "sharm el sheikh", "hurghada",
    "paris", "london", "rome", "milan", "venice", "florence", "madrid", "barcelona",
    "berlin", "munich", "amsterdam", "vienna", "prague", "lisbon", "athens",
    "istanbul", "dubai", "abu dhabi", "doha", "riyadh", "amman", "beirut",
    "tokyo", "osaka", "kyoto", "seoul", "beijing", "shanghai", "bangkok",
    "singapore", "kuala lumpur", "bali", "jakarta", "hong kong",
    "new york", "los angeles", "chicago", "miami", "san francisco", "las vegas",
    "toronto", "vancouver", "mexico city", "rio de janeiro", "sao paulo",
    "sydney", "melbourne", "auckland", "cape town", "marrakech", "casablanca",
]


def _local_extract_city(message: str):
    text = message.lower()
    for city in KNOWN_CITIES:
        if re.search(r"\b" + re.escape(city) + r"\b", text):
            return city.title()
    return None


def extract_city(message: str):
    cache_key = message.strip().lower()
    if cache_key in _city_extract_cache:
        return _city_extract_cache[cache_key]

    local_match = _local_extract_city(message)
    if local_match:
        _city_extract_cache[cache_key] = local_match
        return local_match

    try:
        gemini_rate_limiter.wait()
        city = city_extraction_chain.invoke({"message": message}).strip().strip(".")
        result = None if (not city or city.upper() == "NONE" or len(city) > 50) else city
        _city_extract_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"City extraction failed: {e}")
        return None


# ════════════════════════════════════════════════
# RAG chain
# ════════════════════════════════════════════════
answer_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)

RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful travel accommodation assistant.\n\n"
        "Answer ONLY using the information in the CONTEXT below. "
        "Do not invent hotel names, prices, or amenities not present in the context. "
        "If the context doesn't contain enough information, say so clearly and "
        "suggest the user try a different city or relax their requirements. "
        "Mention specific hotel names, prices, and ratings when relevant. "
        "Be concise and friendly.\n\n"
        "CONTEXT:\n{context}"
    )),
    ("human", "{question}"),
])


def format_retrieved_docs(docs: list) -> str:
    if not docs:
        return "No relevant hotels were found in the database for this query."
    lines = []
    for i, doc in enumerate(docs, 1):
        link = doc.metadata.get("link", "N/A")
        lines.append(f"[{i}] {doc.page_content} (link: {link})")
    return "\n".join(lines)


def rag_answer(user_message: str, history: list) -> str:
    try:
        city = extract_city(user_message)

        if city:
            hotel_count = ensure_city_ready(city)
            if hotel_count == 0:
                return (
                    f"I couldn't find any hotel data for **{city}**. "
                    "This could be a spelling issue or limited availability. "
                    "Please try a different city name."
                )

        search_kwargs = {"k": 5}
        if city:
            search_kwargs["filter"] = {"city": _normalize_city(city)}

        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        retrieved_docs = retriever.invoke(user_message)

        rag_chain = (
            {
                "context": lambda x: format_retrieved_docs(retrieved_docs),
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT_TEMPLATE
            | answer_llm
            | StrOutputParser()
        )

        gemini_rate_limiter.wait()
        return rag_chain.invoke(user_message)

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower():
            return "Rate limit hit. Please wait a moment and try again."
        if "api_key" in err.lower() or "invalid" in err.lower() or "401" in err:
            return "API key issue. Please check GEMINI_API_KEY and SERPAPI_KEY in App Service Configuration."
        return f"Unexpected error: {type(e).__name__}: {err}"


# ════════════════════════════════════════════════
# FastAPI app
# ════════════════════════════════════════════════
app = FastAPI(title="Travel RAG System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Q(BaseModel):
    message: str
    history: list = []


@app.post("/ask")
async def ask(q: Q):
    return {"answer": rag_answer(q.message, q.history)}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Serve the frontend (static/index.html) at the root URL, same origin as the API
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")
