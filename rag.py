"""
rag.py — The RAG pipeline: city extraction, live hotel scraping, vector
storage/retrieval, and grounded answer generation via NVIDIA NIM.

This file is completely independent of FastAPI or any web framework — it
only exposes plain Python functions. `app.py` imports `rag_answer()` from
here and wires it up to HTTP endpoints. This separation means the pipeline
can be tested or reused (e.g. from a CLI script or a notebook) without ever
starting a web server.

  - Uses NVIDIA NIM (meta/llama-3.1-8b-instruct) for both city extraction
    and answer generation, after repeated Gemini free-tier rate-limit
    failures during testing.
  - Uses SerpAPI's 'GoogleSearch' interface (not 'serpapi.Client', which
    doesn't exist on all installs depending on install order between the
    'serpapi' and 'google-search-results' PyPI packages).
  - Wraps every LLM call in exponential backoff, and caches repeated
    identical questions/city lookups in memory.
  - KNOWN_CITIES covers ~180 cities across every region, so most questions
    skip the LLM extraction call entirely via local regex matching.
"""

import os
import re
import time
import threading
from collections import deque
from datetime import datetime, timedelta

from dotenv import load_dotenv
from serpapi import GoogleSearch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Loads variables from a local .env file if one exists (for local development).
# On Hugging Face Spaces, secrets are already injected as real environment
# variables at container start, so this call simply finds no .env file and
# does nothing there — safe to call unconditionally in both environments.
load_dotenv()

# ════════════════════════════════════════════════
# Config / secrets — never hardcoded.
# Local dev: set these in a .env file (see .env.example).
# Hugging Face Spaces: set these in Settings -> Variables and secrets.
# ════════════════════════════════════════════════
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
SERPAPI_KEY = os.environ.get("SERPAPI_KEY")

if not NVIDIA_API_KEY or not SERPAPI_KEY:
    raise RuntimeError(
        "NVIDIA_API_KEY and SERPAPI_KEY must be set — either in a local .env "
        "file (see .env.example) or as Space secrets on Hugging Face "
        "(Settings -> Variables and secrets)."
    )

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


nvidia_rate_limiter = RateLimiter(max_calls=8, period_seconds=60.0)


def call_with_backoff(fn, *args, max_retries: int = 3, **kwargs):
    """
    Calls fn(*args, **kwargs), retrying with exponential backoff on a
    429/rate-limit error. Absorbs short-lived bumps automatically instead
    of surfacing an error on the first hit. Cannot help if a *daily* quota
    is exhausted — that only clears at the provider's reset time.
    """
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last_err = e
            err = str(e)
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2
                    continue
            raise
    raise last_err


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
        params = {
            "engine": "google_hotels",
            "q": f"Hotels in {city}",
            "check_in_date": check_in,
            "check_out_date": check_out,
            "currency": "USD",
            "hl": "en",
            "gl": "us",
            "api_key": SERPAPI_KEY,
        }
        search = GoogleSearch(params)
        results = search.get_dict()
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
# City extraction (local match first, NVIDIA NIM fallback)
# ════════════════════════════════════════════════
city_extractor_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=NVIDIA_API_KEY, temperature=0)

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
    # Middle East / North Africa
    "cairo", "alexandria", "giza", "luxor", "aswan", "sharm el sheikh", "hurghada",
    "istanbul", "ankara", "izmir", "dubai", "abu dhabi", "doha", "riyadh", "jeddah",
    "amman", "beirut", "muscat", "manama", "kuwait city", "tel aviv", "jerusalem",
    "marrakech", "casablanca", "tunis", "algiers", "tripoli", "khartoum",
    # Europe
    "paris", "nice", "lyon", "marseille", "london", "manchester", "edinburgh",
    "rome", "milan", "venice", "florence", "naples", "bologna",
    "madrid", "barcelona", "seville", "valencia", "granada",
    "berlin", "munich", "hamburg", "frankfurt", "cologne",
    "amsterdam", "rotterdam", "the hague", "brussels", "antwerp",
    "vienna", "salzburg", "zurich", "geneva", "bern",
    "prague", "budapest", "warsaw", "krakow", "bucharest", "sofia",
    "lisbon", "porto", "athens", "santorini", "mykonos", "thessaloniki",
    "dublin", "reykjavik", "oslo", "bergen", "stockholm", "gothenburg",
    "copenhagen", "helsinki", "moscow", "st petersburg", "kyiv",
    "belgrade", "zagreb", "ljubljana", "bratislava", "tallinn", "riga", "vilnius",
    # Asia
    "tokyo", "osaka", "kyoto", "yokohama", "sapporo", "fukuoka",
    "seoul", "busan", "beijing", "shanghai", "guangzhou", "shenzhen", "chengdu",
    "hong kong", "macau", "taipei",
    "bangkok", "phuket", "chiang mai", "singapore",
    "kuala lumpur", "penang", "bali", "jakarta", "bandung", "surabaya",
    "manila", "cebu", "hanoi", "ho chi minh city", "phnom penh", "vientiane",
    "yangon", "kathmandu", "colombo",
    "mumbai", "delhi", "bangalore", "chennai", "kolkata", "hyderabad", "jaipur", "goa",
    "islamabad", "karachi", "lahore", "dhaka", "almaty", "tashkent", "baku", "tbilisi", "yerevan",
    # North America
    "new york", "los angeles", "chicago", "miami", "san francisco", "las vegas",
    "boston", "seattle", "washington", "washington dc", "houston", "austin",
    "san diego", "philadelphia", "atlanta", "denver", "orlando", "new orleans",
    "toronto", "vancouver", "montreal", "ottawa", "calgary",
    "mexico city", "cancun", "guadalajara", "tijuana",
    # South America
    "rio de janeiro", "sao paulo", "brasilia", "salvador",
    "buenos aires", "lima", "bogota", "medellin", "santiago", "quito",
    "montevideo", "caracas", "la paz", "asuncion",
    # Africa (sub-Saharan)
    "cape town", "johannesburg", "durban", "nairobi", "lagos", "abuja",
    "accra", "addis ababa", "dar es salaam", "kigali", "kampala",
    # Oceania
    "sydney", "melbourne", "brisbane", "perth", "auckland", "wellington", "queenstown",
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
        nvidia_rate_limiter.wait()
        city = call_with_backoff(city_extraction_chain.invoke, {"message": message}).strip().strip(".")
        result = None if (not city or city.upper() == "NONE" or len(city) > 50) else city
        _city_extract_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"City extraction failed: {e}")
        return None


# ════════════════════════════════════════════════
# RAG chain
# ════════════════════════════════════════════════
answer_llm = ChatNVIDIA(model="meta/llama-3.1-8b-instruct", api_key=NVIDIA_API_KEY, temperature=0.3)

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


# Caches the final answer per normalized question so a repeated/duplicate
# question (common when testing or during a live demo) doesn't cost another
# NVIDIA call.
_answer_cache = {}


def rag_answer(user_message: str, history: list = None) -> str:
    """
    The single entry point the web layer (app.py) calls. Given a raw user
    message, returns a grounded, plain-text answer string.
    """
    cache_key = user_message.strip().lower()
    if cache_key in _answer_cache:
        return _answer_cache[cache_key]

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

        nvidia_rate_limiter.wait()
        answer = call_with_backoff(rag_chain.invoke, user_message)
        _answer_cache[cache_key] = answer
        return answer

    except Exception as e:
        err = str(e)
        if "429" in err or "quota" in err.lower() or "rate" in err.lower():
            return "Rate limit hit. Please wait a moment and try again."
        if "api_key" in err.lower() or "invalid" in err.lower() or "401" in err:
            return "API key issue. Please check NVIDIA_API_KEY and SERPAPI_KEY in your Space secrets."
        return f"Unexpected error: {type(e).__name__}: {err}"
