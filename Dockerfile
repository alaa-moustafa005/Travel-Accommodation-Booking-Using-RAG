# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Hugging Face Spaces (Docker SDK) runs containers as a non-root user by
# convention — create one here so it works both on Spaces and elsewhere.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Install dependencies first (separate layer) so code-only changes don't
# force a full reinstall of everything, including the larger ML packages.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Now copy the actual application code
COPY --chown=user app.py rag.py .
COPY --chown=user static/ ./static/

# Persisted only for the life of the running container on the free tier —
# rag.py re-scrapes on demand, so this doesn't need to survive restarts.
RUN mkdir -p /app/hotel_vector_db_langchain

# Hugging Face Spaces (Docker SDK) expects the app to listen on port 7860.
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
