FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libwebp7 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

COPY requirements.txt requirements-local-embeddings.txt ./
RUN pip install -r requirements.txt

# In-process embeddings are optional. The slim image (default) is ~400 MB and
# runs in well under 300 MB of RAM; set WITH_LOCAL_EMBEDDINGS=1 only on a
# machine with a few GB to spare.
ARG WITH_LOCAL_EMBEDDINGS=0
ARG EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
RUN if [ "$WITH_LOCAL_EMBEDDINGS" = "1" ]; then \
        pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 && \
        pip install -r requirements-local-embeddings.txt && \
        python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')" ; \
    fi

COPY bot ./bot

CMD ["python", "-m", "bot.main"]
