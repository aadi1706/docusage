FROM python:3.11-slim

WORKDIR /app

# System deps for pdf2image, playwright
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir $(grep -v openai-whisper requirements.txt | grep -v '^#' | grep -v '^$' | tr '\n' ' ')
RUN pip install --no-cache-dir git+https://github.com/openai/whisper.git

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
