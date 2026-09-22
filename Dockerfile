FROM python:3.11-slim

WORKDIR /app

# System deps for pdf2image, playwright
RUN apt-get update && apt-get install -y \
    poppler-utils \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ARG LIGHTWEIGHT_MODE=false

COPY requirements.txt requirements-gpu.txt .
RUN pip install --no-cache-dir setuptools wheel
RUN grep -v "openai-whisper" requirements.txt | grep -v "ragas" | grep -v "^#" | grep -v "^$" > /tmp/req_no_whisper.txt && pip install --no-cache-dir -r /tmp/req_no_whisper.txt
RUN pip install --no-cache-dir pysbd appdirs
RUN pip install --no-cache-dir --no-deps ragas==0.1.21
RUN if [ "$LIGHTWEIGHT_MODE" = "false" ]; then \
      pip install --no-cache-dir -r requirements-gpu.txt; \
    fi
RUN pip install --no-cache-dir setuptools && pip install --no-cache-dir openai-whisper --no-build-isolation

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
