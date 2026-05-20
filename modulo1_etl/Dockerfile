FROM python:3.11-slim

WORKDIR /app

# Instala dependências do sistema necessárias para lxml e sentence-transformers
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY parte1_scraping.py .
COPY parte2_inferencia.py .
COPY parte3_embedding.py .
COPY parte4_carga.py .

# Modelo de embedding baixado em build time — evita download a cada execução
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

CMD ["python", "parte4_carga.py"]