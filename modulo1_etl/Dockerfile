# Projeto: Agente RAG Federado — UFRRJ
# Imagem principal do pipeline ETL + agente

FROM python:3.11-slim

# Evita prompts interativos durante o build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Diretório de trabalho dentro do container
WORKDIR /app

# Dependências de sistema mínimas (lxml precisa de libxml2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev \
    libxslt-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python primeiro
# (camada separada do código — rebuild mais rápido)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código do projeto
COPY parte1_scraping_sigaa.py    .
COPY parte2_scraping_docentes.py .
COPY parte3_chunking.py          .
COPY parte4_embedding.py         .
COPY parte5_carga.py             .
COPY teste_llm.py                .

# Volume para persistência do ChromaDB e logs
VOLUME ["/app/chroma_db", "/app/logs"]

# Cache dos modelos de embedding (evita re-download a cada container)
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache
VOLUME ["/app/models_cache"]

# Comando padrão: roda o pipeline ETL completo (partes 1→5)
CMD ["python", "parte5_carga.py"]
