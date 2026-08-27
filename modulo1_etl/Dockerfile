# Projeto: Agente RAG Federado — UFRRJ
# Imagem principal do pipeline ETL + agente
#
# NOTA (fix pós-refactor Set/2026): este Dockerfile agora fica na RAIZ do
# projeto, não em modulo1_etl/. O build context precisa ser a raiz porque
# requirements.txt, config.py, modulo2_inferencia/ e interfaces/ vivem lá,
# fora de modulo1_etl/.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

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

# Copia o projeto preservando a estrutura de pacotes.
# Isso é necessário porque os dois módulos usam convenções de import
# diferentes: modulo1_etl usa imports planos entre si (ex:
# "from db_manager import ..." em parte2_scraping_docentes.py), enquanto
# modulo2_inferencia importa modulo1_etl como pacote qualificado (ex:
# "from modulo1_etl.db_manager import ..." em tools.py). Achatar tudo em
# /app quebraria a segunda convenção.
COPY config.py .
COPY modulo1_etl/ modulo1_etl/
COPY modulo2_inferencia/ modulo2_inferencia/
COPY interfaces/ interfaces/

# PYTHONPATH cobre as duas convenções ao mesmo tempo:
# - /app permite "from modulo1_etl.db_manager import ..." e "import config"
# - /app/modulo1_etl permite "from db_manager import ..." (import plano)
ENV PYTHONPATH=/app:/app/modulo1_etl

# Volume para persistência do ChromaDB e logs
VOLUME ["/app/chroma_db", "/app/logs"]

# Cache dos modelos de embedding (evita re-download a cada container)
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache
VOLUME ["/app/models_cache"]

# Comando padrão: roda o pipeline ETL completo (partes 1→5).
# Chamado pelo caminho do arquivo (não como "-m") para que o Python
# adicione automaticamente modulo1_etl/ ao sys.path, resolvendo os
# imports planos entre parte1..parte5 e db_manager.
CMD ["python", "modulo1_etl/parte5_carga.py"]
