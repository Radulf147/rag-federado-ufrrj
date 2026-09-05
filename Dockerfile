# Projeto: Agente RAG Federado — UFRRJ
# Imagem principal do pipeline ETL + agente
#
# MOVIDO: este arquivo antes vivia em modulo1_etl/ com build context
# restrito a essa pasta. Agora vive na raiz porque o projeto passou a ter
# múltiplos pacotes de primeiro nível (modulo1_etl/, modulo2_inferencia/,
# interfaces/, config.py) que precisam entrar na mesma imagem.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# As duas convencoes de import do projeto coexistem por causa desta linha.
#
# modulo1_etl/ usa imports PLANOS entre seus arquivos (from db_manager import
# ...), enquanto modulo2_inferencia/ e interfaces/ usam imports QUALIFICADOS a
# partir da raiz (from modulo1_etl.db_manager import ...). Sem PYTHONPATH, cada
# forma so funcionava numa maneira de invocar:
#
#   python modulo1_etl/x.py    -> sys.path[0] = /app/modulo1_etl
#                                 planos OK, "import config" QUEBRA
#   python -m modulo1_etl.x    -> sys.path[0] = /app
#                                 qualificados OK, planos QUEBRAM
#
# O ETL roda pela primeira forma, e era por isso que nenhum arquivo do modulo 1
# podia usar config.py — dai parte4 e parte5 relerem os.getenv por conta
# propria, com defaults que podiam divergir do .env sem ninguem notar.
# Com as duas pastas no path, as duas formas funcionam.
ENV PYTHONPATH=/app:/app/modulo1_etl

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

# Copia o código do projeto, preservando a estrutura de pacotes
# (antes era tudo copiado "achatado" para dentro de /app)
COPY config.py             .
COPY modulo1_etl/          ./modulo1_etl/
COPY modulo2_inferencia/   ./modulo2_inferencia/
COPY interfaces/           ./interfaces/

# Volume para persistência do ChromaDB e logs
VOLUME ["/app/chroma_db", "/app/logs"]

# Cache dos modelos de embedding (evita re-download a cada container)
ENV SENTENCE_TRANSFORMERS_HOME=/app/models_cache
VOLUME ["/app/models_cache"]

# Comando padrão: roda o pipeline ETL completo (partes 1→5)
CMD ["python", "modulo1_etl/parte5_carga.py"]
