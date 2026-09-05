"""
Configuração centralizada — Agente RAG Federado UFRRJ.

Todas as variáveis de ambiente e constantes compartilhadas entre o Módulo 1
(ETL) e o Módulo 2 (Inferência) vivem aqui. Antes deste arquivo, INSTANCIA
estava hardcoded e repetida em parte1, parte2, parte4 e parte5 — e os
os.getenv() de embedding/chroma/llm estavam espalhados por parte4, parte5,
db_manager e teste_llm sem um lugar único de verdade.
"""

import os

# --- Identidade da instância (isolamento multi-tenant) ---
# TODO (ADR-001): hoje hardcoded; ao migrar para isolamento físico
# (Document Stores separados por instância), este valor deve vir de
# configuração por deployment, não de uma constante única no código.
INSTANCIA = "sigaa"

# --- Embedding ---
MODELO_EMBEDDING = os.getenv(
    "MODELO_EMBEDDING", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", 384))

# --- ChromaDB ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLECAO = f"rag_{INSTANCIA}"
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", 8000))
CHROMA_REMOTE = os.getenv("CHROMA_REMOTE", "False").lower() in ("true", "1")

# --- SQLite (Document Store genérico) ---
# O default aponta para dados/sigaa.db, que e onde o banco de fato vive no
# repositorio. Era so "sigaa.db", relativo ao diretorio de trabalho: rodando
# fora do Docker a partir da raiz, o SQLite ABRIA UM ARQUIVO NOVO E VAZIO em
# vez de reclamar, e a busca estruturada respondia, com toda a honestidade,
# que nao havia docentes. Dentro do container o docker-compose sobrepoe com
# /app/dados/sigaa.db, entao la nunca deu problema — e por isso passou tanto
# tempo sem ser notado.
DB_PATH = os.getenv("DB_PATH", "dados/sigaa.db")

# --- LLM / Ollama ---
MODELO_LLM = os.getenv("MODELO_LLM", "mistral")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", 10))

# --- HTTP scraping (antes duplicado entre parte1 e parte2) ---
HTTP_HEADERS = {
    "User-Agent": "UFRRJ-IC-RAG/1.0 (Iniciacao Cientifica - pesquisa academica)",
    "Accept-Language": "pt-BR,pt;q=0.9",
}
