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
# ⚠️ SOBREPONÍVEL POR AMBIENTE DESDE 7 SET 2026 — e antes disso NÃO ERA.
#
# `modulo1_etl/reindexar_descritivo.py` afirma, no próprio docstring, que
# "voltar atrás é trocar uma variável de ambiente". Era falso: a coleção saía
# só de INSTANCIA, que é constante no código, e não havia variável nenhuma
# para trocar. Trocar a coleção de produção exigia editar este arquivo.
#
# A frase não deu erro nenhum, porque ninguém tinha tentado trocar ainda. Ela
# descrevia uma reversibilidade que o código não tinha — e reversibilidade
# barata foi o argumento usado para autorizar a reindexação. Mesmo padrão do
# §10 do relatório: a saída plausível que ninguém foi conferir.
#
# O default preserva EXATAMENTE o valor anterior. Nada muda para quem não
# definir a variável; o que muda é que agora existe o que trocar.
CHROMA_COLECAO = os.getenv("CHROMA_COLECAO", f"rag_{INSTANCIA}")
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

# --- Rede simulada (interfaces/rede) ---
# Banco SEPARADO do sigaa.db de proposito. O ETL trata cada execucao como um
# retrato completo e apaga as linhas do tipo que vai recarregar
# (salvar_entidades(..., substituir=True), achado 10). Post de usuario nao e
# entidade do SIGAA e nao pode estar sujeito a esse ciclo: uma recarga do ETL
# apagaria a conversa junto, sem erro nenhum.
REDE_DB_PATH = os.getenv("REDE_DB_PATH", "dados/rede.db")

# Identificador do bot na rede simulada. Mencionar isto num post e o gesto que
# aciona o agente.
BOT_HANDLE = os.getenv("BOT_HANDLE", "@ufrrj")

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
