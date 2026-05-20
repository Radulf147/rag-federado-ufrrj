# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 6: Orquestrador da Pipeline ETL

import logging
from datetime import datetime
from pathlib import Path
from parte1_scraping_home import scrape_sigaa
from parte2_scraping_docentes import scrape_docentes
from parte3_chunking import chunkar_documentos
from parte4_embedding import embedar_documentos, validar_embeddings
from parte5_carga import conectar_store, carregar_documentos, validar_carga

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/orquestrador_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info

if __name__ == "__main__":
    log("\n")
    log("[PIPELINE] Executando o fluxo unificado de dados do ecossistema RAG")
    log("\n" )
    
    # Passo 1: Extração da Home, raspando os serviços públicos gerais da página inicial do SIGAA.
    log("[ETL - EXTRAÇÃO] Coletando informacoes institucionais da pagina publica inicial.")
    docs_home = scrape_sigaa()
    
    # PASSO 2: Extração de Docentes -> Navegando via POST e baixando os perfis em paralelo.
    log("[ETL - EXTRAÇÃO] Executando navegacao dinamica e download paralelo dos perfis dos docentes.")
    docs_docentes = scrape_docentes()
    
    # Consolidação do lote de documentos brutos na memória operacional
    documentos_brutos = docs_home + docs_docentes
    
    if not documentos_brutos:
        log("[ERRO CRÍTICO] Falha na extracao de dados. Pipeline abortada devido a ausencia de documentos.")
        exit(1)
        
    # PASSO 3: Chunking- Quebrando os textos longos em pedaços de 5 sentenças para não estourar o limite da LLM.
    log("[ETL - TRANSFORMAÇÃO] Aplicando quebra estrutural e sobreposicao semantica nos textos.")
    chunks = chunkar_documentos(documentos_brutos)
    
    # PASSO 4: Vetorização (Embedding) -> Convertendo o texto em matrizes matemáticas (vetores).
    log("[ETL - TRANSFORMAÇÃO] Convertendo os fragmentos de texto em vetores numericos de alta densidade.")
    chunks_vetorizados = embedar_documentos(chunks)
    
    # Auditoria de segurança e consistência dos metadados antes da persistência
    if not validar_embeddings(chunks_vetorizados):
        log("[ERRO CRÍTICO] Inconsistencia detectada nos vetores ou metadados de governanca. Pipeline abortada.")
        exit(1)
    
    # PASSO 5: Carga e Criação do Banco -> Salvando os vetores no ChromaDB para busca futura.
    log("[ETL - CARGA] Inicializando conexao remota com o banco de dados vetorial ChromaDB.")
    store = conectar_store(remoto=True)
    
    log("[ETL - CARGA] Persistindo a colecao vetorizada e atualizando o indice de busca.")
    total_gravado = carregar_documentos(chunks_vetorizados, store, limpar_antes=True)
    
    # Auditoria final de isolamento e volumetria dos dados gravados em disco
    if validar_carga(store, n_esperado=len(chunks_vetorizados)):
        log("[SUCESSO] Pipeline ETL concluida. Banco de dados atualizado e homologado para busca.")
    else:
        log("[AVISO] Carga finalizada, mas a auditoria de integridade acusou inconformidades nos metadados.")