# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 4: carga no DocumentStore (ChromaDB)
# Autor: Raul Nascimento

import logging
import warnings
from pathlib import Path
from datetime import datetime
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore

try:
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
    CHROMA_DISPONIVEL = True
except ImportError:
    CHROMA_DISPONIVEL = False


Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/carga_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info


INSTANCIA          = "sigaa"
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLECAO     = f"rag_{INSTANCIA}"
CHROMA_HOST        = "localhost"   # trocar para o IP da faculdade em produção
CHROMA_PORT        = 8000
EMBEDDING_DIM      = 384           # deve coincidir com EMBEDDING_DIM da Parte 3


def conectar_store(remoto: bool = False):
    """
    Retorna um DocumentStore pronto para uso.
    remoto=True usa ChromaDB via HTTP (máquina da faculdade).
    Fallback para InMemoryDocumentStore se chromadb não estiver instalado.
    """
    if not CHROMA_DISPONIVEL:
        log("[STORE] chromadb não instalado — usando InMemoryDocumentStore.")
        return InMemoryDocumentStore()

    if remoto:
        log(f"[STORE] ChromaDB remoto: {CHROMA_HOST}:{CHROMA_PORT}")
        store = ChromaDocumentStore(
            collection_name=CHROMA_COLECAO,
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            embedding_function="default",
        )
    else:
        log(f"[STORE] ChromaDB local: {CHROMA_PERSIST_DIR}")
        store = ChromaDocumentStore(
            collection_name=CHROMA_COLECAO,
            persist_path=CHROMA_PERSIST_DIR,
            embedding_function="default",
        )

    log(f"[STORE] Coleção: '{CHROMA_COLECAO}'")
    return store


def carregar_documentos(
    documentos: list[Document],
    store=None,
    limpar_antes: bool = False,
) -> int:
    """
    Grava documentos vetorizados no store.
    limpar_antes=True apaga a coleção antes — usar na re-indexação semanal.
    Retorna o total de documentos no store após a carga.
    """
    if not documentos:
        log("[CARGA] Nenhum documento recebido.")
        return 0

    if store is None:
        store = conectar_store()

    sem_embedding = [i for i, d in enumerate(documentos) if not d.embedding]
    if sem_embedding:
        log(f"[CARGA] ✗ {len(sem_embedding)} docs sem embedding — execute a Parte 3 primeiro.")
        return 0

    # Remove campo interno do Haystack incompatível com ChromaDB
    for doc in documentos:
        doc.meta.pop("_split_overlap", None)

    if limpar_antes:
        try:
            store.delete_documents(document_ids=[d.id for d in store.filter_documents()])
            log("[CARGA] Coleção limpa.")
        except Exception as e:
            log(f"[CARGA] ⚠ Não foi possível limpar a coleção: {e}")

    log(f"[CARGA] Gravando {len(documentos)} documentos...")

    try:
        store.write_documents(documentos)
    except Exception as e:
        log(f"[CARGA] ✗ Erro ao gravar: {e}")
        return 0

    total = store.count_documents()
    log(f"[CARGA] ✓ {total} documentos no store.")
    return total


def validar_carga(store, n_esperado: int) -> bool:
    """Verifica total de documentos e isolamento por instancia_dona."""
    erros = 0

    total = store.count_documents()
    if total < n_esperado:
        log(f"[VALIDAÇÃO] ✗ {total} documentos no store (esperado >= {n_esperado}).")
        erros += 1
    else:
        log(f"[VALIDAÇÃO] ✓ {total} documentos no store.")

    try:
        amostra = store.filter_documents()[:20]
        errados = [d for d in amostra if d.meta.get("instancia_dona") != INSTANCIA]
        if errados:
            log(f"[VALIDAÇÃO] ✗ {len(errados)} docs com instancia_dona incorreto.")
            erros += 1
        else:
            log(f"[VALIDAÇÃO] ✓ instancia_dona = '{INSTANCIA}' confirmado.")
    except Exception as e:
        log(f"[VALIDAÇÃO] ⚠ Verificação de isolamento falhou: {e}")

    return erros == 0


if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 4 — CARGA NO DOCUMENTSTORE (ChromaDB)")
    log("=" * 60)

    from parte1_scraping import scrape_sigaa
    from parte2_inferencia import scrape_docentes, chunkar_documentos
    from parte3_embedding import embedar_documentos, MODELO_EMBEDDING

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = chunkar_documentos(scrape_sigaa() + scrape_docentes())

    docs_embedados = embedar_documentos(chunks)

    if not docs_embedados:
        log("[RESULTADO] Pipeline ETL falhou.")
        exit(1)

    log(f"[SETUP] {len(docs_embedados)} documentos prontos para carga.")

    store = conectar_store(remoto=False)   # trocar para remoto=True na produção
    total = carregar_documentos(docs_embedados, store, limpar_antes=True)

    if total == 0:
        log("[RESULTADO] Carga falhou.")
        exit(1)

    validar_carga(store, n_esperado=len(docs_embedados))

    log("\n--- Teste de busca ---")
    try:
        from haystack.components.embedders import SentenceTransformersTextEmbedder
        from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

        text_embedder = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
        text_embedder.warm_up()

        query      = "Quais são as áreas de pesquisa dos docentes de computação?"
        query_vec  = text_embedder.run(text=query)["embedding"]
        retriever  = ChromaEmbeddingRetriever(document_store=store, top_k=3)
        resultados = retriever.run(
            query_embedding=query_vec,
            filters={"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
        )["documents"]

        log(f"[BUSCA] '{query}'")
        for i, doc in enumerate(resultados, 1):
            log(f"  {i}. [{doc.meta.get('nome_docente', 'home')}] {doc.content[:100]}...")

    except Exception as e:
        log(f"[BUSCA] ⚠ Teste falhou: {e}")

    log(f"\n[RESUMO] {total} docs carregados | store: {CHROMA_PERSIST_DIR} | coleção: {CHROMA_COLECAO}")
    log("[PARTE 4 CONCLUÍDA — MÓDULO 1 ETL COMPLETO]")