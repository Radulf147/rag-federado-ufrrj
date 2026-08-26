# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 5: carga no DocumentStore (ChromaDB)

import logging
import warnings
import os
from pathlib import Path
from datetime import datetime
from haystack import Document

try:
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore
except ImportError as exc:
    raise ImportError(
        "[STORE] ChromaDB não encontrado. Instale com: pip install chroma-haystack"
    ) from exc

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
CHROMA_HOST        = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT        = int(os.getenv("CHROMA_PORT", 8000))
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIM", 384))


def conectar_store(remoto: bool = False):
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
    if not documentos:
        log("[CARGA] Nenhum documento recebido.")
        return 0

    if store is None:
        store = conectar_store()

    sem_embedding = [i for i, d in enumerate(documentos) if not d.embedding]
    if sem_embedding:
        log(f"[CARGA] erro {len(sem_embedding)} docs sem embedding — execute a Parte 4 primeiro.")
        return 0

    # Remove campo interno do Haystack incompatível com ChromaDB
    for doc in documentos:
        doc.meta.pop("_split_overlap", None)

    if limpar_antes:
        try:
            ids_existentes = [d.id for d in store.filter_documents()]
            if ids_existentes:
                store.delete_documents(document_ids=ids_existentes)
                log("[CARGA] Coleção limpa.")
            else:
                log("[CARGA] Coleção já estava vazia.")
        except Exception as e:
            log(f"[CARGA] atenção Não foi possível limpar a coleção: {e}")

    # Chunks com conteúdo idêntico geram o mesmo hash de ID — ChromaDB rejeita duplicatas.
    vistos: set[str] = set()
    unicos: list[Document] = []
    for doc in documentos:
        if doc.id not in vistos:
            vistos.add(doc.id)
            unicos.append(doc)

    duplicatas = len(documentos) - len(unicos)
    if duplicatas:
        log(f"[CARGA] {duplicatas} chunks duplicados removidos antes da carga.")

    log(f"[CARGA] Gravando {len(unicos)} documentos...")

    try:
        store.write_documents(unicos)
    except Exception as e:
        log(f"[CARGA] erro Erro ao gravar: {e}")
        return 0

    total = store.count_documents()
    log(f"[CARGA] deu certo {total} documentos no store.")
    return total


def validar_carga(store, n_esperado: int) -> bool:
    erros = 0

    total = store.count_documents()
    if total < n_esperado:
        log(f"[VALIDAÇÃO] erro {total} documentos no store (esperado >= {n_esperado}).")
        erros += 1
    else:
        log(f"[VALIDAÇÃO] deu certo {total} documentos no store.")

    try:
        filtro = {
            "field":    "meta.instancia_dona",
            "operator": "==",
            "value":    INSTANCIA,
        }
        docs_da_instancia = store.filter_documents(filters=filtro)
        total_filtrado    = len(docs_da_instancia)

        if total_filtrado == 0 and n_esperado > 0:
            log(
                f"[VALIDAÇÃO] atenção filter_documents retornou 0 com filtro ativo "
                f"(n_esperado={n_esperado}). Verifique suporte a filtros na versão "
                f"instalada do chroma-haystack."
            )
        elif total_filtrado < n_esperado:
            log(
                f"[VALIDAÇÃO] erro Isolamento comprometido: {total_filtrado} docs com "
                f"instancia_dona='{INSTANCIA}' (esperado {n_esperado})."
            )
            erros += 1
        else:
            log(
                f"[VALIDAÇÃO] deu certo {total_filtrado} docs confirmados com "
                f"instancia_dona='{INSTANCIA}'."
            )

    except Exception as e:
        log(f"[VALIDAÇÃO] atenção Verificação de isolamento falhou: {e}")

    return erros == 0


if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 5 — CARGA NO DOCUMENTSTORE (ChromaDB)")
    log("=" * 60)

    from parte1_scraping_sigaa import scrape_sigaa
    from parte2_scraping_docentes import scrape_docentes
    from parte3_chunking import chunkar_documentos
    from parte4_embedding import embedar_documentos, validar_embeddings, MODELO_EMBEDDING

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chunks = chunkar_documentos(scrape_sigaa() + scrape_docentes())

    docs_embedados = embedar_documentos(chunks)

    if not docs_embedados:
        log("[RESULTADO] Pipeline ETL falhou.")
        exit(1)

    log("\n--- Validação pré-carga ---")
    if not validar_embeddings(docs_embedados):
        log("[RESULTADO] Validação pré-carga falhou — carga abortada.")
        exit(1)

    log(f"[SETUP] {len(docs_embedados)} documentos aprovados na validação. Iniciando carga.")

    CHROMA_REMOTE = os.getenv("CHROMA_REMOTE", "False").lower() in ("true", "1")
    store = conectar_store(remoto=CHROMA_REMOTE)
    total = carregar_documentos(docs_embedados, store, limpar_antes=True)

    if total == 0:
        log("[RESULTADO] Carga falhou.")
        exit(1)

    validar_carga(store, n_esperado=total)

    log("[PARTE 5 CONCLUÍDA]")