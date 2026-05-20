# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 7: Script de Teste de Busca Semântica

import logging
from pathlib import Path
from datetime import datetime
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

# Reutiliza as configurações e conectores das etapas anteriores
from parte4_embedding import MODELO_EMBEDDING
from parte5_carga import conectar_store, INSTANCIA

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/busca_teste_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info

if __name__ == "__main__":
    log("=" * 60)
    log("BUSCA — TESTE DE BUSCA SEMÂNTICA NO CHROMADB")
    log("=" * 60)

    # Conecta no store remoto (Docker)
    store = conectar_store(remoto=False)

    try:
        text_embedder = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
        text_embedder.warm_up()

        query = "Quais são as áreas de pesquisa dos docentes de computação?"
        log(f"[BUSCA] Vetorizando a consulta: '{query}'")
        
        query_vec = text_embedder.run(text=query)["embedding"]
        retriever = ChromaEmbeddingRetriever(document_store=store, top_k=3)
        
        resultados = retriever.run(
            query_embedding=query_vec,
            filters={"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
        )["documents"]

        log(f"[RESULTADO] {len(resultados)} documentos encontrados para a busca.")
        for i, doc in enumerate(resultados, 1):
            log(f"  {i}. [{doc.meta.get('nome_docente', 'home')}] {doc.content[:120]}...")

    except Exception as e:
        log(f"[BUSCA] atenção Teste falhou: {e}")

    log("[busca CONCLUÍDA]")