"""
Wiring de infraestrutura — Módulo 2 (Motor de Inferência).

Responsabilidade única: instanciar e conectar os componentes Haystack/Ollama
(document store, embedder, retriever, gerador de chat). Não decide nada, não
executa tools, não sabe se quem vai usar isso é o CLI ou o futuro listener
do Mastodon — só monta as peças e devolve prontas para uso.
"""

from dataclasses import dataclass

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

import config


def _conectar_store() -> ChromaDocumentStore:
    """
    Conecta ao mesmo ChromaDB populado pelo Módulo 1 (ETL).

    NOTA — duplicação deliberada: esta função repete a lógica de
    conectar_store() de modulo1_etl/parte5_carga.py, em vez de importá-la
    de lá. Motivo: parte5_carga.py tem efeito colateral no nível do módulo
    (cria a pasta ./logs/ e chama logging.basicConfig ao ser importado, não
    só ao ser executado) — importar essa função de lá acoplaria o Módulo 2
    a esse efeito colateral. É o mesmo tipo de problema estrutural apontado
    na análise do Módulo 1. Quando parte5_carga.py for limpo desse efeito
    colateral de import, esta função deve ser removida e substituída por um
    import de um `shared/chroma_store.py` único.
    """
    if config.CHROMA_REMOTE:
        return ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            embedding_function="default",
        )
    return ChromaDocumentStore(
        collection_name=config.CHROMA_COLECAO,
        persist_path=config.CHROMA_PERSIST_DIR,
        embedding_function="default",
    )


@dataclass
class ComponentesInferencia:
    """Pacote com tudo que o agente precisa para responder uma pergunta."""

    store: ChromaDocumentStore
    embedder: SentenceTransformersTextEmbedder
    retriever: ChromaEmbeddingRetriever
    chat_generator: OllamaChatGenerator


def montar_componentes() -> ComponentesInferencia:
    """Conecta e inicializa (warm-up incluso) tudo que o agente precisa."""
    store = _conectar_store()

    embedder = SentenceTransformersTextEmbedder(model=config.MODELO_EMBEDDING)
    embedder.warm_up()

    retriever = ChromaEmbeddingRetriever(document_store=store, top_k=config.TOP_K)

    chat_generator = OllamaChatGenerator(model=config.MODELO_LLM, url=config.OLLAMA_HOST)

    return ComponentesInferencia(
        store=store,
        embedder=embedder,
        retriever=retriever,
        chat_generator=chat_generator,
    )
