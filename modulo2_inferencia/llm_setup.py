"""
Wiring de infraestrutura — Módulo 2 (Motor de Inferência).

Responsabilidade única: instanciar e conectar os componentes Haystack/Ollama
(document store, embedder, retriever, gerador de chat). Não decide nada, não
executa tools, não sabe se quem vai usar isso é o CLI ou o futuro listener
do Mastodon — só monta as peças e devolve prontas para uso.
"""

import os
from dataclasses import dataclass

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaChatGenerator

import config

# --- Parâmetros de geração do LLM (Ollama) ---
# Mesmo padrão de leitura de config.py (os.getenv com default), mas moradia
# aqui e não lá: são parâmetros de geração, e só o Módulo 2 gera texto — o
# ETL nunca instancia um LLM. config.py fica com o que é compartilhado
# entre os dois módulos.
#
# NUM_CTX: janela de contexto em tokens. O default do Ollama é 4096, apertado
# para o TOP_K de chunks recuperados via RAG. 8192 dá folga confortável para
# TOP_K=10 perfis de docente.
NUM_CTX = int(os.getenv("NUM_CTX", 8192))

# REASONING_EFFORT: "auto" | "off" | "low" | "medium" | "high".
#
# Corrigido em Set/2026: antes isto ia dentro de generation_kwargs, que a
# integração despeja no dict `options` do Ollama — onde chaves desconhecidas
# são ignoradas em silêncio. O controle real é o parâmetro `think` da API, no
# topo do payload. Ou seja, a configuração existia e não fazia nada.
#
# "auto" (default) não envia `think` nenhum e deixa o modelo no padrão dele.
# Isso importa: passar `think` para um modelo sem raciocínio configurável
# (como o qwen2.5, o modelo atual) é erro, não no-op.
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "auto")


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


def _valor_de_think(efeito: str):
    """Traduz REASONING_EFFORT para o parâmetro `think` do Ollama, ou None."""
    normalizado = (efeito or "").strip().lower()
    if normalizado in ("", "auto"):
        return None  # não enviar `think` — usa o padrão do modelo
    if normalizado in ("off", "false", "none", "no"):
        return False
    if normalizado in ("low", "medium", "high"):
        return normalizado
    raise ValueError(
        f"REASONING_EFFORT inválido: {efeito!r}. "
        "Use auto | off | low | medium | high."
    )


class _ClienteComThink:
    """
    Adaptador que injeta `think` em toda chamada ao Ollama.

    Existe porque a `ollama-haystack 2.2.0` não expõe `think` no `.run()` —
    ela chama `client.chat(model, messages, tools, stream, keep_alive,
    options)` e mais nada — enquanto o cliente `ollama` instalado já aceita o
    parâmetro. Envolver o cliente é menos invasivo que reescrever o `run()`
    inteiro da integração ou saltar da 2.2.0 para a 6.x no meio da fase de
    validação.

    Remover quando a integração for atualizada para uma versão que exponha
    `think` no construtor ou no run().
    """

    def __init__(self, cliente, think):
        self._cliente = cliente
        self._think = think

    def chat(self, **kwargs):
        return self._cliente.chat(think=self._think, **kwargs)

    def __getattr__(self, nome):
        return getattr(self._cliente, nome)


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

    # generation_kwargs vira o dict "options" da chamada ao Ollama, e vale para
    # todos os turnos. Só entram parâmetros de geração aqui: schemas de tool vão
    # pelo parâmetro `tools=` do .run(), em agent.py (ver criar_tools).
    chat_generator = OllamaChatGenerator(
        model=config.MODELO_LLM,
        url=config.OLLAMA_HOST,
        generation_kwargs={"num_ctx": NUM_CTX},
    )

    think = _valor_de_think(REASONING_EFFORT)
    if think is not None:
        chat_generator._client = _ClienteComThink(chat_generator._client, think)

    return ComponentesInferencia(
        store=store,
        embedder=embedder,
        retriever=retriever,
        chat_generator=chat_generator,
    )
