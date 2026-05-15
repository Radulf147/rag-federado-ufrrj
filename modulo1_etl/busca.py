from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.document_stores.chroma import ChromaDocumentStore
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

MODELO    = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INSTANCIA = "sigaa"

store = ChromaDocumentStore(
    collection_name=f"rag_{INSTANCIA}",
    persist_path="./chroma_db",
    embedding_function="default",
)

print(f"Total de documentos no store: {store.count_documents()}\n")

embedder  = SentenceTransformersTextEmbedder(model=MODELO)
embedder.warm_up()

perguntas = [
    "Quais docentes trabalham com inteligência artificial?",
    "Como autenticar documentos no SIGAA?",
    "Quem são os professores do departamento de computação?",
]

retriever = ChromaEmbeddingRetriever(document_store=store, top_k=3)
filtro    = {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA}

for pergunta in perguntas:
    print(f"PERGUNTA: {pergunta}")
    vetor      = embedder.run(text=pergunta)["embedding"]
    resultados = retriever.run(query_embedding=vetor, filters=filtro)["documents"]

    for i, doc in enumerate(resultados, 1):
        nome = doc.meta.get("nome_docente", "card SIGAA")
        print(f"  {i}. [{nome}] {doc.content[:120]}...")
    print()