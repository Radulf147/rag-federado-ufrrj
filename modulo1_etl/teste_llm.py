# teste_llm.py
# Testa o pipeline completo: pergunta → busca no ChromaDB → resposta com phi3:mini

from parte5_carga import conectar_store, INSTANCIA
from parte4_embedding import MODELO_EMBEDDING
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder

MODELO_LLM    = os.getenv("MODELO_LLM", "mistral")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_REMOTE = os.getenv("CHROMA_REMOTE", "False").lower() in ("true", "1")
TOP_K         = 5

TEMPLATE = """
Você é o assistente oficial da UFRRJ.
Responda EXCLUSIVAMENTE com base nos documentos abaixo.
Se a informação não estiver nos documentos, diga:
"Não encontrei essa informação na base de conhecimento do SIGAA."
Não invente informações.

Documentos de referência:
{% for doc in documents %}
---
{{ doc.content }}
{% endfor %}
---

Pergunta: {{ question }}
Resposta:
"""

store     = conectar_store(remoto=CHROMA_REMOTE)
embedder  = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
retriever = ChromaEmbeddingRetriever(document_store=store, top_k=TOP_K)
builder   = PromptBuilder(template=TEMPLATE)
llm       = OllamaGenerator(model=MODELO_LLM, url=OLLAMA_HOST)

embedder.warm_up()

print("=" * 60)
print("Teste RAG — ChromaDB + phi3:mini")
print("Digite 'sair' para encerrar.")
print("=" * 60)

while True:
    pergunta = input("\nPergunta: ").strip()
    if pergunta.lower() in ("sair", "exit", "q"):
        break
    if not pergunta:
        continue

    query_vec = embedder.run(text=pergunta)["embedding"]

    docs = retriever.run(
        query_embedding=query_vec,
        filters={"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
    )["documents"]

    print(f"\n[RAG] {len(docs)} chunks recuperados:")
    for i, doc in enumerate(docs, 1):
        fonte = doc.meta.get("nome_docente") or doc.meta.get("titulo") or "home"
        print(f"  {i}. [{fonte}] {doc.content[:80]}...")

    prompt = builder.run(documents=docs, question=pergunta)["prompt"]

    print("\n[LLM] Gerando resposta...")
    resposta = llm.run(prompt=prompt)["replies"][0]
    print(f"\nResposta: {resposta}")