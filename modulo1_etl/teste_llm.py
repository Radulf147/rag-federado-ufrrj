import os
import re
from parte5_carga import conectar_store, INSTANCIA
from parte4_embedding import MODELO_EMBEDDING
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder

MODELO_LLM    = os.getenv("MODELO_LLM", "mistral")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_REMOTE = os.getenv("CHROMA_REMOTE", "False").lower() in ("true", "1")
TOP_K         = 30

TEMPLATE = """
Você é o assistente oficial da UFRRJ.
Responda EXCLUSIVAMENTE com base nos documentos abaixo.
Se a informação não estiver nos documentos, diga:
"Não encontrei essa informação na base de conhecimento."
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

def extrair_filtros(pergunta: str) -> dict:
    # Filtro padrao basico
    filtros = {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA}

    # Mapeamento de variacoes textuais para o nome exato do metadado no ChromaDB
    departamentos_conhecidos = {
        r"(ci[eê]ncia da computa[cç][aã]o|dcc)": "Ciência da Computação",
        r"(matem[aá]tica|dmat)": "Matemática",
        r"(f[ií]sica|dfis)": "Física",
        r"(qu[ií]mica|dqui)": "Química"
    }

    pergunta_limpa = pergunta.lower()

    for padrao, nome_oficial in departamentos_conhecidos.items():
        if re.search(padrao, pergunta_limpa):
            # Substitui o filtro padrao por um operador lógico AND
            filtros = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
                    {"field": "meta.departamento", "operator": "==", "value": nome_oficial}
                ]
            }
            break

    return filtros

store     = conectar_store(remoto=CHROMA_REMOTE)
embedder  = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
retriever = ChromaEmbeddingRetriever(document_store=store, top_k=TOP_K)
builder   = PromptBuilder(template=TEMPLATE)
llm       = OllamaGenerator(model=MODELO_LLM, url=OLLAMA_HOST)

embedder.warm_up()

print("=" * 60)
print("Agente RAG Federado - UFRRJ")
print("=" * 60)

while True:
    pergunta = input("\nPergunta: ").strip()
    if pergunta.lower() in ("sair", "exit", "q"):
        break
    if not pergunta:
        continue

    filtros_dinamicos = extrair_filtros(pergunta)
    query_vec = embedder.run(text=pergunta)["embedding"]

    docs = retriever.run(
        query_embedding=query_vec,
        filters=filtros_dinamicos
    )["documents"]

    print(f"\n[RAG] {len(docs)} chunks recuperados.")
    
    prompt = builder.run(documents=docs, question=pergunta)["prompt"]

    print("[LLM] Gerando resposta...")
    resposta = llm.run(prompt=prompt)["replies"][0]
    print(f"\nResposta: {resposta}")