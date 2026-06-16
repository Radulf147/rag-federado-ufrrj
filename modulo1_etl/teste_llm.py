import os
import re
import json
from parte5_carga import conectar_store, INSTANCIA
from parte4_embedding import MODELO_EMBEDDING
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.builders import PromptBuilder

URL_SIGAA_BUSCA = "https://sigaa.ufrrj.br/sigaa/public/docente/busca_docentes.jsf?aba=p-academico"
LIMITE_EXIBICAO_SOCIAL = 5
MODELO_LLM    = os.getenv("MODELO_LLM", "mistral")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_REMOTE = os.getenv("CHROMA_REMOTE", "False").lower() in ("true", "1")
TOP_K         = 30
ARQUIVO_JSON  = "logs/docentes_departamentos.json"

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

def carregar_indice_estruturado() -> dict:
    if os.path.exists(ARQUIVO_JSON):
        with open(ARQUIVO_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def classificar_intencao_e_extrair_filtros(pergunta: str) -> dict:
    rota = "vetorial"
    departamento_alvo = None
    filtros = {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA}

    departamentos_conhecidos = {
        r"(ci[eê]ncia da computa[cç][aã]o|dcc)": "Ciência da Computação",
        r"(matem[aá]tica|dmat)": "Matemática",
        r"(f[ií]sica|dfis)": "Física",
        r"(qu[ií]mica|dqui)": "Química"
    }

    pergunta_limpa = pergunta.lower()

    padroes_exaustivos = r"(todos os|lista de|quais s[aã]o os) professores"
    if re.search(padroes_exaustivos, pergunta_limpa):
        rota = "estruturada"

    for padrao, nome_oficial in departamentos_conhecidos.items():
        if re.search(padrao, pergunta_limpa):
            departamento_alvo = nome_oficial
            filtros = {
                "operator": "AND",
                "conditions": [
                    {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
                    {"field": "meta.departamento", "operator": "==", "value": nome_oficial}
                ]
            }
            break

    return {"rota": rota, "departamento": departamento_alvo, "filtros_chroma": filtros}

store = conectar_store(remoto=CHROMA_REMOTE)
embedder = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
retriever = ChromaEmbeddingRetriever(document_store=store, top_k=TOP_K)
builder = PromptBuilder(template=TEMPLATE)
llm = OllamaGenerator(model=MODELO_LLM, url=OLLAMA_HOST)

indice_docentes = carregar_indice_estruturado()
embedder.warm_up()

print("=" * 60)
print("Agente RAG Federado - UFRRJ (Híbrido)")
print("=" * 60)

while True:
    pergunta = input("\nPergunta: ").strip()
    if pergunta.lower() in ("sair", "exit", "q"):
        break
    if not pergunta:
        continue

    analise = classificar_intencao_e_extrair_filtros(pergunta)

    # Rota 1: Busca Determinística (JSON)
    if analise["rota"] == "estruturada" and analise["departamento"]:
        print("\n[ROTEADOR] Intenção de listagem detectada.")
        
        # Busca tolerante (ignora maiúsculas/minúsculas e aceita nomes parciais)
        departamento_alvo = analise["departamento"].lower()
        chave_correta = None
        
        for depto_cadastrado in indice_docentes.keys():
            if departamento_alvo in depto_cadastrado.lower():
                chave_correta = depto_cadastrado
                break
                
        if chave_correta:
            professores = indice_docentes[chave_correta]
            total_professores = len(professores)
            
            if total_professores <= LIMITE_EXIBICAO_SOCIAL:
                lista_formatada = "\n".join(f"- {p}" for p in professores)
                print(f"\nResposta: O {chave_correta} possui {total_professores} docentes:\n{lista_formatada}")
            else:
                print(f"\nResposta: O {chave_correta} possui um total de {total_professores} professores cadastrados.")
                print(f"Para visualizar a relação completa, consulte o portal público: {URL_SIGAA_BUSCA}")
        else:
            print(f"\nResposta: Não localizei registros exatos para '{analise['departamento']}'.")
            if indice_docentes:
                print(f"[DEBUG] O banco possui chaves como: {list(indice_docentes.keys())[:3]}")
                
        continue

    # Rota 2: Busca Semântica (ChromaDB + LLM)
    query_vec = embedder.run(text=pergunta)["embedding"]
    docs = retriever.run(
        query_embedding=query_vec,
        filters=analise["filtros_chroma"]
    )["documents"]

    print(f"\n[RAG] Rota semântica ativada. {len(docs)} chunks recuperados.")
    
    prompt = builder.run(documents=docs, question=pergunta)["prompt"]

    print("[LLM] Gerando resposta...")
    resposta = llm.run(prompt=prompt)["replies"][0]
    print(f"\nResposta: {resposta}")