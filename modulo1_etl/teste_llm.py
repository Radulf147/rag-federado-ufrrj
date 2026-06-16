import os
import re
import json
from rapidfuzz import process, fuzz
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
CHROMA_REMOTE = False  # Força a usar o banco local/docker da mesma rede
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

class RoteadorSigaa:
    def __init__(self, dict_path: str):
        self.indice_docentes = {}
        if os.path.exists(dict_path):
            with open(dict_path, "r", encoding="utf-8") as f:
                self.indice_docentes = json.load(f)
        
        self.departamentos_oficiais = list(self.indice_docentes.keys())
        
        # Mapeamento determinístico para acrônimos comuns
        self.sinonimos = {
            "dcc": "DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
            "dmat": "DEPARTAMENTO DE MATEMÁTICA",
            "dfis": "DEPARTAMENTO DE FÍSICA",
            "dqui": "DEPARTAMENTO DE QUÍMICA",
        }

    def normalizar_departamento(self, query: str, threshold: int = 70) -> str | None:
        """Processa a string extraída e retorna a chave oficial do ChromaDB."""
        query_lower = query.lower().strip()
        
        if query_lower in self.sinonimos:
            return self.sinonimos[query_lower]

        match, score, _ = process.extractOne(
            query_lower, 
            self.departamentos_oficiais, 
            scorer=fuzz.WRatio
        )
        
        if score >= threshold:
            return match
            
        return None

    def classificar_intencao(self, pergunta: str) -> dict:
        rota = "vetorial"
        departamento_exato = None
        filtros = {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA}

        pergunta_limpa = pergunta.lower()
        padroes_exaustivos = r"(todos os|lista de|quais s[aã]o os) professores\s+(do|da|de)?\s*(.*)"
        
        match_intencao = re.search(padroes_exaustivos, pergunta_limpa)
        
        if match_intencao:
            rota = "estruturada"
            entidade_bruta = match_intencao.group(3).strip()
            
            if entidade_bruta:
                departamento_exato = self.normalizar_departamento(entidade_bruta)

            if departamento_exato:
                filtros = {
                    "operator": "AND",
                    "conditions": [
                        {"field": "meta.instancia_dona", "operator": "==", "value": INSTANCIA},
                        {"field": "meta.departamento", "operator": "==", "value": departamento_exato}
                    ]
                }

        return {
            "rota": rota, 
            "departamento_exato": departamento_exato, 
            "filtros_chroma": filtros
        }

roteador = RoteadorSigaa(ARQUIVO_JSON)
store = conectar_store(remoto=CHROMA_REMOTE)
embedder = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
retriever = ChromaEmbeddingRetriever(document_store=store, top_k=TOP_K)
builder = PromptBuilder(template=TEMPLATE)
llm = OllamaGenerator(model=MODELO_LLM, url=OLLAMA_HOST)

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

    analise = roteador.classificar_intencao(pergunta)

    # Rota 1: Busca Determinística (JSON)
    if analise["rota"] == "estruturada":
        print("\n[ROTEADOR] Intenção de listagem detectada.")
        
        depto = analise["departamento_exato"]
        if depto:
            professores = roteador.indice_docentes.get(depto, [])
            total_professores = len(professores)
            
            if total_professores <= LIMITE_EXIBICAO_SOCIAL:
                lista_formatada = "\n".join(f"- {p}" for p in professores)
                print(f"\nResposta: O {depto} possui {total_professores} docentes:\n{lista_formatada}")
            else:
                print(f"\nResposta: O {depto} possui um total de {total_professores} professores cadastrados.")
                print(f"Para visualizar a relação completa, consulte o portal público: {URL_SIGAA_BUSCA}")
        else:
            print("\nResposta: Não consegui identificar o departamento com precisão para realizar a listagem.")
            print("Por favor, reformule a pergunta utilizando a sigla ou o nome do curso.")
            
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