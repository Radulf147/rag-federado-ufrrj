import os
import json
from db_manager import buscar_entidades_por_campo
from parte5_carga import conectar_store, INSTANCIA
from parte4_embedding import MODELO_EMBEDDING
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall

# Configurações Iniciais
MODELO_LLM    = os.getenv("MODELO_LLM", "mistral")
OLLAMA_HOST   = os.getenv("OLLAMA_HOST", "http://localhost:11434")
CHROMA_REMOTE = False  
TOP_K         = 10

print("=" * 60)
print("Inicializando Motor Agentic RAG (Haystack + Ollama)...")
print("=" * 60)

# Inicializa os componentes de Busca Vetorial
store = conectar_store(remoto=CHROMA_REMOTE)
embedder = SentenceTransformersTextEmbedder(model=MODELO_EMBEDDING)
retriever = ChromaEmbeddingRetriever(document_store=store, top_k=TOP_K)
embedder.warm_up()

# Inicializa o Gerador de Chat do Haystack (suporta Agentic behavior)
chat_generator = OllamaChatGenerator(model=MODELO_LLM, url=OLLAMA_HOST)

# =====================================================================
# 1. DEFINIÇÃO DAS FERRAMENTAS (TOOLS) PARA O LLM
# =====================================================================

# O esquema JSON que diz ao LLM quais ferramentas ele possui e como usá-las.
tools_schema = [
    {
        "type": "function",
        "function": {
            "name": "buscar_docentes_por_departamento",
            "description": "Utilize esta ferramenta APENAS quando o usuário pedir para contar ou listar os professores/docentes de um departamento específico (ex: Computação, Física). Retorna dados exatos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "departamento": {
                        "type": "string",
                        "description": "Nome ou sigla do departamento que o usuário deseja buscar (ex: Ciência da Computação, Matemática)"
                    }
                },
                "required": ["departamento"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "busca_vetorial_sigaa",
            "description": "Utilize esta ferramenta para pesquisar descrições, ementas, ou responder perguntas genéricas interpretativas (ex: Quem pesquisa sobre Inteligência Artificial?). Busca em currículos completos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pergunta_semantica": {
                        "type": "string",
                        "description": "A pergunta otimizada para buscar no banco de dados vetorial."
                    }
                },
                "required": ["pergunta_semantica"]
            }
        }
    }
]

# =====================================================================
# 2. IMPLEMENTAÇÃO DAS FUNÇÕES LOCAIS (Executadas pelo Python)
# =====================================================================

def executar_tool_buscar_docentes(departamento: str) -> str:
    """Ferramenta Determinística - Consulta o SQLite (Schema-less)"""
    print(f"🔧 [TOOL EXECUTADA] Consulta estruturada em SQLite pelo departamento: {departamento}")
    
    resultados = buscar_entidades_por_campo("docente", "departamento", departamento)
    
    if not resultados:
        return f"Acesso à Base Estruturada: Não encontrei nenhum docente registrado sob o departamento '{departamento}'."
    
    nomes = sorted([r["nome"] for r in resultados])
    total = len(nomes)
    
    # Lógica Anti-Poluição Visual implementada dentro da ferramenta
    if total <= 10:
        lista = "\n- ".join(nomes)
        return f"Acesso à Base Estruturada: O departamento '{departamento}' tem {total} docentes. São eles:\n- {lista}"
    else:
        return f"Acesso à Base Estruturada: O departamento '{departamento}' tem um total de {total} docentes cadastrados. Não os listarei todos para poupar espaço."

def executar_tool_busca_vetorial(pergunta: str) -> str:
    """Ferramenta Semântica - Consulta o ChromaDB (Textos Livres)"""
    print(f"🧠 [TOOL EXECUTADA] Busca semântica em ChromaDB por: {pergunta}")
    
    # 1. Transforma o texto em embeddings
    query_vec = embedder.run(text=pergunta)["embedding"]
    # 2. Busca no banco
    docs = retriever.run(query_embedding=query_vec)["documents"]
    
    if not docs:
        return "Acesso à Base Vetorial: Nenhuma informação semântica relevante foi encontrada."
    
    # 3. Consolida os textos encontrados
    contexto = "\n---\n".join([d.content for d in docs])
    return f"Acesso à Base Vetorial. Documentos recuperados:\n{contexto}"


# =====================================================================
# 3. O LOOP DO AGENTE (A Orquestração)
# =====================================================================

chat_history = [
    ChatMessage.from_system(
        "Você é o Grok Universitário da UFRRJ. "
        "Você é um agente autônomo. Tem ferramentas à sua disposição. "
        "Sempre decida qual ferramenta usar antes de responder. "
        "Responda num tom direto, prestativo e ligeiramente descontraído."
    )
]

print("Agente pronto! Pode fazer perguntas estruturadas (ex: 'quantos professores tem a física?') ou interpretativas.")

while True:
    pergunta_usuario = input("\nVocê: ").strip()
    if pergunta_usuario.lower() in ("sair", "exit", "q"):
        break
    if not pergunta_usuario:
        continue

    # Adiciona a pergunta ao histórico
    chat_history.append(ChatMessage.from_user(pergunta_usuario))

    # PASSO A: Deixa o LLM pensar e decidir (Pode responder direto ou pedir uma Tool)
    resposta_llm = chat_generator.run(messages=chat_history, generation_kwargs={"tools": tools_schema})
    msg_resposta = resposta_llm["replies"][0]
    
    chat_history.append(msg_resposta) # Guardamos a decisão do LLM no histórico

    # PASSO B: O LLM decidiu usar uma ferramenta?
    if msg_resposta.tool_calls:
        for tool_call in msg_resposta.tool_calls:
            # Identifica a ferramenta e extrai os argumentos passados pelo LLM
            nome_tool = tool_call.tool_name
            argumentos = tool_call.arguments
            
            resultado_da_tool = ""
            
            if nome_tool == "buscar_docentes_por_departamento":
                resultado_da_tool = executar_tool_buscar_docentes(argumentos.get("departamento", ""))
            
            elif nome_tool == "busca_vetorial_sigaa":
                resultado_da_tool = executar_tool_busca_vetorial(argumentos.get("pergunta_semantica", pergunta_usuario))

            # Cria a mensagem com o resultado "cru" da ferramenta
            mensagem_tool = ChatMessage.from_tool(
                tool_result=resultado_da_tool,
                origin=tool_call
            )
            chat_history.append(mensagem_tool)

        # PASSO C: Manda o resultado da ferramenta de volta para o LLM gerar a resposta final em português bonito
        print("[LLM] Processando os dados recebidos das ferramentas...")
        resposta_final = chat_generator.run(messages=chat_history)
        msg_final = resposta_final["replies"][0]
        
        chat_history.append(msg_final)
        print(f"\nGrok UFRRJ: {msg_final.text}")
        
    else:
        # Se o LLM respondeu diretamente sem usar ferramentas (ex: "Bom dia!")
        print(f"\nGrok UFRRJ: {msg_resposta.text}")