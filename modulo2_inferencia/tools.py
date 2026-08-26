"""
Ferramentas (tools) do agente RAG — Módulo 2.

Cada tool tem duas partes:
1. O schema JSON — o que o LLM enxerga e usa para decidir quando chamar.
2. A implementação — o que o Python de fato executa quando é chamada.

Este arquivo NÃO decide quando uma tool é usada — quem decide é o LLM,
orquestrado pelo loop em agent.py. Aqui só vive "o que cada tool faz",
o que a torna testável isoladamente (ex: testar
buscar_docentes_por_departamento sem precisar do Ollama rodando).
"""

from modulo1_etl.db_manager import buscar_entidades_por_campo

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "buscar_docentes_por_departamento",
            "description": (
                "Utilize esta ferramenta APENAS quando o usuário pedir para "
                "contar ou listar os professores/docentes de um departamento "
                "específico (ex: Computação, Física). Retorna dados exatos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "departamento": {
                        "type": "string",
                        "description": (
                            "Nome ou sigla do departamento que o usuário deseja "
                            "buscar (ex: Ciência da Computação, Matemática)"
                        ),
                    }
                },
                "required": ["departamento"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "busca_vetorial_sigaa",
            "description": (
                "Utilize esta ferramenta para pesquisar descrições, ementas, "
                "ou responder perguntas genéricas interpretativas (ex: Quem "
                "pesquisa sobre Inteligência Artificial?). Busca em "
                "currículos completos."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pergunta_semantica": {
                        "type": "string",
                        "description": "A pergunta otimizada para buscar no banco de dados vetorial.",
                    }
                },
                "required": ["pergunta_semantica"],
            },
        },
    },
]


def buscar_docentes_por_departamento(departamento: str) -> str:
    """Ferramenta determinística — consulta o SQLite (schema-less)."""
    print(f"🔧 [TOOL EXECUTADA] Consulta estruturada em SQLite pelo departamento: {departamento}")

    resultados = buscar_entidades_por_campo("docente", "departamento", departamento)

    if not resultados:
        return (
            f"Acesso à Base Estruturada: Não encontrei nenhum docente "
            f"registrado sob o departamento '{departamento}'."
        )

    nomes = sorted(r["nome"] for r in resultados)
    total = len(nomes)

    if total <= 10:
        lista = "\n- ".join(nomes)
        return (
            f"Acesso à Base Estruturada: O departamento '{departamento}' tem "
            f"{total} docentes. São eles:\n- {lista}"
        )
    return (
        f"Acesso à Base Estruturada: O departamento '{departamento}' tem um "
        f"total de {total} docentes cadastrados. Não os listarei todos para "
        f"poupar espaço."
    )


def busca_vetorial_sigaa(pergunta: str, embedder, retriever) -> str:
    """
    Ferramenta semântica — consulta o ChromaDB (textos livres).

    Recebe embedder/retriever como parâmetros em vez de globais do módulo
    (como era em teste_llm.py) para poder ser testada com dublês/mocks sem
    precisar inicializar o Ollama ou o ChromaDB de verdade.
    """
    print(f"🧠 [TOOL EXECUTADA] Busca semântica em ChromaDB por: {pergunta}")

    query_vec = embedder.run(text=pergunta)["embedding"]
    docs = retriever.run(query_embedding=query_vec)["documents"]

    if not docs:
        return "Acesso à Base Vetorial: Nenhuma informação semântica relevante foi encontrada."

    contexto = "\n---\n".join(d.content for d in docs)
    return f"Acesso à Base Vetorial. Documentos recuperados:\n{contexto}"


def criar_dispatcher(embedder, retriever) -> dict:
    """
    Monta o dicionário nome_da_tool -> função executável.

    O agent.py não precisa conhecer a assinatura de cada tool — só chama
    dispatcher[nome](**argumentos_do_llm). Adicionar uma tool nova não exige
    tocar em agent.py, só registrar aqui.
    """
    return {
        "buscar_docentes_por_departamento": lambda departamento="": buscar_docentes_por_departamento(
            departamento
        ),
        "busca_vetorial_sigaa": lambda pergunta_semantica="": busca_vetorial_sigaa(
            pergunta_semantica, embedder, retriever
        ),
    }
