"""
Agente de decisão (roteamento via Tool Calling) — Módulo 2.

Responsabilidade única: dado um histórico de conversa e uma pergunta nova,
decidir se responde direto ou aciona uma tool — SQLite determinístico
(buscar_docentes_por_departamento) ou ChromaDB semântico
(busca_vetorial_sigaa) — executar a tool escolhida, e devolver a resposta
final gerada pelo LLM.

Este é exatamente o componente que decide "qual pipeline seguir" entre
busca objetiva e busca semântica. Antes ficava embutido no meio de
teste_llm.py, misturado com wiring de infraestrutura e I/O de terminal.
Isolado aqui, este módulo não sabe nada sobre CLI ou Mastodon — só recebe
texto e devolve texto. É isso que permite reaproveitá-lo no futuro listener
do Mastodon (Módulo 3) sem duplicar o loop de decisão.
"""

from haystack.dataclasses import ChatMessage

from modulo2_inferencia.tools import TOOLS_SCHEMA, criar_dispatcher

SYSTEM_PROMPT = (
    "Você é o Grok Universitário da UFRRJ. "
    "Você é um agente autônomo. Tem ferramentas à sua disposição. "
    "Sempre decida qual ferramenta usar antes de responder. "
    "Responda num tom direto, prestativo e ligeiramente descontraído."
)


def montar_historico_inicial() -> list[ChatMessage]:
    """Ponto de partida de uma conversa nova, com o system prompt do agente."""
    return [ChatMessage.from_system(SYSTEM_PROMPT)]


def processar_pergunta(
    chat_generator,
    embedder,
    retriever,
    chat_history: list[ChatMessage],
    pergunta_usuario: str,
) -> tuple[str, list[ChatMessage]]:
    """
    Executa um turno completo do agente: decide, eventualmente aciona
    tool(s), e gera a resposta final.

    Retorna (texto_da_resposta, historico_atualizado) — sem imprimir nada e
    sem ler input nenhum, para ser chamável tanto pelo CLI quanto por um
    futuro listener do Mastodon.
    """
    dispatcher = criar_dispatcher(embedder, retriever)

    chat_history = [*chat_history, ChatMessage.from_user(pergunta_usuario)]

    # PASSO A: o LLM decide — responde direto ou pede uma tool
    resposta_llm = chat_generator.run(
        messages=chat_history, generation_kwargs={"tools": TOOLS_SCHEMA}
    )
    msg_resposta = resposta_llm["replies"][0]
    chat_history = [*chat_history, msg_resposta]

    # Sem tool call: o LLM respondeu direto (ex: "Bom dia!")
    if not msg_resposta.tool_calls:
        return msg_resposta.text, chat_history

    # PASSO B: com tool call(s) — executa cada uma
    for tool_call in msg_resposta.tool_calls:
        nome_tool = tool_call.tool_name
        argumentos = tool_call.arguments

        executar = dispatcher.get(nome_tool)
        if executar is None:
            # Antes (teste_llm.py), uma tool desconhecida virava uma string
            # vazia silenciosa. Aqui o próprio LLM recebe o motivo do erro
            # de volta, em vez de um resultado vazio sem explicação.
            resultado_da_tool = f"Erro: ferramenta '{nome_tool}' não existe."
        else:
            resultado_da_tool = executar(**argumentos)

        mensagem_tool = ChatMessage.from_tool(tool_result=resultado_da_tool, origin=tool_call)
        chat_history = [*chat_history, mensagem_tool]

    # PASSO C: manda o(s) resultado(s) de volta pro LLM gerar a resposta final
    resposta_final = chat_generator.run(messages=chat_history)
    msg_final = resposta_final["replies"][0]
    chat_history = [*chat_history, msg_final]

    return msg_final.text, chat_history
