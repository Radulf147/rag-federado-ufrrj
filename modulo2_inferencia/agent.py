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

import json
import os

from haystack.dataclasses import ChatMessage

from modulo2_inferencia.tools import criar_dispatcher, criar_tools

# Quantas vezes o LLM pode falar antes de ser obrigado a concluir. A última
# rodada é sempre feita SEM tools, o que força uma resposta em texto em vez de
# mais uma chamada — o teto termina a conversa, não a trunca no meio.
# 4 cobre com folga o encadeamento mais longo previsto (estruturado → semântico
# → resposta); serve de rede contra laço, não de orçamento apertado.
MAX_RODADAS_TOOL = int(os.getenv("MAX_RODADAS_TOOL", 4))

if MAX_RODADAS_TOOL < 2:
    # Com 1, a única rodada seria a última e portanto rodaria SEM tools: o
    # agente jamais receberia ferramenta nenhuma e responderia de cabeça,
    # parecendo funcionar. É literalmente o defeito silencioso que custou uma
    # bateria inteira de testes em Set/2026 — falhar aqui, alto, é mais barato.
    raise ValueError(
        f"MAX_RODADAS_TOOL={MAX_RODADAS_TOOL} é inválido: precisa ser >= 2, "
        "porque a última rodada é feita sem tools de propósito."
    )

# A redação anterior era: "Você é um agente autônomo. Tem ferramentas à sua
# disposição. Sempre decida qual ferramenta usar antes de responder." Ela não
# autorizava o agente a não saber, e o empurrava a sempre produzir um achado a
# partir do que a ferramenta devolvesse. O resultado medido em Set/2026 foi
# especulação apresentada como fato ("é possível que ela esteja envolvida nessa
# área dada sua experiência"). O contraste é a prova: diante do mesmo corpus, o
# pipeline vetorial puro — que não recebe este prompt — respondeu com cautela.
# A diferença entre os dois não era o modelo nem a recuperação, era a instrução.
SYSTEM_PROMPT = (
    "Você é o Grok Universitário da UFRRJ. Responde perguntas sobre os docentes "
    "usando exclusivamente os dados institucionais do SIGAA aos quais tem acesso "
    "pelas suas ferramentas.\n\n"
    "COMO DECIDIR O CAMINHO\n"
    "- Contagem, listagem ou vínculo de docente a departamento são dados exatos: "
    "use a busca estruturada.\n"
    "- Conteúdo de perfil (formação, áreas de interesse, atuação) é "
    "interpretativo: use a busca semântica.\n"
    "- Perguntas que combinam os dois (ex.: 'quem da Matemática pesquisa "
    "estatística?') pedem primeiro o recorte estruturado, depois o semântico.\n"
    "- Você pode chamar uma ferramenta, ler o resultado e só então decidir a "
    "próxima. Use isso quando o segundo passo depender do que o primeiro "
    "devolveu, em vez de tentar adivinhar as duas chamadas de uma vez.\n"
    "- Se a pergunta não for sobre docentes da UFRRJ, diga que está fora do seu "
    "escopo, sem acionar ferramenta.\n\n"
    "COMO RESPONDER\n"
    "- Afirme apenas o que os dados recuperados dizem explicitamente. Nunca "
    "deduza a área de pesquisa de alguém a partir do departamento, do título ou "
    "do nome de uma disciplina. Se o dado não diz, você não sabe.\n"
    "- Você pode não saber. 'Não encontrei essa informação' é uma resposta "
    "correta e sempre preferível a uma suposição.\n"
    "- Se encontrar apenas parte do pedido, responda com o que encontrou e diga "
    "que é parcial.\n"
    "- Se a informação falta porque o docente não preencheu o campo no SIGAA, "
    "diga isso: a lacuna é do cadastro, não da busca.\n"
    "- Quando souber onde a informação provavelmente está — outra aba do perfil "
    "do docente, o currículo Lattes, a secretaria do departamento — aponte o "
    "caminho em vez de encerrar com um 'não encontrei' seco.\n\n"
    "Tom: direto, prestativo, levemente descontraído. Nunca inventar."
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
    tools = criar_tools(embedder, retriever)

    chat_history = [*chat_history, ChatMessage.from_user(pergunta_usuario)]

    # POR QUE UM LAÇO, E NÃO TRÊS PASSOS FIXOS (corrigido em Set/2026):
    # a versão anterior fazia exatamente uma rodada de tools — pedia, executava,
    # e a chamada final ia SEM tools. O agente nunca conseguia usar o RESULTADO
    # de uma ferramenta para decidir a próxima. Isso contradizia o próprio
    # SYSTEM_PROMPT, que manda "primeiro o recorte estruturado, depois o
    # semântico", e só parecia funcionar porque o modelo emitia as duas chamadas
    # na mesma resposta — acerto de primeira, não encadeamento. Justo a classe
    # ambígua, que é a mais informativa da métrica de roteamento (§3), dependia
    # dessa sorte.
    #
    # As tools são anunciadas em toda rodada menos a última; na última elas são
    # omitidas de propósito, para que o modelo conclua em vez de pedir mais.
    chamadas_ja_feitas: set[tuple[str, str]] = set()

    for rodada in range(1, MAX_RODADAS_TOOL + 1):
        ultima_rodada = rodada == MAX_RODADAS_TOOL

        # As tools vão no parâmetro `tools=`, NÃO em generation_kwargs: o que
        # entra em generation_kwargs vira o dict `options` da chamada ao Ollama,
        # onde schemas de tool são ignorados (ver criar_tools em tools.py).
        resposta_llm = chat_generator.run(
            messages=chat_history,
            **({} if ultima_rodada else {"tools": tools}),
        )
        msg_resposta = resposta_llm["replies"][0]
        chat_history = [*chat_history, msg_resposta]

        # Sem tool call: o LLM respondeu direto (ex.: "Bom dia!") ou concluiu.
        if not msg_resposta.tool_calls:
            break

        for tool_call in msg_resposta.tool_calls:
            nome_tool = tool_call.tool_name
            argumentos = tool_call.arguments

            # Repetir a mesma chamada com os mesmos argumentos não traz dado
            # novo, só queima uma rodada. Devolver isso como erro explícito
            # informa o modelo, em vez de deixá-lo repetir até o teto.
            assinatura = (nome_tool, json.dumps(argumentos, sort_keys=True, ensure_ascii=False))

            executar = dispatcher.get(nome_tool)
            if executar is None:
                # Antes (teste_llm.py), uma tool desconhecida virava uma string
                # vazia silenciosa. Aqui o próprio LLM recebe o motivo do erro
                # de volta, em vez de um resultado vazio sem explicação.
                resultado_da_tool = f"Erro: ferramenta '{nome_tool}' não existe."
            elif assinatura in chamadas_ja_feitas:
                resultado_da_tool = (
                    f"Erro: '{nome_tool}' já foi chamada com estes mesmos argumentos "
                    "nesta conversa, e o resultado está acima. Use o que já tem, "
                    "chame outra ferramenta, ou responda."
                )
            else:
                chamadas_ja_feitas.add(assinatura)
                resultado_da_tool = executar(**argumentos)

            mensagem_tool = ChatMessage.from_tool(tool_result=resultado_da_tool, origin=tool_call)
            chat_history = [*chat_history, mensagem_tool]

    msg_final = chat_history[-1]

    # Resposta vazia é falha, não resposta. Alguns modelos com raciocínio
    # (medido no gpt-oss:20b) despejam tudo no canal de análise e não emitem
    # nada no canal final — done_reason='stop', eval_count>0, texto ''. Antes
    # isso subia como string vazia e aparecia no relatório de testes como uma
    # citação em branco, indistinguível de "o modelo não tinha o que dizer".
    if not (msg_final.text or "").strip():
        return (
            "[FALHA] O modelo não produziu resposta final "
            f"(done_reason={msg_final.meta.get('done_reason')}, "
            f"tokens gerados={msg_final.meta.get('eval_count')}). "
            "Em modelos com raciocínio, costuma ser resposta presa no canal de "
            "análise — ver REASONING_EFFORT em llm_setup.py.",
            chat_history,
        )

    return msg_final.text, chat_history
