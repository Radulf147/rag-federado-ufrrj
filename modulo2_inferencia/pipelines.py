"""
Os três pipelines da fase de validação (CLAUDE.md §3) — Módulo 2.

Existe para responder à mesma pergunta por três caminhos diferentes e permitir
compará-los lado a lado:

1. `responder_vetorial`    — busca semântica pura (ChromaDB), sem tool calling
2. `responder_estruturado` — consulta determinística ao SQLite, sem vetor store
3. `responder_agente`      — o agente com tool calling, que decide entre os dois

Os três têm a MESMA assinatura — (componentes, pergunta) -> ResultadoPipeline —
porque é isso que permite ao runner tratá-los como intercambiáveis e produzir
uma tabela comparativa honesta. Nenhum deles imprime nada nem lê input.

NOTA — o pipeline 3 não é reimplementado aqui: ele delega para
agent.processar_pergunta, que já é o loop de decisão de verdade. Duplicá-lo
faria a comparação medir uma cópia em vez do agente real.
"""

from dataclasses import dataclass, field

from haystack.dataclasses import ChatMessage
from rapidfuzz import fuzz, process

from modulo1_etl.db_manager import buscar_entidades_por_campo
from modulo2_inferencia.agent import montar_historico_inicial, processar_pergunta


@dataclass
class ResultadoPipeline:
    """
    O que um pipeline devolve. `fontes` é o que responde à pergunta do
    CLAUDE.md "usou a fonte certa de dado?" — sem isso a comparação viraria
    só um julgamento de texto.
    """

    pipeline: str
    pergunta: str
    resposta: str
    fontes: list[str] = field(default_factory=list)
    detalhe: str = ""


# --- Pipeline 1: busca semântica pura -------------------------------------

PROMPT_VETORIAL = (
    "Você é o Grok Universitário da UFRRJ. Responda à pergunta usando "
    "EXCLUSIVAMENTE os documentos abaixo. Se a resposta não estiver neles, "
    "diga que não encontrou — não invente.\n\n"
    "DOCUMENTOS:\n{contexto}\n\nPERGUNTA: {pergunta}"
)


def responder_vetorial(componentes, pergunta: str) -> ResultadoPipeline:
    """Retrieval puro + geração. Sem tools: o LLM só vê os chunks recuperados."""
    vetor = componentes.embedder.run(text=pergunta)["embedding"]
    docs = componentes.retriever.run(query_embedding=vetor)["documents"]

    if not docs:
        return ResultadoPipeline(
            pipeline="1-vetorial",
            pergunta=pergunta,
            resposta="Nenhum documento relevante recuperado.",
            fontes=[],
            detalhe="retriever devolveu 0 documentos",
        )

    contexto = "\n---\n".join(d.content for d in docs)
    prompt = PROMPT_VETORIAL.format(contexto=contexto, pergunta=pergunta)

    resposta = componentes.chat_generator.run(messages=[ChatMessage.from_user(prompt)])
    nomes = [d.meta.get("nome_docente", "?") for d in docs]

    return ResultadoPipeline(
        pipeline="1-vetorial",
        pergunta=pergunta,
        resposta=resposta["replies"][0].text,
        fontes=["chromadb"],
        detalhe=f"{len(docs)} chunks recuperados: {', '.join(nomes)}",
    )


# --- Pipeline 2: banco estruturado ----------------------------------------

# Limiar de similaridade para casar o departamento citado na pergunta com um
# departamento real do banco. Abaixo disso, preferimos admitir que não
# entendemos a pergunta a devolver o departamento errado com confiança.
LIMIAR_FUZZY = 70


def _departamentos_conhecidos() -> list[str]:
    """Lista os departamentos distintos que existem de fato no SQLite."""
    todos = buscar_entidades_por_campo("docente", "departamento", "")
    return sorted({d["departamento"] for d in todos if d.get("departamento")})


def _departamento_da_pergunta(pergunta: str) -> tuple[str | None, float]:
    """
    Resolve, sem LLM, qual departamento a pergunta menciona.

    ESCOLHA DE PROJETO: o pipeline 2 é "sem LLM" por definição (CLAUDE.md §3),
    então precisa de algum jeito determinístico de sair da linguagem natural e
    chegar num parâmetro de query. Usamos rapidfuzz — que já era dependência
    declarada do projeto e não estava sendo usada em lugar nenhum — casando a
    pergunta contra os departamentos reais do banco. É deliberadamente burro:
    a graça da comparação é justamente ver onde esse caminho barato acerta
    tanto quanto o agente e onde ele quebra.
    """
    candidatos = _departamentos_conhecidos()
    if not candidatos:
        return None, 0.0

    # processor=str.lower é obrigatório, não cosmético: os departamentos vêm do
    # SIGAA em CAIXA ALTA e as perguntas em caixa mista, e o rapidfuzz compara
    # caractere a caractere. Sem isso, "Departamento de Matemática" contra
    # "DEPARTAMENTO DE MATEMÁTICA" pontua 15 em vez de 100 — medido — e o
    # pipeline 2 rejeitava TODAS as perguntas por ficar abaixo do limiar.
    achado = process.extractOne(
        pergunta, candidatos, scorer=fuzz.partial_ratio, processor=str.lower
    )
    if achado is None or achado[1] < LIMIAR_FUZZY:
        return None, (achado[1] if achado else 0.0)
    return achado[0], achado[1]


def responder_estruturado(componentes, pergunta: str) -> ResultadoPipeline:
    """
    Consulta direta ao SQLite. Não passa pelo vetor store e não chama o LLM —
    a "resposta" é montada por template, o que a torna incapaz de alucinar e
    incapaz de responder qualquer coisa interpretativa.
    """
    departamento, score = _departamento_da_pergunta(pergunta)

    if departamento is None:
        return ResultadoPipeline(
            pipeline="2-estruturado",
            pergunta=pergunta,
            resposta=(
                "Não consegui identificar um departamento na pergunta — este "
                "pipeline só responde a perguntas objetivas sobre departamentos."
            ),
            fontes=[],
            detalhe=f"melhor score fuzzy {score:.0f} < limiar {LIMIAR_FUZZY}",
        )

    docentes = buscar_entidades_por_campo("docente", "departamento", departamento)
    nomes = sorted(d["nome"] for d in docentes)

    if not nomes:
        resposta = f"Nenhum docente registrado no departamento '{departamento}'."
    else:
        resposta = f"O departamento '{departamento}' tem {len(nomes)} docentes: " + ", ".join(nomes)

    return ResultadoPipeline(
        pipeline="2-estruturado",
        pergunta=pergunta,
        resposta=resposta,
        fontes=["sqlite"],
        detalhe=f"departamento casado por fuzzy: '{departamento}' (score {score:.0f})",
    )


# --- Pipeline 3: agente com tool calling ----------------------------------

# Mapeia o nome da tool para a fonte de dados que ela representa, para poder
# dizer objetivamente de onde o agente tirou a resposta.
FONTE_POR_TOOL = {
    "buscar_docentes_por_departamento": "sqlite",
    "busca_vetorial_sigaa": "chromadb",
}


def responder_agente(componentes, pergunta: str) -> ResultadoPipeline:
    """
    O agente real (agent.processar_pergunta), com histórico novo a cada
    pergunta — sem histórico compartilhado, senão uma pergunta contaminaria a
    seguinte e a comparação deixaria de ser pergunta a pergunta.
    """
    texto, historico = processar_pergunta(
        chat_generator=componentes.chat_generator,
        embedder=componentes.embedder,
        retriever=componentes.retriever,
        chat_history=montar_historico_inicial(),
        pergunta_usuario=pergunta,
    )

    tools_usadas = [
        resultado.origin.tool_name
        for msg in historico
        for resultado in (msg.tool_call_results or [])
    ]
    fontes = sorted({FONTE_POR_TOOL.get(t, t) for t in tools_usadas})

    return ResultadoPipeline(
        pipeline="3-agente",
        pergunta=pergunta,
        resposta=texto,
        fontes=fontes,
        detalhe=(
            f"tools chamadas: {', '.join(tools_usadas)}"
            if tools_usadas
            else "nenhuma tool chamada — o LLM respondeu de cabeça"
        ),
    )


PIPELINES = {
    "1-vetorial": responder_vetorial,
    "2-estruturado": responder_estruturado,
    "3-agente": responder_agente,
}
