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

import os

from haystack.tools import Tool

from modulo1_etl.db_manager import buscar_entidades_por_campo, total_de_entidades
import config

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
    {
        "type": "function",
        "function": {
            "name": "buscar_docente_por_nome",
            "description": (
                "Utilize esta ferramenta quando o usuário perguntar sobre UM "
                "docente específico pelo nome — em que departamento ele está, "
                "se ele existe na base, qual o vínculo dele. Retorna dados "
                "exatos do cadastro, não texto de perfil."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "nome": {
                        "type": "string",
                        "description": (
                            "Nome, ou parte do nome, do docente procurado "
                            "(ex: Marcel William Rocha da Silva)"
                        ),
                    }
                },
                "required": ["nome"],
            },
        },
    },
]


def buscar_docentes_por_departamento(departamento: str) -> str:
    """Ferramenta determinística — consulta o SQLite (schema-less)."""
    print(f"🔧 [TOOL EXECUTADA] Consulta estruturada em SQLite pelo departamento: {departamento}")

    resultados = buscar_entidades_por_campo("docente", "departamento", departamento)

    if not resultados:
        # DOIS ZEROS DIFERENTES. Nenhum docente naquele departamento e uma
        # resposta legitima; base vazia e falha de infraestrutura. Ate 5 set
        # 2026 as duas saiam com o mesmo texto, e a segunda virava a resposta
        # errada mais convincente possivel — dita com seguranca, verificavel,
        # e completamente falsa. Tipicamente acontecia rodando fora do Docker,
        # onde o DB_PATH default abria um banco novo e vazio.
        if total_de_entidades("docente") == 0:
            return (
                "Acesso à Base Estruturada: FALHA — a base não contém docente "
                f"nenhum (DB_PATH={config.DB_PATH}). Isto não significa que o "
                "departamento esteja vazio: significa que o banco não foi "
                "carregado ou que o caminho está errado. Rode o ETL, ou "
                "verifique DB_PATH. Não responda como se não houvesse docentes."
            )
        return (
            f"Acesso à Base Estruturada: Não encontrei nenhum docente "
            f"registrado sob o departamento '{departamento}'."
        )

    # A busca é por substring, então um termo pode casar mais de um
    # departamento — "Física" casa tanto DEPARTAMENTO DE FÍSICA (14) quanto
    # DEPARTAMENTO DE EDUCAÇÃO FÍSICA E DESPORTOS (8). Somar os dois num
    # número só devolveria 22, um valor plausível e errado. Quando há
    # ambiguidade, ela é reportada em vez de escondida: o LLM tem contexto
    # para escolher, ou para perguntar ao usuário.
    por_departamento: dict[str, list[str]] = {}
    for r in resultados:
        por_departamento.setdefault(r["departamento"], []).append(r["nome"])

    if len(por_departamento) > 1:
        linhas = [
            f"- {depto}: {len(nomes)} docentes"
            for depto, nomes in sorted(por_departamento.items())
        ]
        return (
            f"Acesso à Base Estruturada: o termo '{departamento}' corresponde a "
            f"{len(por_departamento)} departamentos distintos. Não somei os "
            f"totais — informe ao usuário a distinção ou peça qual deles:\n"
            + "\n".join(linhas)
        )

    nome_exato, nomes = next(iter(por_departamento.items()))
    nomes = sorted(nomes)
    total = len(nomes)

    # TETO DE LISTAGEM, nunca recusa de listar.
    #
    # O corte era `total <= 10`, e acima disso a tool respondia "não os listarei
    # todos para poupar espaço" — o que torna "quais docentes pertencem ao
    # Departamento de Bioquímica?" (11 pessoas) impossível de responder. A
    # bateria de 5 set 2026 reprovou est-06 por isso, e a culpa não era do
    # agente: era a ferramenta se recusando a fazer o que foi pedido.
    #
    # Agora sempre lista, com teto e dizendo quantos ficaram de fora. O total
    # continua exato em qualquer caso, e o que foi omitido fica declarado —
    # omissão silenciosa é o que produz resposta incompleta com cara de
    # completa.
    TETO_LISTAGEM = 40
    lista = "\n- ".join(nomes[:TETO_LISTAGEM])
    if total > TETO_LISTAGEM:
        return (
            f"Acesso à Base Estruturada: O departamento '{nome_exato}' tem "
            f"{total} docentes. Os {TETO_LISTAGEM} primeiros em ordem alfabética "
            f"são:\n- {lista}\n(os outros {total - TETO_LISTAGEM} não foram "
            f"listados; o total acima é exato)"
        )
    return (
        f"Acesso à Base Estruturada: O departamento '{nome_exato}' tem "
        f"{total} docentes. São eles:\n- {lista}"
    )


# ACHADO 03b — limiar de distância, DESLIGADO por padrão.
#
# ATENÇÃO À DIREÇÃO DA COMPARAÇÃO. O `score` que o ChromaEmbeddingRetriever
# devolve é DISTÂNCIA, não similaridade: menor é mais parecido. Medido em
# 4 set 2026 com bge-m3 — "inteligência artificial" recuperou os melhores em
# ~1.02, e "culinária japonesa medieval", que não tem nada a ver com o corpus,
# ainda recuperou 5 documentos, o melhor deles em ~1.37. Escrever o filtro na
# direção intuitiva (score >= limiar) descartaria exatamente os relevantes e
# devolveria só o lixo, sem erro nenhum.
#
# Fica desligado porque calibrar sobre o corpus atual seria calibrar sobre dado
# corrompido (achados 08 e 09). O valor entra depois da recarga, medindo a
# distribuição real — ver a ordem de execução no CLAUDE.md.
_limiar_bruto = os.getenv("LIMIAR_DISTANCIA", "").strip()
LIMIAR_DISTANCIA = float(_limiar_bruto) if _limiar_bruto else None


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

    if LIMIAR_DISTANCIA is not None:
        antes = len(docs)
        docs = [d for d in docs if d.score is None or d.score <= LIMIAR_DISTANCIA]
        if antes != len(docs):
            print(f"🧠 [LIMIAR] {antes - len(docs)} de {antes} documentos acima de "
                  f"{LIMIAR_DISTANCIA} de distância foram descartados.")

    if not docs:
        return "Acesso à Base Vetorial: Nenhuma informação semântica relevante foi encontrada."

    # ACHADO 02: o texto ia sozinho, e os metadados eram jogados fora.
    # Com o perfil fatiado, um pedaço a partir do segundo não continha o nome
    # de ninguém, e o LLM recebia texto sobre alguém sem saber sobre quem —
    # ou omitia a atribuição, ou a inventava. Os perfis deixaram de ser
    # fatiados (parte3_chunking), mas mandar nome, departamento e fonte junto
    # continua sendo o certo: é o que permite ao agente citar corretamente e
    # é de graça, já vem no metadado do documento recuperado.
    blocos = []
    for d in docs:
        nome = d.meta.get("nome_docente") or "(nome ausente no metadado)"
        depto = d.meta.get("departamento") or "(departamento ausente)"
        cabecalho = f"[{nome} — {depto}]"
        fonte = d.meta.get("source_url")
        if fonte:
            cabecalho += f" fonte: {fonte}"
        blocos.append(cabecalho + "\n" + d.content)

    contexto = "\n---\n".join(blocos)
    return f"Acesso à Base Vetorial. Documentos recuperados:\n{contexto}"


def buscar_docente_por_nome(nome: str) -> str:
    """
    Ferramenta determinística — em que departamento está um docente.

    POR QUE EXISTE: a bateria de 5 set 2026 expôs que a pergunta "em qual
    departamento trabalha o professor X?" NÃO TINHA caminho estruturado.
    `buscar_docentes_por_departamento` recebe um departamento, não um nome, e
    a única saída do agente era procurar a pessoa na busca semântica — que
    acerta por recuperação, não por cadastro. Vínculo docente-departamento é
    dado exato e merece resposta exata.
    """
    print(f"🔧 [TOOL EXECUTADA] Consulta estruturada em SQLite pelo nome: {nome}")

    resultados = buscar_entidades_por_campo("docente", "nome", nome)

    if not resultados:
        # Os dois zeros, de novo: base vazia não é o mesmo que pessoa ausente.
        if total_de_entidades("docente") == 0:
            return (
                "Acesso à Base Estruturada: FALHA — a base não contém docente "
                f"nenhum (DB_PATH={config.DB_PATH}). Não responda como se a "
                "pessoa não existisse."
            )
        return (
            f"Acesso à Base Estruturada: Nenhum docente cadastrado com o nome "
            f"'{nome}'."
        )

    if len(resultados) > 1:
        # Teto na listagem: "Silva" casa com 130 docentes, e despejar todos no
        # contexto do LLM custa mais do que informa. O total continua exato.
        TETO = 15
        # Mesma disciplina do achado 06: mais de um casamento é ambiguidade a
        # relatar, não algo a resolver escolhendo o primeiro.
        linhas = "\n".join(
            f"- {r.get('nome')}: {r.get('departamento')}" for r in resultados[:TETO]
        )
        if len(resultados) > TETO:
            linhas += "\n" + f"... e mais {len(resultados) - TETO} docentes."
        return (
            f"Acesso à Base Estruturada: o nome '{nome}' casa com "
            f"{len(resultados)} docentes. Não escolhi por você:\n{linhas}"
        )

    unico = resultados[0]
    return (
        f"Acesso à Base Estruturada: {unico.get('nome')} pertence ao "
        f"{unico.get('departamento')}."
    )


def criar_dispatcher(embedder, retriever) -> dict:
    """
    Monta o dicionário nome_da_tool -> função executável.

    O agent.py não precisa conhecer a assinatura de cada tool — só chama
    dispatcher[nome](**argumentos_do_llm). Adicionar uma tool nova não exige
    tocar em agent.py, só registrar aqui.
    """
    return {
        "buscar_docente_por_nome": lambda nome="": buscar_docente_por_nome(nome),
        "buscar_docentes_por_departamento": lambda departamento="": buscar_docentes_por_departamento(
            departamento
        ),
        "busca_vetorial_sigaa": lambda pergunta_semantica="": busca_vetorial_sigaa(
            pergunta_semantica, embedder, retriever
        ),
    }


def criar_tools(embedder, retriever) -> list[Tool]:
    """
    Converte TOOLS_SCHEMA em objetos Tool do Haystack, que é o formato que o
    OllamaChatGenerator aceita no parâmetro `tools=`.

    POR QUE ISSO EXISTE: antes, agent.py passava TOOLS_SCHEMA cru dentro de
    generation_kwargs. A integração joga generation_kwargs inteiro no dict
    `options` da chamada ao Ollama, então os schemas viravam uma "option"
    chamada 'tools' — que o Ollama ignora — e o parâmetro `tools=` de verdade
    ia como None. Resultado: nenhuma tool era anunciada ao modelo, ele nunca
    emitia tool call, e o agente respondia sempre direto, sem tocar no SQLite
    nem no ChromaDB. As tools nunca chegaram a executar.

    O `function` de cada Tool aponta pro dispatcher para não duplicar o
    mapeamento nome -> callable. O loop de agent.py continua despachando por
    conta própria (ele precisa do resultado para montar o ChatMessage.from_tool),
    então na prática esse callable não é invocado pelo Haystack — mas deixá-lo
    correto evita uma armadilha se algum dia um ToolInvoker entrar no caminho.
    """
    dispatcher = criar_dispatcher(embedder, retriever)

    return [
        Tool(
            name=spec["function"]["name"],
            description=spec["function"]["description"],
            parameters=spec["function"]["parameters"],
            function=dispatcher[spec["function"]["name"]],
        )
        for spec in TOOLS_SCHEMA
    ]
