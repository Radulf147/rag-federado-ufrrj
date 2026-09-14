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

# --- VARIANTE DA CONSULTA SEMÂNTICA ---------------------------------------
#
# Qual texto é de fato embutido pelo recuperador. Experimento pré-registrado em
# `docs/pre_registro_consulta_semantica.md`.
#
# POR QUE EXISTE. A bateria de 14 set mostrou que o agente NÃO manda a pergunta
# do usuário ao recuperador: manda o termo nu — "didática" no lugar de "Algum
# professor atua com didática?" — em 21 de 21 execuções das 7 perguntas
# semânticas. E o termo nu recupera pior neste corpus, porque um perfil esparso
# (nome + departamento + telefone) é curto e o nome do departamento domina o
# vetor dele. Medido: perfis esparsos no top-10 passam de 21% para 49%, e em
# sem-03, sem-04 e sem-09 o agente recupera ZERO docentes do gabarito — ali era
# aritmeticamente impossível responder certo.
#
# A causa raiz é este arquivo: o parâmetro se chama `pergunta_semantica` e a
# descrição manda "otimizar". O modelo obedece.
#
#   v0  o argumento do LLM — O COMPORTAMENTO DE HOJE, e o padrão
#   v1  a pergunta original do usuário, ignorando o argumento do LLM
#   v2  as duas, documentos unidos sem repetir, ordenados por distância
#   v3  o argumento do LLM, mas com o schema pedindo a pergunta na íntegra
#
# ⚠️ O padrão é v0 DE PROPÓSITO: enquanto o experimento não decidir, nada muda
# em produção sem alguém escrever a variável.
VARIANTE_CONSULTA = os.getenv("VARIANTE_CONSULTA", "v0").strip().lower() or "v0"

_VARIANTES_VALIDAS = {"v0", "v1", "v2", "v3"}
if VARIANTE_CONSULTA not in _VARIANTES_VALIDAS:
    # Falhar alto. Um nome errado de variante cairia silenciosamente no v0 e a
    # bateria reportaria "v9 medida" tendo medido o controle — exatamente o
    # número plausível e errado que este projeto trata como o inimigo.
    raise ValueError(
        f"VARIANTE_CONSULTA={VARIANTE_CONSULTA!r} é inválida. "
        f"Use uma de: {', '.join(sorted(_VARIANTES_VALIDAS))}."
    )

# A descrição do parâmetro é a única coisa que a v3 muda. As outras três usam a
# redação original — inclusive a v1 e a v2, porque nelas o que o LLM escreve já
# não decide sozinho o que é embutido.
_DESCRICAO_ARGUMENTO = {
    "v3": (
        "A pergunta do usuário na ÍNTEGRA, exatamente como ele escreveu, sem "
        "resumir, sem extrair palavra-chave e sem reformular."
    ),
}.get(VARIANTE_CONSULTA, "A pergunta otimizada para buscar no banco de dados vetorial.")

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
            # "perguntas genéricas interpretativas" era a outra metade do
            # defeito: sinalizava que perguntas sobre UMA pessoa nomeada não
            # eram para cá, quando são exatamente para cá se o que se pede é
            # conteúdo de perfil.
            "description": (
                "Todo o TEXTO do perfil dos docentes: formação acadêmica, "
                "áreas de interesse, atuação, descrição pessoal. Serve tanto "
                "para pergunta ampla ('quem pesquisa Inteligência "
                "Artificial?') quanto para uma pessoa nomeada ('qual a "
                "formação de fulano?') — o que decide é o dado pedido ser "
                "texto de perfil, não a pergunta citar um nome."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pergunta_semantica": {
                        "type": "string",
                        "description": _DESCRICAO_ARGUMENTO,
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
            # A redação anterior abria com "quando o usuário perguntar sobre UM
            # docente específico pelo nome", e isso casava com QUALQUER pergunta
            # que citasse uma pessoa. Na bateria de 5 set derrubou "qual é a
            # formação acadêmica de Filipe Braida?" e "quais são as áreas de
            # interesse de Marcel?" — 6 execuções, todos os erros de roteamento
            # da rodada. Agora abre pelo que DEVOLVE, e nomeia o destino certo.
            "description": (
                "Vínculo de UMA pessoa: dado o nome, diz a que departamento "
                "ela pertence, ou que não está cadastrada. Isso é tudo o que "
                "devolve. NÃO tem formação acadêmica, áreas de interesse, "
                "atuação, contato nem qualquer outro texto do perfil — para "
                "esses use busca_vetorial_sigaa, inclusive quando a pergunta "
                "nomear a pessoa."
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


def _textos_a_embutir(pergunta: str, pergunta_original: str | None) -> list[str]:
    """
    Qual texto vai ao embedder, segundo `VARIANTE_CONSULTA`.

    `pergunta` é o argumento que o LLM escolheu; `pergunta_original` é o que o
    usuário de fato perguntou. Quando a original não chega — CLI antigo, teste,
    qualquer chamador que não a passe —, v1 e v2 caem de volta no argumento do
    LLM. Cair de volta é melhor que estourar, mas não é silencioso: sem a
    original, v1 É v0, e o registro em `registro_busca` mostra isso.
    """
    original = (pergunta_original or "").strip()

    if VARIANTE_CONSULTA == "v1" and original:
        return [original]
    if VARIANTE_CONSULTA == "v2" and original and original != pergunta:
        return [original, pergunta]
    # v0, v3, e os casos sem pergunta original: o argumento do LLM.
    return [pergunta]


def busca_vetorial_sigaa(
    pergunta: str,
    embedder,
    retriever,
    pergunta_original: str | None = None,
    registro_busca: list | None = None,
) -> str:
    """
    Ferramenta semântica — consulta o ChromaDB (textos livres).

    Recebe embedder/retriever como parâmetros em vez de globais do módulo
    (como era em teste_llm.py) para poder ser testada com dublês/mocks sem
    precisar inicializar o Ollama ou o ChromaDB de verdade.

    `pergunta_original` e `registro_busca` são opcionais e o padrão preserva o
    comportamento anterior: sem eles, e com VARIANTE_CONSULTA=v0, esta função
    faz exatamente o que fazia antes de 14 set.
    """
    textos = _textos_a_embutir(pergunta, pergunta_original)

    for t in textos:
        print(f"🧠 [TOOL EXECUTADA] Busca semântica em ChromaDB por: {t}")

    if len(textos) == 1:
        query_vec = embedder.run(text=textos[0])["embedding"]
        docs = retriever.run(query_embedding=query_vec)["documents"]
    else:
        # v2 — união. Cada consulta traz seus TOP_K; junta sem repetir e ordena
        # por DISTÂNCIA (menor é mais parecido), cortando de novo em TOP_K.
        #
        # A chave de deduplicação é `d.id`, e não o nome: dois chunks da mesma
        # pessoa são documentos distintos, e o nome funde pessoas homônimas.
        #
        # `score` None vai para o fim em vez de quebrar a ordenação — um
        # documento sem distância não tem como competir por proximidade.
        vistos, unidos = set(), []
        for t in textos:
            vec = embedder.run(text=t)["embedding"]
            for d in retriever.run(query_embedding=vec)["documents"]:
                if d.id not in vistos:
                    vistos.add(d.id)
                    unidos.append(d)
        unidos.sort(key=lambda d: float("inf") if d.score is None else d.score)
        docs = unidos[: config.TOP_K]

    # O QUE FOI DE FATO EMBUTIDO. Sem isto a instrumentação de adfcc07 passaria
    # a mentir: ela lê o argumento do ToolCall, que na v1 e na v2 não é o texto
    # que chegou ao embedder. Um registro que descreve a intenção e não o ato é
    # pior que nenhum.
    if registro_busca is not None:
        registro_busca.append(
            {
                "variante": VARIANTE_CONSULTA,
                "argumento_do_llm": pergunta,
                "embutido": textos,
                "documentos": len(docs),
            }
        )

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


def criar_dispatcher(
    embedder, retriever, pergunta_original: str | None = None, registro_busca: list | None = None
) -> dict:
    """
    Monta o dicionário nome_da_tool -> função executável.

    O agent.py não precisa conhecer a assinatura de cada tool — só chama
    dispatcher[nome](**argumentos_do_llm). Adicionar uma tool nova não exige
    tocar em agent.py, só registrar aqui.

    `pergunta_original` e `registro_busca` entram pelo fecho (closure) e não
    pelo schema — de propósito. O LLM não pode escolhê-los nem enxergá-los:
    são contexto do sistema, não argumento de ferramenta. Ambos são opcionais,
    e sem eles o dispatcher é o de antes.
    """
    return {
        "buscar_docente_por_nome": lambda nome="": buscar_docente_por_nome(nome),
        "buscar_docentes_por_departamento": lambda departamento="": buscar_docentes_por_departamento(
            departamento
        ),
        "busca_vetorial_sigaa": lambda pergunta_semantica="": busca_vetorial_sigaa(
            pergunta_semantica, embedder, retriever, pergunta_original, registro_busca
        ),
    }


def criar_tools(
    embedder, retriever, pergunta_original: str | None = None, registro_busca: list | None = None
) -> list[Tool]:
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
    dispatcher = criar_dispatcher(embedder, retriever, pergunta_original, registro_busca)

    return [
        Tool(
            name=spec["function"]["name"],
            description=spec["function"]["description"],
            parameters=spec["function"]["parameters"],
            function=dispatcher[spec["function"]["name"]],
        )
        for spec in TOOLS_SCHEMA
    ]
