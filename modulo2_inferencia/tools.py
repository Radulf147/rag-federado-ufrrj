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
from interfaces.tipos import TIPOS

# A tool semantica NAO sai do registro: ela nao filtra dado estruturado de um
# tipo, ela varre o texto livre de todos. Fica literal, e a descricao carrega a
# correcao de Set/2026 -- "perguntas genericas interpretativas" sinalizava que
# pergunta sobre UMA pessoa nomeada nao era para ca, quando e exatamente para ca
# se o que se pede e conteudo de perfil.
SCHEMA_SEMANTICO = {
    "type": "function",
    "function": {
        "name": "busca_vetorial_sigaa",
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
                    "description": "A pergunta otimizada para buscar no banco de dados vetorial.",
                }
            },
            "required": ["pergunta_semantica"],
        },
    },
}


def _schema_da_busca(busca) -> dict:
    """Uma Busca do registro vira o schema que o LLM le."""
    return {
        "type": "function",
        "function": {
            "name": busca.nome_tool,
            "description": busca.descricao,
            "parameters": {
                "type": "object",
                "properties": {
                    busca.parametro: {
                        "type": "string",
                        "description": busca.descricao_parametro,
                    }
                },
                "required": [busca.parametro],
            },
        },
    }


# ⚠️ A ORDEM E FIXADA DE PROPOSITO, e nao e detalhe de arrumacao.
#
# Esta lista e literalmente o prompt que o LLM le para escolher a ferramenta, e
# a acuracia de roteamento de 97,8% da fase 3 foi medida com ESTA ordem. Deixar
# a ordem sair de iteracao de dicionario faria a reestruturacao mudar, de
# graca, uma variavel que o experimento controlava -- e o numero novo pareceria
# igualmente valido.
#
# Tool que o registro conhece e esta lista nao menciona entra no fim, em vez de
# sumir em silencio.
ORDEM_DAS_TOOLS = (
    "buscar_docentes_por_departamento",
    "busca_vetorial_sigaa",
    "buscar_docente_por_nome",
)


def _montar_schema() -> list[dict]:
    por_nome = {"busca_vetorial_sigaa": SCHEMA_SEMANTICO}
    for tipo in TIPOS.values():
        for busca in tipo.buscas:
            por_nome[busca.nome_tool] = _schema_da_busca(busca)

    ordenadas = [por_nome.pop(n) for n in ORDEM_DAS_TOOLS if n in por_nome]
    # o que sobrou e tipo novo, ainda sem lugar declarado na ordem
    return ordenadas + list(por_nome.values())


TOOLS_SCHEMA = _montar_schema()


def _entidades(tipo, busca, valor: str) -> list[dict]:
    """Consulta comum aos dois formatos, com o rastro que a execucao imprime."""
    print(
        f"🔧 [TOOL EXECUTADA] Consulta estruturada em SQLite pelo "
        f"{busca.campo}: {valor}"
    )
    return buscar_entidades_por_campo(tipo.nome, busca.campo, valor)


def buscar_agrupado(tipo, busca, valor: str) -> str:
    """
    Conta e lista as entidades de um grupo, relatando ambiguidade.

    Generalizacao de `buscar_docentes_por_departamento`. Os substantivos vem do
    registro (`interfaces/tipos.py`), nao de heuristica sobre o nome do campo:
    o texto que o LLM le e artefato medido, e uma regra esperta acertaria hoje
    e erraria calada no primeiro tipo novo.
    """
    resultados = _entidades(tipo, busca, valor)

    if not resultados:
        # DOIS ZEROS DIFERENTES. Nenhuma entidade naquele grupo e uma resposta
        # legitima; base vazia e falha de infraestrutura. Ate 5 set 2026 as duas
        # saiam com o mesmo texto, e a segunda virava a resposta errada mais
        # convincente possivel — dita com seguranca, verificavel, e
        # completamente falsa.
        if total_de_entidades(tipo.nome) == 0:
            return (
                f"Acesso à Base Estruturada: FALHA — a base não contém {tipo.singular} "
                f"nenhum (DB_PATH={config.DB_PATH}). Isto não significa que "
                f"{busca.artigo} {busca.singular_do_campo} esteja vazio: significa "
                "que o banco não foi carregado ou que o caminho está errado. Rode "
                f"o ETL, ou verifique DB_PATH. Não responda como se não houvesse "
                f"{tipo.plural}."
            )
        return (
            f"Acesso à Base Estruturada: Não encontrei nenhum {tipo.singular} "
            f"registrado sob {busca.artigo} {busca.singular_do_campo} '{valor}'."
        )

    # A busca e por substring, entao um termo pode casar mais de um grupo —
    # "Fisica" casa DEPARTAMENTO DE FISICA (26) e DEPARTAMENTO DE EDUCACAO
    # FISICA E DESPORTOS (18). Somar os dois num numero so devolveria 44, um
    # valor plausivel e errado. Ambiguidade e relatada, nao escondida.
    por_grupo: dict[str, list[str]] = {}
    for r in resultados:
        por_grupo.setdefault(r[busca.campo], []).append(r[tipo.campo_rotulo])

    if len(por_grupo) > 1:
        linhas = [
            f"- {grupo}: {len(itens)} {tipo.plural}"
            for grupo, itens in sorted(por_grupo.items())
        ]
        return (
            f"Acesso à Base Estruturada: o termo '{valor}' corresponde a "
            f"{len(por_grupo)} {busca.plural_do_campo} distintos. Não somei os "
            f"totais — informe ao usuário a distinção ou peça qual deles:\n"
            + "\n".join(linhas)
        )

    exato, itens = next(iter(por_grupo.items()))
    itens = sorted(itens)
    total = len(itens)

    # TETO DE LISTAGEM, nunca recusa de listar. O corte era `total <= 10`, e
    # acima disso a tool respondia "nao os listarei todos para poupar espaco" —
    # o que torna "quais docentes pertencem ao Departamento de Bioquimica?" (11
    # pessoas) impossivel de responder. A bateria de 5 set 2026 reprovou est-06
    # por isso, e a culpa nao era do agente: era a ferramenta se recusando a
    # fazer o que foi pedido. Agora sempre lista, e o que foi omitido fica
    # declarado — omissao silenciosa produz resposta incompleta com cara de
    # completa.
    TETO_LISTAGEM = 40
    lista = "\n- ".join(itens[:TETO_LISTAGEM])
    if total > TETO_LISTAGEM:
        return (
            f"Acesso à Base Estruturada: {busca.artigo.upper()} "
            f"{busca.singular_do_campo} '{exato}' tem "
            f"{total} {tipo.plural}. Os {TETO_LISTAGEM} primeiros em ordem alfabética "
            f"são:\n- {lista}\n(os outros {total - TETO_LISTAGEM} não foram "
            f"listados; o total acima é exato)"
        )
    return (
        f"Acesso à Base Estruturada: {busca.artigo.upper()} "
        f"{busca.singular_do_campo} '{exato}' tem "
        f"{total} {tipo.plural}. São eles:\n- {lista}"
    )


def buscar_um_ou_ambiguo(tipo, busca, valor: str) -> str:
    """
    Espera uma entidade. Se vier mais de uma, RELATA em vez de escolher.

    Generalizacao de `buscar_docente_por_nome`. Mais de um casamento e
    ambiguidade a reportar, nao algo a resolver escolhendo o primeiro — mesma
    disciplina do achado 06.
    """
    resultados = _entidades(tipo, busca, valor)

    if not resultados:
        # Os dois zeros, de novo: base vazia nao e o mesmo que entidade ausente.
        if total_de_entidades(tipo.nome) == 0:
            return (
                f"Acesso à Base Estruturada: FALHA — a base não contém {tipo.singular} "
                f"nenhum (DB_PATH={config.DB_PATH}). Não responda como se "
                f"{tipo.referente} não existisse."
            )
        return (
            f"Acesso à Base Estruturada: Nenhum {tipo.singular} cadastrado com "
            f"{busca.artigo} {busca.singular_do_campo} '{valor}'."
        )

    if len(resultados) > 1:
        # Teto na listagem: "Silva" casa com 130 docentes, e despejar todos no
        # contexto do LLM custa mais do que informa. O total continua exato.
        TETO = 15
        linhas = "\n".join(
            f"- {r.get(tipo.campo_rotulo)}: {r.get(tipo.campo_vinculo)}"
            for r in resultados[:TETO]
        )
        if len(resultados) > TETO:
            linhas += "\n" + f"... e mais {len(resultados) - TETO} {tipo.plural}."
        return (
            f"Acesso à Base Estruturada: {busca.artigo} {busca.singular_do_campo} "
            f"'{valor}' casa com "
            f"{len(resultados)} {tipo.plural}. Não escolhi por você:\n{linhas}"
        )

    unico = resultados[0]
    return (
        f"Acesso à Base Estruturada: {unico.get(tipo.campo_rotulo)} pertence ao "
        f"{unico.get(tipo.campo_vinculo)}."
    )


FORMATOS = {
    "agrupado": buscar_agrupado,
    "um_ou_ambiguo": buscar_um_ou_ambiguo,
}


# Os dois nomes abaixo existem para nao quebrar quem os importa direto
# (testes de regressao, instantaneo do criterio de aceite do D0). O corpo saiu
# daqui e virou molde no registro.
def buscar_docentes_por_departamento(departamento: str) -> str:
    tipo = TIPOS["docente"]
    return buscar_agrupado(tipo, tipo.buscas[0], departamento)


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
    # D1: o cabeçalho sai do `rotulo`, não de `nome_docente`. Um documento de
    # curso ou de componente curricular sairia daqui como "(nome ausente no
    # metadado)" — o achado 02 voltando por outra porta, texto chegando ao LLM
    # sem dizer de quem é.
    #
    # `rotulo()` recua para `nome_docente` quando o campo não existe, então a
    # saída para docente é IDÊNTICA, byte a byte, à de antes desta mudança. É o
    # critério de aceite do D0, e há teste que o fixa.
    from interfaces.identidade import rotulo as _rotulo

    blocos = []
    for d in docs:
        nome = _rotulo(d.meta)
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
    Vinculo de UMA pessoa. POR QUE EXISTE: a bateria de 5 set 2026 expos que a
    pergunta "em qual departamento trabalha o professor X?" NAO TINHA caminho
    estruturado, e a unica saida do agente era a busca semantica — que acerta
    por recuperacao, nao por cadastro.

    ⚠️ ITEM 9 DO BACKLOG: o casamento e por SUBSTRING CONTIGUA, entao
    "Leandro Alvim" NAO acha "LEANDRO GUIMARAES MARQUES ALVIM". O defeito esta
    em `db_manager.buscar_entidades_por_campo` e nao foi corrigido aqui.
    """
    tipo = TIPOS["docente"]
    return buscar_um_ou_ambiguo(tipo, tipo.buscas[1], nome)


def criar_dispatcher(embedder, retriever) -> dict:
    """
    Monta o dicionario nome_da_tool -> funcao executavel, A PARTIR DO REGISTRO.

    O agent.py nao precisa conhecer a assinatura de cada tool — so chama
    dispatcher[nome](**argumentos_do_llm). Tipo novo no registro aparece aqui
    sozinho: era esta a promessa do D0.
    """
    despacho = {
        "busca_vetorial_sigaa": lambda pergunta_semantica="": busca_vetorial_sigaa(
            pergunta_semantica, embedder, retriever
        ),
    }
    for tipo in TIPOS.values():
        for busca in tipo.buscas:
            formatador = FORMATOS[busca.formato]

            def executar(_t=tipo, _b=busca, _f=formatador, **argumentos):
                return _f(_t, _b, argumentos.get(_b.parametro, ""))

            despacho[busca.nome_tool] = executar
    return despacho


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
