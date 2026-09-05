"""
Calibração do LIMIAR_DISTANCIA da busca semântica — segunda metade do achado 03.

O QUE ESTE SCRIPT DECIDE: a partir de que distância um documento recuperado do
ChromaDB deixa de ser resposta e passa a ser ruído.

POR QUE PRECISA DE UM SCRIPT, E NÃO DE UM CHUTE: o modo de falha que o limiar
existe para impedir é silencioso. Medido em 4 set 2026, "culinária japonesa
medieval" — que não tem relação nenhuma com um corpus de docentes da UFRRJ —
recuperava perfis e os entregava ao LLM como contexto. Sem limiar, a busca
semântica NUNCA responde "não achei": devolve sempre os TOP_K menos distantes,
por mais distantes que estejam.

ATENÇÃO À DIREÇÃO: o `score` do ChromaEmbeddingRetriever é DISTÂNCIA, não
similaridade. Menor é mais parecido. O filtro é `score <= limiar`.

A MEDIDA TEM DE SER SOBRE O TOP_K, E NÃO SOBRE O CORPUS
-------------------------------------------------------
A primeira versão deste script varria os 1302 documentos e calculava precisão
sobre tudo que ficasse abaixo do limiar. O número saía (precisão 0.108) e não
significava nada: em produção o retriever devolve TOP_K=10, e o limiar corta
DENTRO desses 10. Medir sobre o corpus inteiro descreve um sistema que não
existe — erro plausível e errado, que é o que este projeto mais teme.

O que se mede aqui é o que o agente de fato vê:

    por consulta de dentro do domínio ... a distância do 1º e do 10º resultado,
                                          e quantos dos 10 são relevantes
    por consulta de fora do domínio ..... a distância do 1º resultado

O limiar precisa ficar ABAIXO do melhor resultado de toda consulta fora do
domínio (senão ela recebe contexto) e ACIMA do 1º resultado de toda consulta de
dentro (senão ela emudece). Se essas duas condições não puderem valer ao mesmo
tempo, o limiar sozinho não resolve o achado 03 — e o script diz isso em vez de
inventar um número.

VERDADE-BASE
------------
Um documento é "relevante" para o termo T quando o texto dele contém T
literalmente. Os termos saem dos campos "Áreas de interesse" mais frequentes do
corpus — não são escolhidos a dedo, para não enviesar o resultado.

O limite, dito na cara: isso é casamento LEXICAL. Quem escreveu "aprendizado de
máquina" é relevante para "inteligência artificial" e conta aqui como não
relevante. Logo "relevantes no top-10" é um PISO. A decisão do limiar, porém,
se apoia nas colunas de distância, que não sofrem desse viés.

    docker compose run --rm etl python -m modulo2_inferencia.calibrar_limiar
"""

import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import config

SAIDA = Path("docs/calibracao_limiar.md")

# Consultas claramente fora do domínio de um corpus de docentes da UFRRJ.
# Respondem "quão perto o corpus chega quando NÃO existe resposta?".
CONSULTAS_FORA = [
    "culinária japonesa medieval",
    "resultado do campeonato escocês de futebol",
    "como trocar o óleo de uma motocicleta",
    "preço do bitcoin hoje",
    "letra da música que toca no rádio",
    "receita de pão de queijo mineiro",
]

MIN_DOCENTES_POR_TERMO = 5
MAX_TERMOS = 12

# TOP_K real da aplicação: é dentro desta janela que o limiar atua.
TOP_K = config.TOP_K

LIMIARES = [round(0.90 + 0.02 * i, 2) for i in range(36)]


def normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.lower().split())


def conectar():
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    if config.CHROMA_REMOTE:
        return ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            embedding_function="default",
        )
    return ChromaDocumentStore(
        collection_name=config.CHROMA_COLECAO,
        persist_path=config.CHROMA_PERSIST_DIR,
        embedding_function="default",
    )


def termos_frequentes(documentos) -> list[tuple[str, int]]:
    """Termos mais comuns dos campos 'Áreas de interesse', derivados do corpus."""
    contagem: Counter = Counter()
    for doc in documentos:
        casamento = re.search(
            r"Áreas de interesse:\s*(.+?)(?=\s+(?:Currículo Lattes|Endereço|Sala|Telefone|E-mail):|$)",
            doc.content or "",
            re.DOTALL,
        )
        if not casamento:
            continue
        for pedaco in re.split(r"[;,\n]| e ", casamento.group(1)):
            termo = normalizar(pedaco)
            if 8 <= len(termo) <= 45 and len(termo.split()) <= 4:
                contagem[termo] += 1
    return [
        (t, n) for t, n in contagem.most_common(MAX_TERMOS * 3) if n >= MIN_DOCENTES_POR_TERMO
    ][:MAX_TERMOS]


def buscar(consulta, embedder, retriever, k):
    vetor = embedder.run(text=consulta)["embedding"]
    docs = retriever.run(query_embedding=vetor, top_k=k)["documents"]
    return [(d, d.score) for d in docs if d.score is not None]


def main() -> None:
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack_integrations.components.retrievers.chroma import ChromaEmbeddingRetriever

    store = conectar()
    documentos = store.filter_documents()
    print(f"[CALIBRAÇÃO] {len(documentos)} documentos no store. TOP_K={TOP_K}.")

    termos = termos_frequentes(documentos)
    if not termos:
        raise SystemExit("[CALIBRAÇÃO] nenhum termo frequente o bastante — corpus vazio?")

    embedder = SentenceTransformersTextEmbedder(model=config.MODELO_EMBEDDING)
    embedder.warm_up()
    retriever = ChromaEmbeddingRetriever(document_store=store)

    dentro = {}
    for termo, n_corpus in termos:
        topo = buscar(termo, embedder, retriever, TOP_K)
        alvo = normalizar(termo)
        marcados = [(normalizar(d.content).find(alvo) >= 0, s) for d, s in topo]
        dentro[termo] = {
            "no_corpus": n_corpus,
            "topo": marcados,
            "d1": marcados[0][1],
            "dk": marcados[-1][1],
            "relevantes": sum(1 for rel, _ in marcados if rel),
        }
        print(f"[CALIBRAÇÃO] {termo:28} 1º={dentro[termo]['d1']:.3f} "
              f"{TOP_K}º={dentro[termo]['dk']:.3f} "
              f"relevantes no topo={dentro[termo]['relevantes']}/{TOP_K}")

    fora = {}
    for consulta in CONSULTAS_FORA:
        topo = buscar(consulta, embedder, retriever, TOP_K)
        fora[consulta] = topo[0][1]
        print(f"[CALIBRAÇÃO] FORA {consulta:44} 1º={fora[consulta]:.3f}")

    escrever_relatorio(dentro, fora, len(documentos))


def escrever_relatorio(dentro, fora, total) -> None:
    d1_max = max(v["d1"] for v in dentro.values())
    fora_min = min(fora.values())
    separavel = d1_max < fora_min

    curva = []
    for limiar in LIMIARES:
        vivas = sum(1 for v in dentro.values() if v["d1"] <= limiar)
        mantidos = sum(1 for v in dentro.values() for rel, s in v["topo"] if rel and s <= limiar)
        perdidos = sum(1 for v in dentro.values() for rel, s in v["topo"] if rel and s > limiar)
        caladas = sum(1 for d in fora.values() if d > limiar)
        curva.append((limiar, vivas, mantidos, perdidos, caladas))

    # REGRA DE ESCOLHA — assimétrica, e a assimetria é deliberada.
    #
    # A primeira versão exigia calar TODAS as consultas de fora E manter TODAS as
    # de dentro. Nenhum limiar cumpria as duas, e o script concluía "nenhum
    # serve" — tecnicamente verdade e praticamente inútil, porque escondia que
    # existe um corte que remove boa parte do ruído sem custo nenhum.
    #
    # O critério certo parte de qual erro é pior. Emudecer uma consulta legítima
    # produz "não encontrei" para uma pergunta que TEM resposta: é resposta
    # errada com cara de cautela, e o usuário não tem como perceber. Deixar
    # passar ruído entrega documentos irrelevantes a um agente que hoje recebe
    # nome e departamento de cada um (achado 02) e está autorizado a dizer que
    # não sabe (achado 07) — ou seja, à camada que consegue julgar pertinência
    # semanticamente, coisa que aritmética de distância não faz.
    #
    # Logo: nunca sacrificar consulta legítima, e pegar toda redução de ruído
    # que sair de graça.
    sem_custo = [c for c in curva if c[1] == len(dentro) and c[3] == 0]
    recomendado = max(sem_custo, key=lambda c: (c[4], -c[0])) if sem_custo else None

    # Registrado só para o relatório mostrar o que a alternativa agressiva custa.
    agressivos = [c for c in curva if c[4] == len(fora)]
    agressivo = max(agressivos, key=lambda c: c[2]) if agressivos else None

    partes = [
        "# Calibração do limiar de distância — achado 03",
        "",
        f"Gerado por `modulo2_inferencia/calibrar_limiar.py` em "
        f"{datetime.now(timezone.utc).astimezone():%Y-%m-%d %H:%M}.",
        "",
        f"- Corpus: **{total}** documentos · embedding `{config.MODELO_EMBEDDING}`",
        f"- **TOP_K = {TOP_K}** — a medida é feita sobre a janela que o agente de fato vê",
        f"- Termos de dentro do domínio: **{len(dentro)}**, derivados dos campos "
        f"*Áreas de interesse* mais frequentes (mín. {MIN_DOCENTES_POR_TERMO} docentes)",
        f"- Consultas de fora do domínio: **{len(fora)}**",
        "",
        "> O `score` é **distância**: menor é mais parecido. O filtro é `score <= limiar`.",
        "",
        "## A pergunta que decide tudo",
        "",
        "O limiar precisa ficar **abaixo** do melhor resultado de toda consulta fora "
        "do domínio — senão ela recebe contexto que não existe — e **acima** do "
        "primeiro resultado de toda consulta de dentro — senão ela emudece.",
        "",
        f"- Pior 1º resultado entre as consultas de dentro: **{d1_max:.3f}**",
        f"- Melhor 1º resultado entre as consultas de fora: **{fora_min:.3f}**",
        "",
        (f"**As duas faixas se separam** ({d1_max:.3f} < {fora_min:.3f}): existe limiar que "
         "atende as duas condições."
         if separavel else
         f"**As duas faixas se sobrepõem** ({d1_max:.3f} ≥ {fora_min:.3f}): não existe limiar "
         "que cale todo o ruído sem emudecer alguma consulta legítima."),
        "",
        "## Dentro do domínio",
        "",
        f"| Termo | Docentes no corpus | 1º | {TOP_K}º | Relevantes no topo |",
        "|---|---|---|---|---|",
    ]
    for termo, v in sorted(dentro.items(), key=lambda kv: kv[1]["d1"]):
        partes.append(
            f"| {termo} | {v['no_corpus']} | {v['d1']:.3f} | {v['dk']:.3f} | "
            f"{v['relevantes']} de {TOP_K} |"
        )

    partes += [
        "",
        "## Fora do domínio",
        "",
        "| Consulta | 1º resultado |",
        "|---|---|",
    ]
    for consulta, d in sorted(fora.items(), key=lambda kv: kv[1]):
        partes.append(f"| {consulta} | {d:.3f} |")

    partes += [
        "",
        "## Curva de decisão",
        "",
        f"| Limiar | Consultas de dentro com resposta | Relevantes mantidos | Relevantes perdidos | Consultas de fora caladas |",
        "|---|---|---|---|---|",
    ]
    for limiar, vivas, mantidos, perdidos, caladas in curva:
        marca = " ← **recomendado**" if recomendado and limiar == recomendado[0] else ""
        partes.append(
            f"| {limiar:.2f} | {vivas} de {len(dentro)} | {mantidos} | {perdidos} | "
            f"{caladas} de {len(fora)}{marca} |"
        )

    partes += ["", "## Recomendação", ""]
    if recomendado:
        limiar, vivas, mantidos, perdidos, caladas = recomendado
        partes += [
            f"**`LIMIAR_DISTANCIA={limiar:.2f}`** — mantém resposta em **todas** as "
            f"{vivas} consultas de dentro do domínio, sem perder **nenhum** dos "
            f"{mantidos} documentos relevantes do topo, e ainda cala "
            f"**{caladas} das {len(fora)}** consultas de puro ruído.",
            "",
            "É o maior ganho que sai de graça. A escolha é assimétrica de propósito: "
            "emudecer uma consulta legítima produz \"não encontrei\" para uma pergunta "
            "que **tem** resposta — erro invisível, com cara de cautela. Deixar passar "
            "ruído entrega documentos irrelevantes a um agente que recebe nome e "
            "departamento de cada um (achado 02) e está autorizado a dizer que não sabe "
            "(achado 07). Distância absoluta não distingue *pergunta sem resposta* de "
            "*pergunta legítima sobre tema pouco representado*; o modelo, lendo o que "
            "veio, distingue.",
        ]
        if agressivo and agressivo[0] != limiar:
            partes += [
                "",
                f"**A alternativa agressiva não compensa.** Com "
                f"`LIMIAR_DISTANCIA={agressivo[0]:.2f}` as {len(fora)} consultas de ruído "
                f"ficariam caladas, mas ao preço de emudecer "
                f"{len(dentro) - agressivo[1]} consulta(s) legítima(s) e descartar "
                f"{agressivo[3]} documentos relevantes.",
            ]
    else:
        piores = sorted(dentro.items(), key=lambda kv: -kv[1]["d1"])[:3]
        partes += [
            "**Nenhum limiar serve, e forçar um seria pior que não ter.**",
            "",
            "Não existe corte que cale todas as consultas fora do domínio sem "
            "emudecer alguma consulta legítima. As consultas de dentro cujo melhor "
            "resultado fica mais longe:",
            "",
            "| Termo | 1º resultado |",
            "|---|---|",
        ]
        for termo, v in piores:
            partes.append(f"| {termo} | {v['d1']:.3f} |")
        partes += [
            "",
            f"Contra um melhor-resultado de **{fora_min:.3f}** vindo de fora do domínio.",
            "",
            "### O que fazer em vez disso",
            "",
            "O limiar corta por distância absoluta, e distância absoluta não "
            "distingue *pergunta sem resposta* de *pergunta legítima sobre um tema "
            "pouco representado*. O problema do achado 03 não está no corte:",
            "",
            "- **O agente já tem a saída certa disponível.** O `SYSTEM_PROMPT` "
            "autoriza dizer \"não encontrei\", e a tool devolve nome e departamento "
            "de cada documento desde o achado 02 — o modelo consegue julgar "
            "pertinência lendo o que veio, que é julgamento semântico, não "
            "aritmética de distância.",
            "- **Um limiar frouxo ainda tem valor**, não para calar ruído, mas para "
            "evitar que o pior dos 10 entre no contexto quando os 10 são ruins.",
            "- **Um corte relativo** (descartar o que estiver muito além do 1º "
            "resultado) responde melhor à pergunta \"estes documentos são "
            "comparáveis entre si?\" do que um corte absoluto. Fica registrado como "
            "alternativa, não implementado.",
        ]

    partes += [
        "",
        "### O que estes números NÃO são",
        "",
        "\"Relevantes no topo\" é um **piso**: a verdade-base é lexical e conta como "
        "relevante só o documento que contém o termo literalmente. Vizinho semântico "
        "legítimo — quem escreveu \"aprendizado de máquina\" para a consulta "
        "\"inteligência artificial\" — entra como não relevante. As colunas de "
        "distância, que são as que decidem o limiar, não sofrem desse viés.",
        "",
        "Uma versão anterior deste script calculava precisão sobre os "
        f"{total} documentos do corpus filtrados por distância, ignorando o TOP_K. "
        "Aquele número descrevia um sistema que não existe e foi descartado.",
        "",
    ]

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text("\n".join(partes), encoding="utf-8")
    print(f"[CALIBRAÇÃO] relatório em {SAIDA}")
    print(f"[CALIBRAÇÃO] {'recomendado: LIMIAR_DISTANCIA=%.2f' % recomendado[0] if recomendado else 'NENHUM limiar separa domínio de não-domínio.'}")


if __name__ == "__main__":
    main()
