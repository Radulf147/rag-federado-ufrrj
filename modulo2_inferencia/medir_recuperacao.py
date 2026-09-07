"""
Linha de base da RECUPERAÇÃO — quanto do que existe no corpus a busca acha.

    docker compose run --rm agente python -m modulo2_inferencia.medir_recuperacao

POR QUE EXISTE
--------------
Em 7 set 2026 o orientando perguntou na rede simulada quais docentes de
computação do IM trabalham com IA, e apontou um docente que faltava. Ao
investigar apareceu coisa pior: `RAIMUNDO JOSE MACARIO COSTA`, cujo perfil
começa com *"Áreas de interesse: Inteligência Artificial"*, estava na posição
**60 de 1302**. Medido para `inteligência artificial`: **recall@10 = 2/11**.

Nenhum instrumento do projeto teria acusado isso. A fase 3 mede PRECISÃO — se o
agente cita quem não devia — e é declaradamente cega a quem faltou.

O GABARITO, E A CIRCULARIDADE QUE ELE TEM
-----------------------------------------
Um docente entra no gabarito de um tema se o documento dele **contém a frase
literalmente**. É cru de propósito: é auditável, não depende de julgamento e não
precisa de rótulo humano.

⚠️ **Isso torna a busca por palavra 100% por construção.** O gabarito É o
casamento literal. O número que a linha de base por palavra produz não é
evidência de que busca por palavra funciona bem — é a definição do gabarito
devolvida. Serve só para uma coisa, e essa coisa importa: estabelece que **a
informação está no corpus e é alcançável**, então o que a busca semântica não
acha, ela não acha por escolha de ranking e não por ausência de dado.

OS TEMAS NÃO SÃO ESCOLHIDOS A DEDO
----------------------------------
Saem das próprias `Áreas de interesse` do corpus, pela frequência — mesma
disciplina de `calibrar_limiar.py`, que já tinha sido escrito assim para não
medir sobre uma lista conveniente. `inteligência artificial` aparece entre eles
se e somente se o corpus a colocar lá; ela vem marcada como ORIGEM porque foi o
caso que originou a investigação, e o leitor tem direito de saber disso ao ler o
número dela.

⚠️ PREVISÃO, ESCRITA ANTES DE RODAR (7 set 2026)
------------------------------------------------
Registro aqui para que o resultado possa me contradizer. O commit deste arquivo
antecede o commit do relatório, e a ordem está no histórico do git.

1. O recall@10 será BAIXO na maioria dos temas, não só em `inteligência
   artificial` — mediana abaixo de 50%.
2. Os temas com gabarito GRANDE terão recall@10 pior, por teto aritmético:
   10 posições não cabem 40 docentes.
3. A posição do pior documento do gabarito será MUITO maior que 10 — centenas.

**O QUE ME DERRUBA:** se o recall@10 for alto nos outros temas, o problema é
específico de `inteligência artificial` e as três hipóteses do
`docs/backlog_avaliacao.md` item 7 (texto institucional diluindo, lista longa de
interesses diluindo, falta de busca por palavra) estão erradas. Nesse caso a
investigação recomeça, e este arquivo fica como o registro de que eu previ o
contrário.

RESULTADO DA 1ª RODADA: mediana do recall@10 = **15%**. As previsões 1 e 3
confirmadas, a 2 parcial. A previsão que me derrubaria foi negada com folga.

⚠️ 2ª PREVISÃO, com o gabarito corrigido (escrita antes de rodar de novo)
--------------------------------------------------------------------------
O gabarito passou a ser o texto DESCRITIVO, sem o nome do departamento.

4. `formação docente` encolhe muito — era 47 e é nome de departamento — e o
   recall@10 dela **cai**, porque os acertos vinham dos perfis vazios daquele
   departamento.
5. `inteligência artificial`, `políticas públicas` e `teoria da história` mudam
   pouco: não são nome de departamento nenhum.
6. A mediana do recall@10 **piora** ou fica igual. Não melhora.

**O QUE ME DERRUBA AQUI:** se a mediana subir, então o gabarito contaminado
estava *escondendo* acertos em vez de inventá-los, e a leitura que fiz do
diagnóstico está errada.

NÃO ALTERA NADA DO SISTEMA. Só lê o Chroma e mede.
"""

import argparse
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import config

# Frases com menos docentes que isto tornam o recall ruído: um acerto move o
# número em 30 pontos. Frases com mais que o teto viram tema guarda-chuva
# ("educação"), onde nem se sabe o que uma resposta certa seria.
MIN_GABARITO = 6
MAX_GABARITO = 60

# Uma palavra só casa demais ("matemática" dentro de "educação matemática") e
# transforma o gabarito em outra coisa.
MIN_PALAVRAS = 2

TERMO_DE_ORIGEM = "inteligencia artificial"


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.upper().split())


def _carregar_documentos() -> list:
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    store = ChromaDocumentStore(
        collection_name=config.CHROMA_COLECAO,
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        embedding_function="default",
    )
    return store.filter_documents()


def frases_de_interesse(documentos: list) -> Counter:
    """
    As frases das `Áreas de interesse`, contadas sobre o corpus.

    Reusa `secoes` de interfaces/respaldo.py, que já sabe cortar o campo no
    ponto certo e já está coberto por teste — reescrever a separação aqui
    criaria uma segunda versão da mesma regra, e uma das duas ficaria errada.
    """
    from interfaces.respaldo import secoes

    contagem: Counter = Counter()
    for doc in documentos:
        campos = secoes(doc.content or "")
        area = campos.get("AREAS DE INTERESSE", "")
        if not area:
            continue
        vistas = set()
        for pedaco in re.split(r"[;,.]", area):
            frase = " ".join(pedaco.split()).strip()
            if len(frase.split()) < MIN_PALAVRAS or len(frase) < 8:
                continue
            vistas.add(frase)
        contagem.update(vistas)
    return contagem


def escolher_temas(documentos: list, quantos: int) -> list[str]:
    contagem = frases_de_interesse(documentos)
    elegiveis = [
        (frase, n) for frase, n in contagem.most_common()
        if MIN_GABARITO <= n <= MAX_GABARITO
    ]
    temas = [f for f, _ in elegiveis[:quantos]]
    # A origem entra mesmo que a frequência a deixe de fora, e vem MARCADA.
    if not any(_normalizar(t) == _normalizar(TERMO_DE_ORIGEM) for t in temas):
        temas.append(TERMO_DE_ORIGEM)
    return temas


def gabarito(documentos: list, tema: str) -> set[str]:
    """
    Quem escreveu a frase SOBRE SI — só Perfil, Formação e Áreas de interesse.

    ⚠️ CORRIGIDO EM 7 SET 2026. A primeira versão procurava a frase no documento
    INTEIRO, e o documento começa com `Departamento: X`. Consequência medida: a
    consulta `formação docente` marcava 47 docentes, entre eles todo mundo do
    `DEPARTAMENTO DE FORMAÇÃO DOCENTE/IM` cujo perfil está vazio — pessoas que
    não escreveram nada sobre si e entravam no gabarito pelo nome do lugar onde
    trabalham. O recall de 70% do teto naquele tema era artefato disto.

    `texto_descritivo` já existia em `interfaces/respaldo.py` e foi escrito
    exatamente para remover essa colisão, com teste próprio. Não aplicá-lo aqui
    foi descuido meu, e é a SEGUNDA medição desta linha de trabalho a errar pela
    mesma causa.

    Efeito colateral correto: quem tem perfil vazio não pode entrar em gabarito
    nenhum. Se a pessoa não escreveu, não há o que o sistema devesse achar.
    """
    from interfaces.respaldo import texto_descritivo

    alvo = _normalizar(tema)
    return {
        d.meta.get("nome_docente")
        for d in documentos
        if alvo in _normalizar(texto_descritivo(d.content or ""))
    }


def gabarito_contaminado(documentos: list, tema: str) -> set[str]:
    """A versão antiga, mantida só para reportar o TAMANHO do erro."""
    alvo = _normalizar(tema)
    return {
        d.meta.get("nome_docente")
        for d in documentos
        if alvo in _normalizar(d.content or "")
    }


def medir(componentes, documentos: list, tema: str, ks: tuple[int, ...]) -> dict:
    esperados = gabarito(documentos, tema)
    contaminado = gabarito_contaminado(documentos, tema)
    embedding = componentes.embedder.run(text=tema)["embedding"]
    ranking = componentes.retriever.run(
        query_embedding=embedding, top_k=len(documentos)
    )["documents"]

    posicao = {
        d.meta.get("nome_docente"): i
        for i, d in enumerate(ranking, 1)
        if d.meta.get("nome_docente") not in ()
    }
    posicoes = sorted(posicao[n] for n in esperados if n in posicao)

    return {
        "tema": tema,
        "origem": _normalizar(tema) == _normalizar(TERMO_DE_ORIGEM),
        "gabarito": len(esperados),
        "gabarito_contaminado": len(contaminado),
        "inflacao": len(contaminado) - len(esperados),
        "recall": {
            k: sum(1 for p in posicoes if p <= k) for k in ks
        },
        "precisao_10": sum(1 for p in posicoes if p <= 10),
        "pior_posicao": posicoes[-1] if posicoes else None,
        "mediana_posicao": posicoes[len(posicoes) // 2] if posicoes else None,
        "posicoes": posicoes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temas", type=int, default=8)
    parser.add_argument("--saida", type=Path,
                        default=Path("docs/recuperacao_linha_de_base.json"))
    args = parser.parse_args()

    # INTERMEDIÁRIOS ANTES DO RESULTADO — regra operacional fixada em
    # docs/relatorio_fase5.md §10. Número sem intermediário conferível não
    # entra em decisão.
    print("=" * 74)
    print("INTERMEDIARIOS")
    print("=" * 74)
    documentos = _carregar_documentos()
    print(f"  chroma ............. {config.CHROMA_HOST}:{config.CHROMA_PORT}"
          f" / {config.CHROMA_COLECAO}")
    print(f"  documentos ......... {len(documentos)}")

    com_area = sum(1 for d in documentos
                   if "AREAS DE INTERESSE" in _normalizar(d.content or ""))
    print(f"  com Areas de interesse  {com_area}"
          f"  ({100 * com_area / max(len(documentos), 1):.0f}%)")

    temas = escolher_temas(documentos, args.temas)
    print(f"  temas selecionados .. {len(temas)}"
          f"  (frequencia entre {MIN_GABARITO} e {MAX_GABARITO} docentes)")

    from modulo2_inferencia.llm_setup import montar_componentes
    componentes = montar_componentes()
    print(f"  embedding .......... {componentes.embedder.model}")
    print(f"  TOP_K de producao .. {config.TOP_K}")

    ks = (10, 20, 50, 100)
    resultados = [medir(componentes, documentos, t, ks) for t in temas]

    print()
    print("=" * 74)
    print("LINHA DE BASE -- recall da busca semantica contra o casamento literal")
    print("=" * 74)
    cab = (f"{'tema':34} {'gab':>4} {'infl':>5} "
           + " ".join(f"{'@'+str(k):>7}" for k in ks) + f" {'pior':>6}")
    print(cab)
    print("-" * len(cab))
    for r in sorted(resultados, key=lambda x: -x["gabarito"]):
        marca = " *" if r["origem"] else "  "
        celulas = " ".join(
            f"{r['recall'][k]:>3}/{r['gabarito']:<3}" for k in ks
        )
        pior = r["pior_posicao"] if r["pior_posicao"] is not None else "-"
        infl = f"+{r['inflacao']}" if r["inflacao"] else "-"
        print(f"{r['tema'][:32]:32}{marca} {r['gabarito']:>4} {infl:>5} "
              f"{celulas} {pior:>6}")
    print("\n  * tema que originou a investigacao (backlog item 7)")
    print("  'gab'  = docentes que escreveram a frase SOBRE SI (Perfil,")
    print("           Formacao, Areas de interesse) -- sem o nome do departamento")
    print("  'infl' = quantos o gabarito ANTIGO contava a mais, por casar com o")
    print("           nome do departamento. E o tamanho do erro da 1a medicao.")
    print("  'pior' = posicao do ultimo do gabarito no ranking de "
          f"{len(documentos)} documentos")

    with_10 = [r["recall"][10] / r["gabarito"] for r in resultados if r["gabarito"]]
    with_10.sort()
    mediana = with_10[len(with_10) // 2] if with_10 else 0
    print(f"\n  MEDIANA do recall@10 entre os temas: {100 * mediana:.0f}%")

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(
        json.dumps(
            {
                "gerado_em": datetime.now().isoformat(timespec="seconds"),
                "documentos": len(documentos),
                "embedding": componentes.embedder.model,
                "top_k_producao": config.TOP_K,
                "criterio_gabarito": "casamento literal da frase no documento",
                "circularidade_declarada": (
                    "busca por palavra daria 100% POR CONSTRUCAO; o gabarito e "
                    "o casamento literal. Serve para estabelecer que a "
                    "informacao esta no corpus, nao para comparar metodos"
                ),
                "resultados": resultados,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nEscrito em {args.saida}")


if __name__ == "__main__":
    main()
