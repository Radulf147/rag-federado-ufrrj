"""
Terceira hipótese do item 7: falta uma busca por PALAVRA ao lado da semântica.

    docker compose run --rm agente python -m modulo2_inferencia.medir_hibrido

⚠️ O PROBLEMA DE MEDIR ISTO, E ELE É SÉRIO
-------------------------------------------
O gabarito deste projeto é **casamento literal da frase**. Uma busca por palavra
consultada com a própria frase acerta **100% por construção** — não é resultado,
é a definição do gabarito devolvida em outro formato. Publicar esse número como
evidência de que busca por palavra funciona seria fraude estatística, ainda que
involuntária.

Por isso o teste tem DUAS partes, e só a segunda vale como evidência:

  PARTE A — consulta = a frase exata.
      A busca por palavra dá ~100% POR CONSTRUÇÃO. Não é achado. Serve para
      duas coisas honestas: mostrar o TETO alcançável, e medir quanto um
      híbrido recupera do que a semântica perde.

  PARTE B — consulta = uma PARÁFRASE que não contém a frase.
      É o caso real: ninguém digita o rótulo do campo do SIGAA. Aqui a busca
      por palavra pode falhar, e é isto que decide a hipótese.

AS PARÁFRASES SÃO ESCRITAS ANTES DE RODAR
-----------------------------------------
Estão fixadas abaixo, no código, e o commit deste arquivo antecede o do
resultado. Foram escritas para EVITAR a frase do gabarito de propósito: se eu
deixasse a palavra escapar, estaria devolvendo a Parte A disfarçada de Parte B.

⚠️ PREVISÃO, ANTES DE RODAR (7 set 2026)
-----------------------------------------
10. PARTE A: busca por palavra ~100%. Tautologia, e vem marcada como tal.
11. PARTE B: a busca por palavra CAI muito — a paráfrase não tem os termos —, e
    fica PIOR que a semântica. É para isso que a semântica existe.
12. O híbrido (RRF) fica >= a melhor das duas nas DUAS partes, sem ficar abaixo
    da semântica na Parte B.

**O QUE ME DERRUBA:** se na Parte B a busca por palavra empatar ou ganhar da
semântica, então o embedding não está agregando nada além do que a sobreposição
de palavras já dá, e a conclusão do projeto sobre precisar de busca semântica
fica sem apoio neste corpus.

BM25 ESCRITO À MÃO, e de propósito
----------------------------------
São ~30 linhas, os parâmetros estão à vista (k1=1.5, b=0.75, os padrões da
literatura) e não há biblioteca escolhendo por nós. Uma dependência a mais
esconderia decisões dentro dela.
"""

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

import config
from modulo2_inferencia.medir_recuperacao import (
    _abrir_store,
    escolher_temas,
    gabarito,
)

# Escritas ANTES de rodar, evitando a frase do gabarito de propósito.
# A chave é o tema como ele sai do corpus; o valor é o que um aluno digitaria.
PARAFRASES = {
    "POLITICAS PUBLICAS": "quem estuda acao do governo e gestao estatal",
    "FORMACAO DE PROFESSORES": "quem prepara futuros educadores para a escola",
    "EDUCACAO ESPECIAL": "docentes da area de inclusao e deficiencia",
    "FORMACAO DOCENTE": "capacitacao de quem vai lecionar",
    "TEORIA DA HISTORIA": "historiografia e pensamento sobre o passado",
    "inteligencia artificial": "IA e aprendizado de maquina",
}

# Palavras que aparecem em quase todo perfil e não discriminam nada. Lista
# curta e à vista: uma lista grande viraria um filtro que ninguém audita.
VAZIAS = {
    "DE", "DA", "DO", "DAS", "DOS", "E", "EM", "NA", "NO", "NAS", "NOS",
    "A", "O", "AS", "OS", "UM", "UMA", "PARA", "POR", "COM", "QUE", "QUEM",
    "SE", "SOBRE", "AO", "AOS", "PELA", "PELO", "SUA", "SEU",
}

K1, B = 1.5, 0.75      # padrões da literatura
K_RRF = 60             # padrão do artigo original de Reciprocal Rank Fusion


def _norm(texto: str) -> str:
    d = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in d if not unicodedata.combining(c)).upper()


def tokenizar(texto: str) -> list[str]:
    return [
        t for t in re.split(r"[^A-Z0-9]+", _norm(texto))
        if len(t) >= 3 and t not in VAZIAS
    ]


class BM25:
    def __init__(self, documentos: list):
        self.docs = documentos
        self.tokens = [tokenizar(d.content or "") for d in documentos]
        self.tam = [len(t) for t in self.tokens]
        self.media = sum(self.tam) / max(len(self.tam), 1)
        self.freq = [Counter(t) for t in self.tokens]
        aparicoes: Counter = Counter()
        for t in self.tokens:
            aparicoes.update(set(t))
        n = len(documentos)
        self.idf = {
            termo: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for termo, df in aparicoes.items()
        }

    def ranquear(self, consulta: str) -> list:
        termos = tokenizar(consulta)
        notas = []
        for i, freq in enumerate(self.freq):
            nota = 0.0
            for termo in termos:
                f = freq.get(termo, 0)
                if not f:
                    continue
                norma = 1 - B + B * self.tam[i] / max(self.media, 1)
                nota += self.idf.get(termo, 0) * f * (K1 + 1) / (f + K1 * norma)
            if nota > 0:
                notas.append((nota, i))
        notas.sort(key=lambda x: -x[0])
        return [self.docs[i] for _, i in notas]


def fundir(*rankings: list) -> list:
    """
    Reciprocal Rank Fusion — sem peso a calibrar, e é essa a graça.

    ⚠️ A CHAVE É A IDENTIDADE (D1, 7 set 2026). Chaveado por nome, dois
    homônimos davam DOIS erros de uma vez: o segundo sumia da saída **e a
    pontuação dele somava na do primeiro** — um documento perdido e outro com
    nota inflada. Pego por `testes/test_identidade_entidade.py`, escrito antes
    da correção e que reprovou a versão anterior.
    """
    from interfaces.identidade import identidade

    nota: dict = {}
    guarda: dict = {}
    for ranking in rankings:
        for posicao, doc in enumerate(ranking, 1):
            chave = identidade(doc.meta)
            nota[chave] = nota.get(chave, 0.0) + 1.0 / (K_RRF + posicao)
            guarda.setdefault(chave, doc)
    ordem = sorted(nota, key=lambda c: -nota[c])
    return [guarda[c] for c in ordem]


def recall(ranking: list, esperados: set, k: int) -> int:
    from interfaces.identidade import identidade

    vistos = {identidade(d.meta) for d in ranking[:k]}
    return len(esperados & vistos)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--colecao", default=f"{config.CHROMA_COLECAO}_descritivo")
    parser.add_argument("--temas-de", default=config.CHROMA_COLECAO)
    parser.add_argument("--saida", type=Path,
                        default=Path("docs/recuperacao_hibrida.json"))
    args = parser.parse_args()

    print("=" * 78)
    print("INTERMEDIARIOS")
    print("=" * 78)
    docs = _abrir_store(args.colecao).filter_documents()
    fonte = _abrir_store(args.temas_de).filter_documents()
    print(f"  colecao medida ..... {args.colecao}  ({len(docs)} docs)")
    print(f"  gabarito e temas de  {args.temas_de}  ({len(fonte)} docs)")

    temas = escolher_temas(fonte, 8)
    print(f"  temas .............. {len(temas)}")

    bm25 = BM25(docs)
    print(f"  BM25: {len(bm25.idf)} termos distintos,"
          f" documento medio {bm25.media:.0f} tokens (k1={K1}, b={B})")

    faltando = [t for t in temas if t not in PARAFRASES]
    if faltando:
        raise SystemExit(
            f"ABORTADO: sem parafrase escrita para {faltando}. Escrever uma "
            "agora, DEPOIS de ver os temas, seria escolher a consulta sabendo "
            "o que ela vai medir."
        )

    from modulo2_inferencia.llm_setup import montar_componentes
    comp = montar_componentes()
    from haystack_integrations.components.retrievers.chroma import (
        ChromaEmbeddingRetriever,
    )
    comp.retriever = ChromaEmbeddingRetriever(
        document_store=_abrir_store(args.colecao), top_k=config.TOP_K
    )
    print(f"  embedding .......... {comp.embedder.model}")

    def semantico(consulta: str) -> list:
        emb = comp.embedder.run(text=consulta)["embedding"]
        return comp.retriever.run(query_embedding=emb, top_k=len(docs))["documents"]

    resultados = []
    for parte, usar_parafrase in (("A (frase exata)", False), ("B (parafrase)", True)):
        print()
        print("=" * 78)
        titulo = f"PARTE {parte}"
        if not usar_parafrase:
            titulo += "   -- palavra e 100% POR CONSTRUCAO, nao e achado"
        print(titulo)
        print("=" * 78)
        print(f"{'tema':26} {'gab':>4} {'semantico':>10} {'palavra':>9} {'hibrido':>9}")
        print("-" * 78)
        soma = {"sem": 0, "pal": 0, "hib": 0, "gab": 0}
        for tema in temas:
            esperados = gabarito(fonte, tema)
            # `gabarito` já devolve identidades (D1); o filtro compara com as
            # identidades da coleção medida, não com nomes.
            from interfaces.identidade import identidade as _id
            esperados = {n for n in esperados
                         if n in {_id(d.meta) for d in docs}}
            consulta = PARAFRASES[tema] if usar_parafrase else tema
            r_sem = semantico(consulta)
            r_pal = bm25.ranquear(consulta)
            r_hib = fundir(r_sem, r_pal)
            s, p, h = (recall(r, esperados, 10) for r in (r_sem, r_pal, r_hib))
            g = len(esperados)
            soma["sem"] += s; soma["pal"] += p; soma["hib"] += h; soma["gab"] += g
            print(f"{tema[:25]:26} {g:>4} {s:>6}/{g:<3} {p:>5}/{g:<3} {h:>5}/{g:<3}")
            resultados.append({
                "parte": parte, "tema": tema, "consulta": consulta,
                "gabarito": g, "semantico": s, "palavra": p, "hibrido": h,
            })
        g = soma["gab"]
        print("-" * 78)
        print(f"{'TOTAL':26} {g:>4} {soma['sem']:>6}/{g:<3} "
              f"{soma['pal']:>5}/{g:<3} {soma['hib']:>5}/{g:<3}")
        if not usar_parafrase:
            print("\n  ^ o 'palavra' desta parte e TAUTOLOGIA: o gabarito E o")
            print("    casamento literal. Nao entra como evidencia.")

    args.saida.write_text(json.dumps({
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "colecao": args.colecao,
        "parafrases": PARAFRASES,
        "bm25": {"k1": K1, "b": B, "k_rrf": K_RRF},
        "aviso": ("na parte A a busca por palavra e 100% por construcao; "
                  "so a parte B vale como evidencia"),
        "resultados": resultados,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nEscrito em {args.saida}")


if __name__ == "__main__":
    main()
