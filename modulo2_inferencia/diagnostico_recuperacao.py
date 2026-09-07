"""
Diagnóstico da recuperação — POR QUE o recall é 15%.

    docker compose run --rm agente python -m modulo2_inferencia.diagnostico_recuperacao

A linha de base (`medir_recuperacao.py`) mediu QUANTO se perde. Isto tenta
descobrir ONDE. Não corrige nada e não altera nada: só lê o Chroma.

O ENIGMA QUE MOTIVA
-------------------
`FORMACAO DOCENTE` acerta 70% das vagas do TOP_10; `FORMACAO DE PROFESSORES`,
que é quase sinônimo, acerta 0%. Nenhuma das três hipóteses do
`docs/backlog_avaliacao.md` item 7 explica isso, e tentar corrigir sem explicar
seria mexer sem saber.

AS QUATRO PERGUNTAS
-------------------
A. As distâncias SEPARAM alguma coisa? Se a mediana da distância aos docentes do
   gabarito for igual à do corpus inteiro, o embedding não carrega sinal nenhum
   para aquela frase, e mexer no texto indexado não resolve.

B. CONTROLE DE ABSURDO. Uma consulta sem relação com o corpus deve ficar mais
   longe que uma consulta legítima. Se ficar à mesma distância, o espaço está
   colapsado e o problema não é o texto — é o embedding neste corpus.
   `culinária japonesa medieval` é a consulta que `docs/calibracao_limiar.md` já
   usou para isso; reusada aqui de propósito, para os dois números serem
   comparáveis.

C. O TAMANHO DO DOCUMENTO decide? Se o TOP_10 for sistematicamente mais curto
   que o corpus, o texto institucional está dominando os documentos longos —
   é a hipótese 1 do item 7, e ela fica testável antes de qualquer recarga.

D. QUEM OCUPA AS VAGAS quando o recall é zero, e o que esses documentos têm.

Imprime intermediários antes de cada resultado, pela regra de
`docs/relatorio_fase5.md` §10.
"""

import statistics
import unicodedata

import config

# Os dois lados do enigma, mais a origem da investigação.
TEMAS = [
    "formacao docente",
    "formacao de professores",
    "politicas publicas",
    "inteligencia artificial",
]

# Reusada de docs/calibracao_limiar.md de propósito: os dois números passam a
# ser comparáveis entre si.
ABSURDO = "culinaria japonesa medieval"


def _norm(texto: str) -> str:
    d = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in d if not unicodedata.combining(c)).upper()


def main() -> None:
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    store = ChromaDocumentStore(
        collection_name=config.CHROMA_COLECAO,
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        embedding_function="default",
    )
    docs = store.filter_documents()

    from modulo2_inferencia.llm_setup import montar_componentes
    comp = montar_componentes()

    tamanhos = [len(d.content or "") for d in docs]
    print("=" * 76)
    print("INTERMEDIARIOS")
    print("=" * 76)
    print(f"  documentos ......... {len(docs)}")
    print(f"  tamanho: mediana {statistics.median(tamanhos):.0f} chars"
          f"  | min {min(tamanhos)}  max {max(tamanhos)}")
    print(f"  embedding .......... {comp.embedder.model}")
    print(f"  limiar de producao . {config.LIMIAR_DISTANCIA if hasattr(config, 'LIMIAR_DISTANCIA') else '(em tools.py)'}")

    def ranquear(consulta: str):
        emb = comp.embedder.run(text=consulta)["embedding"]
        return comp.retriever.run(query_embedding=emb, top_k=len(docs))["documents"]

    # ---------------------------------------------------------------- A e B ---
    print()
    print("=" * 76)
    print("A. AS DISTANCIAS SEPARAM?  +  B. CONTROLE DE ABSURDO")
    print("=" * 76)
    print(f"{'consulta':30} {'gab':>4} {'d.1o':>6} {'d.gab':>7} {'d.corpus':>9} {'separa?':>9}")
    print("-" * 76)

    linhas = []
    for consulta in TEMAS + [ABSURDO]:
        ranking = ranquear(consulta)
        alvo = _norm(consulta)
        dist_gab, dist_todos = [], []
        for d in ranking:
            dist_todos.append(d.score)
            if alvo in _norm(d.content or ""):
                dist_gab.append(d.score)
        med_gab = statistics.median(dist_gab) if dist_gab else float("nan")
        med_todos = statistics.median(dist_todos)
        delta = med_todos - med_gab if dist_gab else float("nan")
        marca = "-" if not dist_gab else (f"{delta:+.3f}")
        eh_absurdo = consulta == ABSURDO
        print(f"{consulta[:29]:30} {len(dist_gab):>4} {ranking[0].score:6.3f} "
              f"{med_gab:7.3f} {med_todos:9.3f} {marca:>9}"
              + ("   <- ABSURDO" if eh_absurdo else ""))
        linhas.append((consulta, ranking, dist_todos))

    print()
    print("  d.1o     = distancia do 1o colocado")
    print("  d.gab    = MEDIANA da distancia aos docentes que escreveram a frase")
    print("  d.corpus = MEDIANA da distancia a TODOS os 1302")
    print("  separa?  = quanto o gabarito esta mais perto que o corpus.")
    print("             perto de zero = o embedding nao distingue quem escreveu")
    print("             a frase de quem nao escreveu.")

    faixas = {c: (min(t), max(t)) for c, _, t in linhas}
    print()
    print("  FAIXA DE DISTANCIA DO CORPUS INTEIRO, por consulta:")
    for c, (lo, hi) in faixas.items():
        print(f"    {c[:34]:36} {lo:.3f} .. {hi:.3f}   (amplitude {hi - lo:.3f})")

    # ------------------------------------------------------------------- C ---
    print()
    print("=" * 76)
    print("C. O TAMANHO DO DOCUMENTO DECIDE O RANKING?")
    print("=" * 76)
    mediana_corpus = statistics.median(tamanhos)
    print(f"{'consulta':30} {'med.TOP10':>10} {'med.corpus':>11} {'med.gabarito':>13}")
    print("-" * 76)
    for consulta, ranking, _ in linhas:
        alvo = _norm(consulta)
        top10 = [len(d.content or "") for d in ranking[:10]]
        gab = [len(d.content or "") for d in ranking if alvo in _norm(d.content or "")]
        m_gab = f"{statistics.median(gab):.0f}" if gab else "-"
        print(f"{consulta[:29]:30} {statistics.median(top10):10.0f} "
              f"{mediana_corpus:11.0f} {m_gab:>13}")

    # ------------------------------------------------------------------- D ---
    print()
    print("=" * 76)
    print("D. QUEM OCUPA AS VAGAS -- os dois lados do enigma")
    print("=" * 76)
    for consulta in ("formacao docente", "formacao de professores"):
        ranking = ranquear(consulta)
        alvo = _norm(consulta)
        print(f"\n--- {consulta!r}")
        print(f"{'#':>2} {'dist':>6} {'chars':>6}  {'docente':32} escreveu?")
        for i, d in enumerate(ranking[:10], 1):
            tem = alvo in _norm(d.content or "")
            print(f"{i:2} {d.score:6.3f} {len(d.content or ''):6}  "
                  f"{(d.meta.get('nome_docente') or '')[:31]:32} {'SIM' if tem else '-'}")


if __name__ == "__main__":
    main()
