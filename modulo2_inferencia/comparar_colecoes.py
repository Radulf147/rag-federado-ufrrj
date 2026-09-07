"""
A mesma pergunta nas duas coleções — comparando a RESPOSTA, não o recall.

    docker compose run --rm agente python -m modulo2_inferencia.comparar_colecoes

Pré-registro em `docs/pre_registro_troca_colecao.md`, commitado ANTES desta
execução. As perguntas, as previsões e os rótulos de classificação estão lá e
não são reescritos aqui.

O QUE ISTO MEDE, E POR QUE NÃO BASTAVA O QUE JÁ ESTAVA MEDIDO
------------------------------------------------------------
`docs/backlog_avaliacao.md` item 7 mediu recall@10: 14% -> 27% ao indexar só o
texto descritivo. O que ele NÃO pode medir é o efeito de ter tirado 556
docentes do índice, porque quem não escreveu perfil nunca esteve em gabarito
nenhum. Removê-los só pode subir aquela métrica e só pode piorar a resposta
para quem perguntar sobre eles.

Aqui a unidade não é o documento recuperado: é o texto que o usuário lê.

TRÊS EXECUÇÕES POR CÉLULA, E ISSO NÃO É ZELO
--------------------------------------------
Os posts 10 e 19 da rede simulada respondem à mesma pergunta com listas
diferentes, na mesma coleção. Uma execução por célula mediria o sorteio. As
três ficam gravadas separadas; divergência entre elas é resultado, não ruído a
esconder numa média.

A PARTE B NÃO CHAMA O LLM
-------------------------
A previsão 15 é sobre o limiar de distância, e para testá-la basta o
recuperador. Roda separada, é barata, e não depende do modelo estar de pé.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import config
from modulo2_inferencia.agent import montar_historico_inicial, processar_pergunta
from modulo2_inferencia.medir_recuperacao import _abrir_store

# Copiadas do pré-registro. Os grupos A e B são o texto LITERAL dos posts da
# rede simulada (ids 1, 4, 6 e 16), não reescritas.
PERGUNTAS = [
    ("A1", "controle",
     "quantos professores tem o Departamento de Computação hoje?"),
    ("A2", "real",
     "quais docentes do Departamento de Ciência da Computação trabalham com "
     "inteligência artificial?"),
    ("A3", "controle",
     "qual a nota de corte da monitoria de Cálculo 1 esse ano?"),
    ("B1", "real",
     "Quais docentes de computacao do IM sao de IA?"),
    ("C1", "decide",
     "o que o professor Leandro Alvim pesquisa?"),
    ("C2", "decide",
     "quais são as áreas de interesse do professor Marcel William Rocha da Silva?"),
]

# Os dois docentes do grupo C, com o nome como está no metadado. Usado só para
# responder "o documento da própria pessoa foi recuperado?".
ALVOS_C = {
    "C1": "LEANDRO GUIMARAES MARQUES ALVIM",
    "C2": "MARCEL WILLIAM ROCHA DA SILVA",
}


def _tools_chamadas(historico) -> list[str]:
    """Nomes das tools acionadas no turno, na ordem."""
    nomes = []
    for msg in historico:
        for chamada in (getattr(msg, "tool_calls", None) or []):
            nome = getattr(chamada, "tool_name", None) or getattr(
                chamada, "name", "?"
            )
            nomes.append(nome)
    return nomes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--colecoes", nargs="+",
        default=[config.CHROMA_COLECAO, f"{config.CHROMA_COLECAO}_descritivo"],
    )
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--saida", type=Path,
                        default=Path("docs/comparacao_colecoes.json"))
    parser.add_argument("--so-recuperacao", action="store_true",
                        help="roda so a parte B (sem LLM)")
    args = parser.parse_args()

    print("=" * 78)
    print("INTERMEDIARIOS")
    print("=" * 78)
    from modulo2_inferencia.llm_setup import montar_componentes
    from haystack_integrations.components.retrievers.chroma import (
        ChromaEmbeddingRetriever,
    )

    comp = montar_componentes()
    stores = {c: _abrir_store(c) for c in args.colecoes}
    for nome, store in stores.items():
        print(f"  {nome:26} {len(store.filter_documents()):5} documentos")
    print(f"  embedding .......... {comp.embedder.model}")
    print(f"  llm ................ {config.MODELO_LLM}")
    from modulo2_inferencia.tools import LIMIAR_DISTANCIA
    print(f"  limiar ............. {LIMIAR_DISTANCIA}")
    print(f"  top_k .............. {config.TOP_K}")
    print(f"  perguntas .......... {len(PERGUNTAS)}"
          f"  x {len(args.colecoes)} colecoes x {args.repeticoes} execucoes"
          f" = {len(PERGUNTAS) * len(args.colecoes) * args.repeticoes}")

    retrievers = {
        c: ChromaEmbeddingRetriever(document_store=s, top_k=config.TOP_K)
        for c, s in stores.items()
    }

    # ------------------------------------------------------------ PARTE B ---
    # Previsão 15: o limiar de 1.24 não salva o grupo C. Sem LLM.
    print()
    print("=" * 78)
    print("PARTE B -- O QUE A BUSCA DEVOLVE (previsao 15, sem LLM)")
    print("=" * 78)
    recuperacao = []
    for cid, alvo in ALVOS_C.items():
        pergunta = next(p for i, _, p in PERGUNTAS if i == cid)
        for colecao in args.colecoes:
            emb = comp.embedder.run(text=pergunta)["embedding"]
            docs = retrievers[colecao].run(query_embedding=emb)["documents"]
            nomes = [d.meta.get("nome_docente") for d in docs]
            sob_limiar = [
                d for d in docs
                if LIMIAR_DISTANCIA is None or (d.score or 0) <= LIMIAR_DISTANCIA
            ]
            registro = {
                "id": cid, "colecao": colecao, "alvo": alvo,
                "alvo_recuperado": alvo in nomes,
                "posicao_do_alvo": nomes.index(alvo) + 1 if alvo in nomes else None,
                "passaram_o_limiar": len(sob_limiar),
                "de": len(docs),
                "primeiros": [
                    {"nome": d.meta.get("nome_docente"), "dist": round(d.score, 3)}
                    for d in docs[:5]
                ],
            }
            recuperacao.append(registro)
            onde = ("SIM, posicao " + str(registro["posicao_do_alvo"])
                    if registro["alvo_recuperado"] else "NAO")
            print(f"\n--- {cid}  {colecao}")
            print(f"    alvo no TOP_{config.TOP_K}: {onde}")
            print(f"    passaram o limiar: {len(sob_limiar)} de {len(docs)}")
            for i, d in enumerate(docs[:5], 1):
                marca = " <- O PROPRIO" if d.meta.get("nome_docente") == alvo else ""
                print(f"    {i} {d.score:6.3f}  "
                      f"{(d.meta.get('nome_docente') or '')[:44]}{marca}")

    if args.so_recuperacao:
        print("\n--so-recuperacao: parando antes do LLM.")
        return

    # ------------------------------------------------------------ PARTE A ---
    print()
    print("=" * 78)
    print("PARTE A -- A RESPOSTA QUE O USUARIO LE")
    print("=" * 78)
    respostas = []
    for cid, grupo, pergunta in PERGUNTAS:
        for colecao in args.colecoes:
            for n in range(1, args.repeticoes + 1):
                print(f"\n{'=' * 78}\n{cid} [{grupo}]  {colecao}  exec {n}"
                      f"\n  > {pergunta}\n{'-' * 78}")
                try:
                    texto, historico = processar_pergunta(
                        comp.chat_generator,
                        comp.embedder,
                        retrievers[colecao],
                        montar_historico_inicial(),
                        pergunta,
                    )
                    erro = None
                except Exception as exc:          # noqa: BLE001
                    # Falha de LLM é dado, não motivo para perder as outras 35.
                    texto, historico, erro = "", [], f"{type(exc).__name__}: {exc}"
                    print(f"  !! FALHOU: {erro}")
                tools = _tools_chamadas(historico)
                if texto:
                    print(f"  tools: {tools or '(nenhuma)'}")
                    print(texto)
                respostas.append({
                    "id": cid, "grupo": grupo, "pergunta": pergunta,
                    "colecao": colecao, "execucao": n,
                    "tools": tools, "resposta": texto, "erro": erro,
                })
                # Grava a cada resposta: 36 chamadas ao LLM são longas o
                # bastante para a sessão morrer no meio, e perder tudo por
                # gravar só no fim seria erro nosso, não do modelo.
                args.saida.write_text(json.dumps({
                    "gerado_em": datetime.now().isoformat(timespec="seconds"),
                    "pre_registro": "docs/pre_registro_troca_colecao.md",
                    "colecoes": args.colecoes,
                    "repeticoes": args.repeticoes,
                    "modelo_llm": config.MODELO_LLM,
                    "modelo_embedding": comp.embedder.model,
                    "limiar": LIMIAR_DISTANCIA,
                    "top_k": config.TOP_K,
                    "aviso": ("classificacao (CERTA/ABSTEM/INVENTA/DESVIA) e "
                              "manual, contra o pre-registro; este arquivo "
                              "guarda o texto cru"),
                    "recuperacao": recuperacao,
                    "respostas": respostas,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n\nEscrito em {args.saida} ({len(respostas)} respostas)")
    print("A classificacao contra o pre-registro e MANUAL e vai no backlog.")


if __name__ == "__main__":
    main()
