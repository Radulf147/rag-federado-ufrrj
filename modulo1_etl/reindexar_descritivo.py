"""
Reindexa o corpus vetorizando SÓ o que o docente escreveu sobre si.

    docker compose run --rm agente python -m modulo1_etl.reindexar_descritivo

O QUE MUDA
----------
Hoje o texto vetorizado é o documento inteiro:

    Docente: MONICA PINHEIRO FERNANDES. Departamento: DEPARTAMENTO DE FORMAÇÃO
    DOCENTE/IM. Currículo Lattes: link não informado Sala: 305 Telefone: 248
    E-mail: monicapinheiro@uol.com.br

Aqui passa a ser apenas `Perfil`, `Formação` e `Áreas de interesse`. No exemplo
acima isso é **vazio**, e o documento sai do índice semântico.

POR QUE, E O QUE FOI MEDIDO ANTES
---------------------------------
`docs/backlog_avaliacao.md` item 7. A busca semântica acha 14% (mediana) de quem
escreveu a frase no perfil, e o TOP_10 é 7× mais curto que o gabarito: perfis
vazios, cujo único conteúdo indexado é o NOME DO DEPARTAMENTO, ocupam as vagas.
Para a consulta `formação de professores`, os três primeiros são pessoas do
`DEPARTAMENTO DE FORMAÇÃO DOCENTE/IM` que não escreveram nada.

Duas consequências, as duas desejadas:

1. A colisão com nome temático de departamento deixa de existir.
2. Perfil sem conteúdo descritivo sai do índice semântico. **É o correto:** não
   há nada semântico num documento que só diz onde a pessoa trabalha. Eles
   continuam no SQLite, que é quem responde contagem e listagem, e é por onde
   essas perguntas já são roteadas.

O NOME E O DEPARTAMENTO NÃO SE PERDEM
-------------------------------------
Continuam no METADADO, e `busca_vetorial_sigaa` já monta o cabeçalho de cada
documento recuperado a partir dele (correção do achado 02). O agente segue
sabendo de quem é cada trecho; o que muda é que o nome do departamento deixa de
COMPETIR pela similaridade.

DUAS MUDANÇAS DE UMA VEZ — E POR ISSO EXISTE O MODO `filtrado`
--------------------------------------------------------------
Indexar só o descritivo faz **duas** coisas ao mesmo tempo:

    (a) tira o texto institucional do que é vetorizado
    (b) remove do índice quem não tem conteúdo descritivo nenhum

Se eu medisse só o resultado final, atribuiria o ganho inteiro a (a) — e (b)
sozinho já melhora o recall por aritmética, porque tira concorrentes da disputa
pelas 10 vagas. Seriam duas causas confundidas num número só.

    --modo filtrado      texto ORIGINAL, só quem tem descritivo   -> isola (b)
    --modo descritivo    texto descritivo, só quem tem            -> (a) + (b)

A diferença entre as duas coleções é o efeito de (a) sozinho. Sem esse
controle, a conclusão seria plausível e sem apoio.

ESCREVE NUMA COLEÇÃO NOVA
-------------------------
Nada é sobrescrito. A coleção original fica intacta, as duas podem ser medidas
lado a lado, e voltar atrás é trocar uma variável de ambiente. Isto não é
excesso de zelo: a hipótese pode estar errada, e o critério de aceite
(`docs/recuperacao_linha_de_base.json`) pode reprovar a mudança.

NÃO TOCA NO SIGAA. O conteúdo já está gravado; isto lê, recorta e re-vetoriza.
"""

import argparse

from haystack import Document

import config
from interfaces.respaldo import texto_descritivo

COLECAO_NOVA = f"{config.CHROMA_COLECAO}_descritivo"


def _store(colecao: str):
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    return ChromaDocumentStore(
        collection_name=colecao,
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        embedding_function="default",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origem", default=config.CHROMA_COLECAO)
    parser.add_argument("--destino", default=None)
    parser.add_argument(
        "--modo", choices=("descritivo", "filtrado"), default="descritivo",
        help=(
            "descritivo: indexa so o texto descritivo (a correcao proposta). "
            "filtrado: mantem o texto ORIGINAL inteiro, apenas excluindo quem "
            "nao tem conteudo descritivo. Serve de CONTROLE."
        ),
    )
    args = parser.parse_args()
    if args.destino is None:
        args.destino = f"{config.CHROMA_COLECAO}_{args.modo}"

    print("=" * 74)
    print("INTERMEDIARIOS")
    print("=" * 74)
    origem = _store(args.origem)
    docs = origem.filter_documents()
    print(f"  origem ............. {args.origem}  ({len(docs)} documentos)")
    print(f"  destino ............ {args.destino}")
    print(f"  embedding .......... {config.MODELO_EMBEDDING}"
          f"  dim {config.EMBEDDING_DIM}")

    novos, vazios = [], []
    for d in docs:
        descritivo = texto_descritivo(d.content or "")
        if not descritivo.strip():
            vazios.append(d.meta.get("nome_docente"))
            continue
        # O metadado vai INTEIRO: nome, departamento, siape e source_url
        # continuam disponíveis para a tool montar o cabeçalho. O que muda é só
        # o que entra no vetor.
        meta = dict(d.meta)
        meta["indexacao"] = args.modo
        meta["chars_originais"] = len(d.content or "")
        # No modo `filtrado` o texto indexado continua sendo o ORIGINAL. A
        # única diferença para a coleção de produção é quem ficou de fora — e é
        # exatamente isso que ele isola.
        conteudo = descritivo if args.modo == "descritivo" else (d.content or "")
        novos.append(Document(content=conteudo, meta=meta))

    print(f"\n  com conteudo descritivo .. {len(novos)}"
          f"  ({100 * len(novos) / max(len(docs), 1):.0f}%)")
    print(f"  SEM conteudo (excluidos) . {len(vazios)}"
          f"  ({100 * len(vazios) / max(len(docs), 1):.0f}%)")

    if novos:
        import statistics
        antes = statistics.median(len(d.content or "") for d in docs)
        depois = statistics.median(len(d.content) for d in novos)
        print(f"\n  tamanho mediano ANTES .... {antes:.0f} chars (corpus inteiro)")
        print(f"  tamanho mediano DEPOIS ... {depois:.0f} chars (so os indexados)")

    print(f"\n  exemplo de excluido: {vazios[0] if vazios else '-'}")
    if novos:
        print(f"  exemplo de indexado: {novos[0].meta.get('nome_docente')}")
        print(f"      {novos[0].content[:160]}...")

    print()
    print("=" * 74)
    print("VETORIZANDO")
    print("=" * 74)
    from haystack.components.embedders import SentenceTransformersDocumentEmbedder

    embedder = SentenceTransformersDocumentEmbedder(model=config.MODELO_EMBEDDING)
    embedder.warm_up()
    com_vetor = embedder.run(documents=novos)["documents"]
    dim = len(com_vetor[0].embedding) if com_vetor else 0
    print(f"  vetorizados {len(com_vetor)}, dimensao {dim}")
    if dim != config.EMBEDDING_DIM:
        raise SystemExit(
            f"ABORTADO: dimensao {dim} != EMBEDDING_DIM {config.EMBEDDING_DIM}. "
            "Indice e consulta ficariam em espacos diferentes, e isso nao daria "
            "erro nenhum depois — so recuperacao ruim (armadilha 3)."
        )

    destino = _store(args.destino)
    destino.write_documents(com_vetor)
    gravados = len(destino.filter_documents())
    print(f"  gravados em {args.destino}: {gravados}")
    if gravados != len(com_vetor):
        raise SystemExit(
            f"ABORTADO: gravei {len(com_vetor)} e a colecao tem {gravados}."
        )

    print()
    print("A colecao original permanece intacta. Para medir a nova:")
    print(f"  python -m modulo2_inferencia.medir_recuperacao --colecao {args.destino}")


if __name__ == "__main__":
    main()
