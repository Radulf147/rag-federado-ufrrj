"""
Reconstrói o SQLite de docentes a partir do que já está no ChromaDB.

NÃO é uma etapa do pipeline — é utilitário de recuperação. Existe porque o
`sigaa.db` era escrito dentro do container do ETL sem volume montado e morria
no `--rm`. O conserto (montar ./dados) está no docker-compose.yml; este script
é o que evita ter que raspar o SIGAA de novo só para recuperar o dado
estruturado, já que os metadados (nome, departamento, siape) sobrevivem nos
chunks do Chroma.

    python -m modulo1_etl.reconstruir_sqlite [--dedup]

Tem de ser com `-m`, a partir da raiz do projeto. A forma
`python modulo1_etl/reconstruir_sqlite.py` **quebra**: ela põe a pasta do
script no sys.path em vez da raiz, e o `import config` daqui não resolve.
Foi assim que este script falhou na primeira tentativa de uso.

--dedup agrupa por (nome, departamento) em vez de por siape. Ver o comentário
em `chave_de_pessoa` para o porquê disso não ser detalhe.
"""

import argparse
import sqlite3
import sys

from haystack_integrations.document_stores.chroma import ChromaDocumentStore

import config
from modulo1_etl.db_manager import DB_PATH, init_db, salvar_entidades


def carregar_perfis() -> list[dict]:
    """Puxa um registro por siape distinto, a partir dos chunks de perfil."""
    store = ChromaDocumentStore(
        collection_name=config.CHROMA_COLECAO,
        host=config.CHROMA_HOST,
        port=config.CHROMA_PORT,
        embedding_function="default",
    )

    por_siape: dict[str, dict] = {}
    for doc in store.filter_documents():
        if doc.meta.get("content_type") != "docente_perfil":
            continue
        siape = doc.meta.get("siape")
        nome = doc.meta.get("nome_docente")
        depto = doc.meta.get("departamento")
        if not (siape and nome and depto):
            continue
        # Vários chunks compartilham o mesmo siape; o primeiro basta, os
        # metadados são idênticos entre chunks do mesmo perfil.
        por_siape.setdefault(siape, {"nome": nome, "departamento": depto, "siape": siape})

    return list(por_siape.values())


def chave_de_pessoa(registro: dict) -> tuple[str, str]:
    """
    Chave de deduplicação: (nome, departamento).

    O SIGAA registra a mesma pessoa sob vários siapes — medido em Set/2026:
    1278 siapes para 703 nomes distintos, e 353 dos 370 nomes repetidos tinham
    conteúdo de perfil IDÊNTICO, mesmo nome e mesmo departamento. Contar por
    siape infla o total em ~82%.

    O risco desta chave é fundir homônimos reais que estejam no mesmo
    departamento. Dado que os perfis duplicados são byte a byte iguais, o
    risco oposto (contar a mesma pessoa 5 vezes) é muito maior e já está
    materializado nos dados.
    """
    return (registro["nome"], registro["departamento"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dedup",
        action="store_true",
        help="agrupa por (nome, departamento) em vez de contar cada siape",
    )
    args = parser.parse_args()

    registros = carregar_perfis()
    if not registros:
        print("[ERRO] Nenhum perfil de docente encontrado no Chroma.")
        print("       O ETL já rodou? Verifique com ./rag.sh status.")
        return 1

    print(f"[CHROMA] {len(registros)} siapes distintos recuperados.")

    if args.dedup:
        unicos: dict[tuple[str, str], dict] = {}
        for r in registros:
            unicos.setdefault(chave_de_pessoa(r), r)
        removidos = len(registros) - len(unicos)
        registros = list(unicos.values())
        print(f"[DEDUP] {removidos} duplicatas removidas -> {len(registros)} pessoas.")

    # Zera a tabela antes: este script reconstrói, não acumula.
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM entidades_sigaa WHERE tipo_entidade = 'docente'")
        conn.commit()

    # substituir=True: reconstruir é refazer o retrato, não somar a ele.
    salvar_entidades("docente", registros, substituir=True)
    print(f"[SQLITE] {len(registros)} docentes gravados em {DB_PATH}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
