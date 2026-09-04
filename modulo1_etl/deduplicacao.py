"""
Deduplicação de docentes — Módulo 1.

O SIGAA registra a mesma pessoa sob vários SIAPEs. Medido em Set/2026 sobre a
carga real: 1278 SIAPEs para 703 pessoas distintas, e 353 dos 370 nomes
repetidos tinham conteúdo de perfil byte a byte IDÊNTICO, mesmo nome e mesmo
departamento. Contar por SIAPE infla os totais em ~82% (Matemática aparecia
com 56 docentes em vez de 29).

Isso não é detalhe cosmético: a pergunta "quantos docentes tem o departamento
X" é justamente a pergunta objetiva e verificável da fase de validação. Com a
inflação, os três pipelines concordariam entre si e os três estariam errados —
o pior resultado possível, porque a concordância pareceria validação.

Além da contagem, a duplicata desperdiça o TOP_K do retrieval: numa busca com
TOP_K=10 chegavam 4 cópias do mesmo perfil, ocupando o lugar de 4 pessoas
diferentes.

Este módulo é usado pelo pipeline (parte5_carga) e pelo utilitário de limpeza
do store já carregado (`python -m modulo1_etl.deduplicacao`).
"""

from haystack import Document


def chave_pessoa(doc: Document) -> tuple[str, str]:
    """
    Identidade de uma pessoa: (nome, departamento) — não o SIAPE.

    O risco desta chave é fundir homônimos reais lotados no mesmo
    departamento. Dado que os perfis duplicados são idênticos, o risco oposto
    — contar a mesma pessoa cinco vezes — é maior e já está materializado nos
    dados.
    """
    return (doc.meta.get("nome_docente", ""), doc.meta.get("departamento", ""))


def deduplicar_documentos(documentos: list[Document]) -> tuple[list[Document], int]:
    """
    Mantém um SIAPE por pessoa, descartando os demais.

    Opera sobre documentos de perfil (antes do chunking), para que a duplicata
    nunca chegue a virar chunk nem a ser vetorizada — economiza embedding além
    de corrigir a contagem. Documentos que não são perfil de docente passam
    intactos.
    """
    siape_escolhido: dict[tuple[str, str], str] = {}
    for doc in documentos:
        if doc.meta.get("content_type") != "docente_perfil":
            continue
        chave = chave_pessoa(doc)
        siape = doc.meta.get("siape", "")
        # Menor SIAPE como critério estável: duas execuções do ETL escolhem o
        # mesmo registro, então a carga é reprodutível.
        if chave not in siape_escolhido or siape < siape_escolhido[chave]:
            siape_escolhido[chave] = siape

    mantidos = []
    for doc in documentos:
        if doc.meta.get("content_type") != "docente_perfil":
            mantidos.append(doc)
            continue
        if doc.meta.get("siape") == siape_escolhido.get(chave_pessoa(doc)):
            mantidos.append(doc)

    return mantidos, len(documentos) - len(mantidos)


def limpar_store() -> int:
    """
    Remove do ChromaDB já carregado os chunks das pessoas duplicadas.

    Utilitário de recuperação: evita ter que raspar o SIGAA de novo só para
    corrigir uma carga anterior. Devolve quantos chunks foram apagados.
    """
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    import config

    # Respeita CHROMA_REMOTE, como llm_setup.py e parte5_carga.py fazem.
    # Antes conectava sempre por host/porta: com CHROMA_REMOTE=False isso
    # apontava para um Chroma diferente do que guarda o dado, achava zero
    # chunks e imprimia "0 de pessoas duplicadas" — sucesso aparente tendo
    # deduplicado nada. Falha silenciosa, exatamente a classe que este
    # utilitário existe para corrigir.
    if config.CHROMA_REMOTE:
        store = ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            embedding_function="default",
        )
    else:
        store = ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            persist_path=config.CHROMA_PERSIST_DIR,
            embedding_function="default",
        )

    docs = store.filter_documents()
    mantidos, _ = deduplicar_documentos(docs)
    ids_mantidos = {d.id for d in mantidos}
    apagar = [d.id for d in docs if d.id not in ids_mantidos]

    print(f"[DEDUP] {len(docs)} chunks no store; {len(apagar)} de pessoas duplicadas.")
    if apagar:
        store.delete_documents(apagar)
        print(f"[DEDUP] {len(store.filter_documents())} chunks restantes.")
    return len(apagar)


if __name__ == "__main__":
    limpar_store()
