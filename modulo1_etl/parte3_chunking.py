# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 3: chunking estrutural

import logging
from datetime import datetime
from pathlib import Path
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/chunking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info

CHUNK_SENTENCES = 5
CHUNK_OVERLAP   = 1

def chunkar_documentos(documentos: list[Document]) -> list[Document]:
    # Divide documentos em chunks de sentenças com overlap.
    if not documentos:
        log("[CHUNKING] Nenhum documento recebido.")
        return []

    log(f"\n[CHUNKING] {len(documentos)} documentos...")

    import warnings
    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=CHUNK_SENTENCES,
        split_overlap=CHUNK_OVERLAP,
        language="pt",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splitter.warm_up()

    resultado = splitter.run(documents=documentos)
    chunks    = resultado["documents"]

    log(f"[CHUNKING] {len(documentos)} docs → {len(chunks)} chunks.")
    return chunks

if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 3 — TESTE ISOLADO DE CHUNKING")
    log("=" * 60)

    # Teste unitário para validar o arquivo
    doc_teste = Document(content="Esta é a frase um. Esta é a frase dois. Esta é a frase três. Esta é a frase quatro. Esta é a frase cinco. Esta é a frase seis.")
    chunks_teste = chunkar_documentos([doc_teste])
    
    for i, c in enumerate(chunks_teste, 1):
        log(f"Chunk {i}: {c.content}")
        
    log("[PARTE 3 CONCLUÍDA]")