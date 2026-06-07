# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 4: embedding dos chunks

import logging
from pathlib import Path
from datetime import datetime
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/embedding_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info

# Modelo multilingual para português — trocar por rufimelo/bert-large-portuguese-cased-sts em produção
MODELO_EMBEDDING = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM    = 1024    # 1024 para a faculdade, 384 para local pc fraco
INSTANCIA        = "sigaa"

def embedar_documentos(documentos: list[Document]) -> list[Document]:
    # Vetoriza cada chunk com o modelo de embedding. Retorna docs com campo embedding preenchido.
    if not documentos:
        log("[EMBEDDING] Nenhum documento recebido.")
        return []

    log(f"[EMBEDDING] {len(documentos)} chunks — modelo: {MODELO_EMBEDDING}")

    embedder = SentenceTransformersDocumentEmbedder(model=MODELO_EMBEDDING)
    embedder.warm_up()

    docs_embedados = embedder.run(documents=documentos)["documents"]

    sem_embedding = [i for i, d in enumerate(docs_embedados) if not d.embedding]
    if sem_embedding:
        log(f"[EMBEDDING] deu erro {len(sem_embedding)} docs sem embedding.")
    else:
        log(f"[EMBEDDING] deu certo {len(docs_embedados)} chunks vetorizados (dim={len(docs_embedados[0].embedding)}).")

    return docs_embedados

def validar_embeddings(documentos: list[Document]) -> bool:
    # Verifica embeddings preenchidos, dimensão correta e metadados de governança intactos.
    if not documentos:
        log("[VALIDAÇÃO] erro Nenhum documento.")
        return False

    erros = 0

    sem_embedding = [i for i, d in enumerate(documentos) if not d.embedding]
    if sem_embedding:
        log(f"[VALIDAÇÃO] erro {len(sem_embedding)} docs sem embedding.")
        erros += 1
    else:
        log("[VALIDAÇÃO] deu certo Todos os docs têm embedding.")

    dim_errada = [i for i, d in enumerate(documentos) if d.embedding and len(d.embedding) != EMBEDDING_DIM]
    if dim_errada:
        log(f"[VALIDAÇÃO] erro Dimensão incorreta em {len(dim_errada)} docs (esperado {EMBEDDING_DIM}).")
        erros += 1
    else:
        log(f"[VALIDAÇÃO] deu certo Dimensão {EMBEDDING_DIM} confirmada.")

    for campo in ["instancia_dona", "source_url", "scraped_at"]:
        faltando = [i for i, d in enumerate(documentos) if campo not in d.meta]
        if faltando:
            log(f"[VALIDAÇÃO] erro '{campo}' ausente em {len(faltando)} docs.")
            erros += 1
        else:
            log(f"[VALIDAÇÃO] deu certo '{campo}' OK.")

    errados = [d for d in documentos if d.meta.get("instancia_dona") != INSTANCIA]
    if errados:
        log(f"[VALIDAÇÃO] erro {len(errados)} docs com instancia_dona incorreto.")
        erros += 1
    else:
        log(f"[VALIDAÇÃO] deu certo instancia_dona = '{INSTANCIA}'.")

    return erros == 0

if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 4 — EMBEDDING")
    log("=" * 60)

    from parte1_scraping_home import scrape_sigaa
    from parte2_scraping_docentes import scrape_docentes
    from parte3_chunking import chunkar_documentos

    chunks = chunkar_documentos(scrape_sigaa() + scrape_docentes())

    if not chunks:
        log("[RESULTADO] Nenhum chunk gerado.")
        exit(1)

    log(f"[SETUP] {len(chunks)} chunks prontos.")

    docs_embedados = embedar_documentos(chunks)

    if not docs_embedados:
        log("[RESULTADO] Embedding falhou.")
        exit(1)

    validar_embeddings(docs_embedados)

    log("\n- Amostra (primeiros 3 chunks) -")
    for i, doc in enumerate(docs_embedados[:3]):
        log(f"  [{i+1}] {doc.content[:100]}...")
        log(f"       embedding: [{doc.embedding[0]:.4f}, ..., {doc.embedding[-1]:.4f}] dim={len(doc.embedding)}")

    log(f"\n[RESUMO] {len(docs_embedados)} chunks vetorizados | dim={EMBEDDING_DIM} | modelo={MODELO_EMBEDDING}")
    log("[PARTE 4 CONCLUÍDA]")