# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 3: chunking estrutural

import logging
from datetime import datetime
from pathlib import Path
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter

Path("logs").mkdir(exist_ok=True)

# Cria um logger específico para este arquivo
logger = logging.getLogger("chunking")
logger.setLevel(logging.INFO)
logger.propagate = False # Impede que o log vaze para a Parte 5

# Cria os manipuladores
arquivo_handler = logging.FileHandler(
    f"logs/chunking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 
    encoding="utf-8"
)
console_handler = logging.StreamHandler()

# Define o padrão de texto do log
formatador = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
arquivo_handler.setFormatter(formatador)
console_handler.setFormatter(formatador)

# Conecta tudo ao logger
logger.addHandler(arquivo_handler)
logger.addHandler(console_handler)

# Mantém a variável 'log' funcionando
log = logger.info

CHUNK_SENTENCES = 5
CHUNK_OVERLAP   = 1

def chunkar_documentos(documentos: list[Document]) -> list[Document]:
    # Divide documentos em chunks de sentenças com overlap.
    if not documentos:
        log("[CHUNKING] Nenhum documento recebido.")
        return []

    log(f"\n[CHUNKING] {len(documentos)} documentos...")

    # ACHADO 02: perfil de docente NÃO é fatiado.
    #
    # O perfil é uma unidade semântica curta e autocontida, e só a primeira
    # sentença carrega "Docente: X. Departamento: Y.". Fatiando de 5 em 5
    # sentenças, todo pedaço a partir do segundo perdia o nome da pessoa —
    # medido: 38% dos chunks do corpus não continham o docente a que se
    # referiam. Recuperar um desses chunks devolvia texto sobre alguém sem
    # dizer sobre quem, e o LLM ou omitia a atribuição ou a inventava.
    #
    # Manter inteiro resolve na origem e cabe folgado no bge-m3 (8192 tokens).
    # O splitter continua aqui para quando o corpus voltar a ter documentos
    # longos — as outras abas do SIGAA, do achado 05.
    perfis = [d for d in documentos if d.meta.get("content_type") == "docente_perfil"]
    longos = [d for d in documentos if d.meta.get("content_type") != "docente_perfil"]

    if not longos:
        log(f"[CHUNKING] {len(perfis)} perfis mantidos inteiros; nada a fatiar.")
        return perfis

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

    resultado = splitter.run(documents=longos)
    chunks    = resultado["documents"]

    log(f"[CHUNKING] {len(perfis)} perfis inteiros + {len(longos)} docs longos → {len(chunks)} chunks.")
    return perfis + chunks

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