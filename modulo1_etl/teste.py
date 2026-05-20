# inspecionar_db.py
import chromadb

client     = chromadb.PersistentClient(path="./chroma_db")
colecao    = client.get_collection("rag_sigaa")

print(f"Total de documentos: {colecao.count()}")

# Mostra os primeiros 5
amostra = colecao.get(limit=5, include=["documents", "metadatas"])
for i, (doc, meta) in enumerate(zip(amostra["documents"], amostra["metadatas"]), 1):
    print(f"\n[{i}] {meta.get('nome_docente', meta.get('titulo', 'sem nome'))}")
    print(f"     Instancia: {meta.get('instancia_dona')}")
    print(f"     Fonte: {meta.get('source_url')}")
    print(f"     Texto: {doc[:120]}...")