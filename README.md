# RAG Federado UFRRJ

Agente de IA baseado em RAG integrado a uma rede social federada (Mastodon/ActivityPub).
Projeto de Iniciação Científica — Ciência da Computação, UFRRJ.

**Autor:** Raul Nascimento  
**Stack:** Python · Haystack v2 · ChromaDB · Mastodon.py

## Estrutura

| Módulo | Status | Descrição |
|--------|--------|-----------|
| `modulo1_etl/parte1_scraping.py` | ✅ Concluída | Scraping do SIGAA público |
| `modulo1_etl/parte2_chunking.py` | 🔄 Em desenvolvimento | Chunking com Haystack |
| `modulo1_etl/parte3_embedding.py` | ⏳ Pendente | Vetorização com SentenceTransformers |
| `modulo1_etl/parte4_carga.py` | ⏳ Pendente | Carga no DocumentStore |
| `modulo2_inferencia/` | ⏳ Fase futura | Listener Mastodon + pipeline de resposta |

## Como executar

```bash
pip install -r requirements.txt
python modulo1_etl/parte1_scraping.py
```

## Decisões Arquiteturais

- **ADR-001:** Document Stores separados por instância em produção
- **ADR-002:** Embedding multilingual em português (paraphrase-multilingual-MiniLM-L12-v2)
- **ADR-003:** Haystack v2 como framework RAG principal