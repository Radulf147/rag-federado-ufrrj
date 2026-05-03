# RAG Federado UFRRJ

Agente de IA baseado em RAG integrado a uma rede social federada (Mastodon/ActivityPub).  
Projeto de Iniciação Científica — Ciência da Computação, UFRRJ.

**Autor:** Raul Nascimento  
**Stack:** Python · Haystack v2 · ChromaDB · Mastodon.py

---

## Requisitos

- Python 3.12 ou superior
- pip atualizado: `python -m pip install --upgrade pip`

### Instalação das dependências

```bash
# Core do pipeline ETL
pip install haystack-ai
pip install httpx
pip install beautifulsoup4
pip install lxml

# Embedding
pip install sentence-transformers

# Chunking por sentença
pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Banco vetorial
pip install chromadb
pip install chroma-haystack

# Módulo 2 — fase futura
# pip install Mastodon.py
```

Ou instale tudo de uma vez:

```bash
pip install -r requirements.txt
```

---

## Estrutura

| Módulo | Status | Descrição |
|--------|--------|-----------|
| `modulo1_etl/parte1_scraping.py` | ✅ Concluída | Scraping do SIGAA público |
| `modulo1_etl/parte2_inferencia.py` | ✅ Concluída | Scraping de docentes + chunking (async) |
| `modulo1_etl/parte3_embedding.py` | ✅ Concluída | Vetorização com SentenceTransformers |
| `modulo1_etl/parte4_carga.py` | ✅ Concluída | Carga no ChromaDB com teste de busca |
| `modulo2_inferencia/` | ⏳ Fase futura | Listener Mastodon + pipeline de resposta |

---

## Como executar

Cada parte pode ser executada de forma standalone ou integrada.  
Para rodar o pipeline ETL completo (Partes 1 a 4):

```bash
python modulo1_etl/parte4_carga.py
```

Para rodar cada parte individualmente:

```bash
python modulo1_etl/parte1_scraping.py
python modulo1_etl/parte2_inferencia.py
python modulo1_etl/parte3_embedding.py
python modulo1_etl/parte4_carga.py
```

Logs gerados automaticamente na pasta `logs/` a cada execução.  
O banco vetorial é persistido localmente em `chroma_db/`.

---

## Decisões Arquiteturais

- **ADR-001:** Document Stores separados por instância em produção (isolamento físico)
- **ADR-002:** Embedding multilingual em português (`paraphrase-multilingual-MiniLM-L12-v2`)
- **ADR-003:** Haystack v2 como framework RAG principal
