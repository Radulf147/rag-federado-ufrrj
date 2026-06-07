# 🤖 Agente RAG Federado UFRRJ

Agente de Inteligência Artificial baseado em RAG (Retrieval-Augmented Generation) integrado a uma rede social federada (Mastodon/ActivityPub). 
Projeto de Iniciação Científica — Ciência da Computação, UFRRJ.

**Autor:** Raul Nascimento  
**Stack:** Python · Haystack v2 · ChromaDB · Docker · Sentence-Transformers · Mastodon.py

---

## 🏗️ Estrutura do Projeto

O pipeline de dados (ETL) e inferência é dividido modularmente:

| Módulo / Arquivo | Status | Descrição |
|------------------|--------|-----------|
| `parte1_scraping_sigaa.py` | ✅ Concluído | Extração e sanitização do portal público do SIGAA. |
| `parte2_scraping_docentes.py` | ✅ Concluído | Scraping em 3 níveis dos departamentos e perfis de docentes. |
| `parte3_chunking.py` | ✅ Concluído | Divisão estrutural de sentenças usando Haystack e NLTK. |
| `parte4_embedding.py` | ✅ Concluído | Vetorização local com o modelo multilingual `MiniLM-L12-v2`. |
| `parte5_carga.py` | ✅ Concluído | Consolidação e carga persistente no ChromaDB. |
| `teste_llm.py` | ✅ Concluído | Interface de inferência do Agente (Perguntas & Respostas). |
| `rag.sh` | ✅ Concluído | Script wrapper para orquestração simplificada do Docker. |

---

## 🚀 Como Executar (Ambiente Recomendado: Docker)

O projeto foi empacotado via Docker para garantir que dependências (como bancos vetoriais e caches de modelos de IA) sejam gerenciadas sem conflitos. 

**Pré-requisitos:** Docker e Docker Compose instalados na máquina host.

Para operar o projeto, utilize o script de conveniência `rag.sh` fornecido na raiz:

```bash
# 1. Construa a imagem da aplicação (necessário apenas na 1ª vez ou se mudar os requirements)
./rag.sh build

# 2. Rode o pipeline de dados completo (Sobe o banco ChromaDB + Scraping → Chunking → Embedding → Carga)
./rag.sh etl

# 3. Interaja com o Agente (Abre um terminal interativo com o RAG e o LLM)
./rag.sh agente