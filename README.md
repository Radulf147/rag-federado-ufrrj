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
| `parte4_embedding.py` | ✅ Concluído | Vetorização local com suporte a múltiplos modelos (MiniLM ou BGE-M3). |
| `parte5_carga.py` | ✅ Concluído | Consolidação e carga persistente no ChromaDB. |
| `teste_llm.py` | ✅ Concluído | Interface de inferência do Agente (Perguntas & Respostas). |
| `rag.sh` | ✅ Concluído | Script wrapper para orquestração simplificada do Docker. |

---

## 🧠 Modelos e Hardware (Tiers de Operação)

A arquitetura suporta duas configurações de *embedding* e inferência dependendo do hardware disponível:

* **Ambiente de Desenvolvimento (Local/Sem GPU dedicada):**
  * **Embedding:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Dimensão: 384).
  * **LLM:** `phi3:mini`.
  * *Objetivo:* Modelos leves e rápidos, ideais para testar a pipeline de extração e o código em computadores pessoais (CPUs), priorizando a economia de RAM.
  
* **Ambiente de Produção (Servidor DCC com RTX 5070 16GB VRAM):**
  * **Embedding:** `BAAI/bge-m3` (Dimensão: 1024).
  * **LLM:** `qwen3`.
  * *Objetivo:* Modelos de ponta com suporte multilíngue robusto e maior janela de contexto. O setup de produção utiliza a GPU dedicada do laboratório para garantir máxima precisão semântica na recuperação de vetores e respostas mais profundas.

---

## 🚀 Como Executar (Ambiente Recomendado: Docker)

O projeto foi empacotado via Docker para garantir que as dependências do sistema e o banco de vetores sejam gerenciados sem conflitos ou complexidades.

**Pré-requisitos:** `Docker` e `Docker Compose` instalados.

Para operar o projeto, utilize o script de conveniência `rag.sh` fornecido na raiz:

```bash
# 1. Construa a imagem da aplicação (necessário na 1ª vez ou se atualizar pacotes Python)
./rag.sh build

# 2. Rode o pipeline de dados completo (Sobe o banco ChromaDB + Scraping → Chunking → Embedding → Carga)
./rag.sh etl

# 3. Interaja com o Agente (Abre um terminal interativo com o RAG e o LLM)
./rag.sh agente