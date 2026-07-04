# 🤖 Agente RAG Federado UFRRJ

Agente de Inteligência Artificial baseado em RAG (Retrieval-Augmented Generation) integrado a uma rede social federada (Mastodon/ActivityPub).
Projeto de Iniciação Científica — Ciência da Computação, UFRRJ.

**Autor:** Raul Nascimento
**Orientador:** Marcel William Rocha da Silva
**Stack:** Python · Haystack v2 · ChromaDB · SQLite · Docker · Sentence-Transformers · Ollama · Mastodon.py

---

## 🏗️ Estrutura do Projeto

### Módulo 1 — Pipeline ETL (✅ concluído e validado)

| Arquivo | Status | Descrição |
|---|---|---|
| `parte1_scraping_sigaa.py` | ✅ Concluído | Extração dos cartões de serviço da home pública do SIGAA. |
| `parte2_scraping_docentes.py` | ✅ Concluído | Scraping em 3 níveis (departamentos → SIAPE via POST JSF → perfis, assíncrono). |
| `parte3_chunking.py` | ✅ Concluído | Segmentação estrutural por sentenças (Haystack + NLTK). |
| `parte4_embedding.py` | ✅ Concluído | Vetorização local, modelo selecionável por variável de ambiente. |
| `parte5_carga.py` | ✅ Concluído | Carga e deduplicação no ChromaDB, com validação de isolamento por `instancia_dona`. |
| `parte6_pipeline.py` | ✅ Concluído | Orquestrador — executa o ETL completo em sequência única. |
| Ingestão em SQLite | 🟡 Parcial | Dados estruturados (ex.: listagem de docentes por departamento) já são persistidos como JSON no SQLite durante o ETL. **Ainda não consumidos pelo agente de inferência.** |

### Módulo 2 — Motor de Inferência (🟡 em desenvolvimento)

| Componente | Status | Descrição |
|---|---|---|
| `teste_llm.py` | ✅ Funcional | RAG simples: busca vetorial no ChromaDB + geração via LLM local (Ollama), com filtro obrigatório por `instancia_dona`. Testado com resultados qualitativamente coerentes. |
| Roteamento via Tool Calling | 🔴 Em desenvolvimento | LLM decidindo autonomamente entre consulta ao ChromaDB (semântica) ou ao SQLite (determinística), via Haystack 2.x. Ainda não validado sistematicamente. |
| Integração com Mastodon | ⚪ Não iniciado | Event Listener de menções via `Mastodon.py`, publicação de resposta em thread. |

### Infraestrutura (✅ concluída)

| Arquivo | Status | Descrição |
|---|---|---|
| `Dockerfile` | ✅ Concluído | Imagem da aplicação com dependências e cache do modelo de embedding. |
| `docker-compose.yml` | ✅ Concluído | Orquestra `chromadb`, `etl` (efêmero) e `agente` (interativo, profile). |
| `rag.sh` | ✅ Concluído | Script wrapper: `build`, `etl`, `agente`, `chroma`, `logs`, `status`, `limpar`. |
| `.env` | ✅ Concluído | Externaliza modelo de embedding, dimensão vetorial e endereço do ChromaDB por ambiente. |

---

## 🧠 Modelos e Hardware (Tiers de Operação)

| | Ambiente de Desenvolvimento | Ambiente de Produção (Servidor DCC) |
|---|---|---|
| **Embedding** | `paraphrase-multilingual-MiniLM-L12-v2` (dim=384) | `BAAI/bge-m3` (dim=1024) |
| **LLM** | `phi3:mini` ou `mistral` | `mistral` / `qwen3` (via Ollama no servidor) |
| **GPU** | Não obrigatória (CPU) | RTX 5070, acesso via SSH tunnel ou Docker com passthrough NVIDIA |
| **ChromaDB** | Local (`./chroma_db`) | Remoto, container `chromadb` no Docker Compose |
| **Objetivo** | Iteração rápida, custo mínimo de RAM | Precisão semântica máxima, produção institucional |

O modelo e a dimensão do embedding são lidos do `.env` via `os.getenv()` — nenhuma edição de código é necessária para trocar de ambiente.

---

## 🚀 Como Executar

### Local (desenvolvimento, sem Docker)

```bash
pip install -r requirements.txt

# sobe o Ollama localmente (terminal separado)
ollama serve
ollama pull phi3:mini

# roda o pipeline ETL completo (scraping → chunking → embedding → carga)
python parte5_carga.py

# interage com o agente
python teste_llm.py
```

### Servidor institucional (Docker + GPU)

**Pré-requisitos:** Docker, Docker Compose (v1 `docker-compose` ou v2 `docker compose`, conforme a versão instalada no servidor) e, opcionalmente, NVIDIA Container Toolkit.

```bash
chmod +x rag.sh

# constrói a imagem (1ª vez ou após atualizar dependências)
./rag.sh build

# sobe o ChromaDB e roda o pipeline ETL completo
./rag.sh etl

# abre o agente interativo
./rag.sh agente
```

**Acesso ao Ollama do servidor a partir de uma máquina local** (sem replicar o modelo localmente):

```bash
ssh usuario@www.dcc.ufrrj.br -L9999:invaders:11434
```

Com o túnel aberto, aponte o `OllamaGenerator` para `http://localhost:9999`.

### Comandos úteis do `rag.sh`

```bash
./rag.sh status   # mostra containers rodando
./rag.sh logs     # acompanha logs do ETL em tempo real
./rag.sh limpar   # remove containers e volumes (apaga o banco — pede confirmação)
```

---

## 🔒 Isolamento e Governança

- O campo `instancia_dona` é **hardcoded em cada módulo**, nunca aceito como parâmetro de execução — decisão deliberada de segurança (ADR: isolamento por construção).
- Na PoC atual, o isolamento é **lógico**: uma única base ChromaDB com filtro obrigatório por metadado em toda consulta.
- Em produção multi-instância, o isolamento planejado é **físico**: Document Stores inteiramente separados por instância (ADR-001).
- Toda carga no ChromaDB passa por validação automática de: presença de metadados obrigatórios (`instancia_dona`, `source_url`, `scraped_at`), dimensão correta do embedding, e ausência de vazamento entre instâncias.

---

## 📌 Limitações Conhecidas

- Provisionamento de uma nova instância exige duplicar os módulos ETL manualmente e trocar a constante `INSTANCIA` — não escalável para múltiplos setores sem refatoração futura.
- O chunking por sentenças usa o tokenizador padrão do NLTK (inglês) na ausência de um arquivo de abreviações em português, podendo segmentar incorretamente em abreviações como "Prof.", "Dr.", "Av.".
- O tamanho de chunk (5 sentenças, overlap 1) foi definido por heurística, sem validação empírica sistemática — candidato a experimento futuro.

---

## 🗺️ Próximos Passos

1. Validar o consumo dos dados do SQLite pelo agente de inferência.
2. Finalizar e testar sistematicamente o roteamento via Tool Calling (ChromaDB vs. SQLite).
3. Implementar o Event Listener de menções no Mastodon.
4. Migrar o isolamento lógico (metadado) para isolamento físico (stores separados) em produção.
