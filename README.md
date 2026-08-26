# 🤖 Agente RAG Federado UFRRJ

Agente de Inteligência Artificial baseado em RAG (Retrieval-Augmented Generation) integrado a uma rede social federada (Mastodon/ActivityPub).
Projeto de Iniciação Científica — Ciência da Computação, UFRRJ.

**Autor:** Raul Nascimento
**Orientador:** Marcel William Rocha da Silva
**Stack:** Python · Haystack v2 · ChromaDB · SQLite · Docker · Sentence-Transformers · Ollama · Mastodon.py

---

## 🏗️ Estrutura do Projeto

```
raiz/
├── config.py                          # variáveis de ambiente centralizadas (parcial — ver Limitações)
├── modulo1_etl/                       # pipeline de coleta e carga (intocado no refactor de Set/2026)
│   ├── parte1_scraping_sigaa.py
│   ├── parte2_scraping_docentes.py
│   ├── parte3_chunking.py
│   ├── parte4_embedding.py
│   ├── parte5_carga.py                # também orquestra o pipeline completo (ver nota abaixo)
│   └── db_manager.py
├── modulo2_inferencia/                # motor de inferência (extraído de teste_llm.py)
│   ├── llm_setup.py                   # wiring: ChromaDocumentStore, embedder, retriever, OllamaChatGenerator
│   ├── tools.py                       # schema + implementação das tools (busca vetorial / SQLite)
│   └── agent.py                       # loop de decisão (roteamento via Tool Calling)
└── interfaces/                        # camadas de entrada, independentes do motor
    └── cli.py                         # REPL de terminal (antigo loop de teste_llm.py)
```

> ⚠️ **Nota sobre nomenclatura:** os arquivos `parteN` do Módulo 1 documentam a ordem cronológica de criação, não a responsabilidade de cada um. Uma renomeação por responsabilidade (ex.: `scraper_docentes.py`, `chunking.py`) está prevista, mas ainda não foi feita — ver Próximos Passos.

### Módulo 1 — Pipeline ETL (✅ concluído e validado)

| Arquivo | Status | Descrição |
|---|---|---|
| `parte1_scraping_sigaa.py` | ✅ Concluído | Extração dos cartões de serviço da home pública do SIGAA. |
| `parte2_scraping_docentes.py` | ✅ Concluído | Scraping em 3 níveis (departamentos → SIAPE via POST JSF → perfis, assíncrono). |
| `parte3_chunking.py` | ✅ Concluído | Segmentação estrutural por sentenças (Haystack + NLTK). |
| `parte4_embedding.py` | ✅ Concluído | Vetorização local, modelo selecionável por variável de ambiente. |
| `parte5_carga.py` | ✅ Concluído | Carga e deduplicação no ChromaDB, com validação de isolamento por `instancia_dona`. Seu bloco `__main__` também orquestra o ETL completo (scrape → chunk → embed → validar → carregar). |
| Ingestão em SQLite | 🟡 Parcial | Dados estruturados (ex.: listagem de docentes por departamento) já são persistidos como JSON no SQLite durante o ETL. |

> 📌 Não existe um `parte6_pipeline.py` separado — a orquestração do pipeline completo vive hoje dentro de `parte5_carga.py`. Extrair essa responsabilidade para um `pipeline.py` próprio é um passo futuro (ver Próximos Passos).

### Módulo 2 — Motor de Inferência (🟡 em desenvolvimento)

| Componente | Status | Descrição |
|---|---|---|
| `modulo2_inferencia/agent.py` | ✅ Funcional | Loop de decisão: recebe uma pergunta, roteia via Tool Calling (Haystack 2.x) entre busca vetorial no ChromaDB (semântica) e consulta ao SQLite (determinística), e devolve a resposta. Erros de tool não reconhecida agora retornam mensagem explícita ao LLM em vez de string vazia. |
| `modulo2_inferencia/tools.py` | ✅ Funcional | Schema e implementação das duas tools, recebendo embedder/retriever por parâmetro (não mais globais de módulo) — testável isoladamente sem subir o Ollama. |
| `modulo2_inferencia/llm_setup.py` | ✅ Funcional | Wiring dos componentes Haystack/Ollama. Conecta ao ChromaDB de forma independente do Módulo 1 (ver nota abaixo). |
| `interfaces/cli.py` | ✅ Funcional | REPL de terminal, separado do motor de inferência — ponto de entrada via `python -m interfaces.cli`. |
| Roteamento via Tool Calling | 🟡 Implementado, não validado sistematicamente | Lógica já codada e funcional; faltam testes repetidos e tratamento mais robusto de casos de borda. |
| Integração com Mastodon | ⚪ Não iniciado | Event Listener de menções via `Mastodon.py`, publicação de resposta em thread. Vai consumir o mesmo `agent.py` usado pela CLI, sem duplicar o loop de roteamento. |

> ℹ️ `teste_llm.py` foi descontinuado. Ele concentrava wiring, schema das tools, implementação das tools, o loop de roteamento e a interface CLI num único arquivo — o que impedia testes unitários e forçaria duplicação de lógica quando o listener do Mastodon fosse implementado. Essas responsabilidades foram separadas conforme a estrutura acima.
>
> `llm_setup.py` duplica (~15 linhas) a lógica de conexão ao ChromaDB em vez de importar `conectar_store` de `parte5_carga.py`, porque esse arquivo dispara `logging.basicConfig` e cria a pasta `./logs/` como efeito colateral só de ser importado. Há um `TODO` no código apontando para unificar isso quando esse efeito colateral for removido do Módulo 1.

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
python -m modulo1_etl.parte5_carga

# interage com o agente (a partir da raiz do projeto)
python -m interfaces.cli
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

Com o túnel aberto, aponte o `OllamaGenerator` (em `modulo2_inferencia/llm_setup.py`) para `http://localhost:9999`.

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

- Provisionamento de uma nova instância exige duplicar os módulos ETL manualmente e trocar a constante `INSTANCIA` — não escalável para múltiplos setores sem refatoração futura. A constante está repetida em `parte1`, `parte2`, `parte4` e `parte5`, o que torna esse refactor mais arriscado enquanto não for centralizada.
- `config.py` centraliza apenas as variáveis de ambiente usadas pelo Módulo 2 até o momento — o Módulo 1 ainda faz `os.getenv()` de forma espalhada por arquivo (`CHROMA_HOST`, `EMBEDDING_DIM`, `MODELO_EMBEDDING`, `MODELO_LLM`, `OLLAMA_HOST`, `DB_PATH`).
- Logging no Módulo 1 hoje coexiste em três estilos diferentes (boilerplate repetido em `parte2`/`parte3`/`parte4`, `logging.basicConfig` global em `parte5`, e `logging.getLogger(__name__)` sem handler em `db_manager.py`).
- `modulo1_etl/` não é um pacote Python formal (sem `__init__.py`) — os imports funcionam hoje por resolução de namespace package do Python 3, mas isso é sensível à estrutura de diretório de trabalho.
- O chunking por sentenças usa o tokenizador padrão do NLTK (inglês) na ausência de um arquivo de abreviações em português, podendo segmentar incorretamente em abreviações como "Prof.", "Dr.", "Av.".
- O tamanho de chunk (5 sentenças, overlap 1) foi definido por heurística, sem validação empírica sistemática — candidato a experimento futuro.

---

## 🗺️ Próximos Passos

1. Validar sistematicamente o roteamento via Tool Calling (ChromaDB vs. SQLite) já implementado em `agent.py` — testes repetidos e tratamento de casos de borda.
2. Implementar o Event Listener de menções no Mastodon em `interfaces/mastodon_listener.py`, reaproveitando o `agent.py` já existente.
3. Extrair um `pipeline.py` próprio de dentro de `parte5_carga.py`, separando orquestração de carga.
4. Centralizar `INSTANCIA` e as demais variáveis de ambiente do Módulo 1 em `config.py`.
5. Renomear os arquivos `parteN` do Módulo 1 por responsabilidade (ex.: `scraper_docentes.py`, `chunking.py`) em vez de ordem cronológica.
6. Unificar os três estilos de logging do Módulo 1 num `shared/logging_config.py` comum.
7. Migrar o isolamento lógico (metadado) para isolamento físico (stores separados) em produção.
