# Arquitetura e operação — Agente RAG para dados institucionais da UFRRJ

> **O que este documento é.** A descrição da arquitetura, das decisões de
> projeto e de como rodar o sistema.
>
> **O que ele não é.** Não é o relatório de resultados. Nenhum número de
> avaliação é defendido aqui; quando um aparece, é para justificar uma decisão
> de engenharia, e a apuração está em [`relatorio_ic.md`](relatorio_ic.md).
>
> **Escopo.** Só a aba de docentes do SIGAA. É o recorte que foi medido.
>
> Retrato do repositório em 12 set 2026, commit `f1501ff`.

---

## 1. O problema

O SIGAA concentra dados que a comunidade acadêmica precisa, mas a navegação é
fragmentada: achar em qual departamento um docente está, quem dá aula de certo
tema, ou a lista completa de um departamento exige saber de antemão por onde
entrar. A informação existe e não está acessível.

A proposta é um agente que receba a pergunta em linguagem natural e responda a
partir dos dados do próprio SIGAA — com a exigência de **não inventar**. Num
sistema que fala sobre pessoas reais de uma instituição real, uma resposta
inventada é pior que nenhuma resposta.

O destino é uma rede social federada (Mastodon / ActivityPub), onde o agente
seria acionado por menção. **Essa parte não está implementada** — ver §2.3.

---

## 2. Arquitetura

Três camadas, desacopladas: uma prepara os dados, uma responde, uma entrega.

### 2.1. Módulo 1 — ETL (`modulo1_etl/`)

Mantém a base sincronizada com o SIGAA. O scraping é assíncrono
(`httpx` + `asyncio`), o parsing tolerante ao HTML antigo do SIGAA
(`BeautifulSoup4`).

| etapa | arquivo | o que faz |
|---|---|---|
| 2 | `parte2_scraping_docentes.py` | varre os departamentos, depois cada perfil individual |
| 3 | `parte3_chunking.py` | divide em blocos de 5 sentenças com 1 de sobreposição (`CHUNK_SENTENCES = 5`, `CHUNK_OVERLAP = 1`) |
| 4 | `parte4_embedding.py` | vetoriza com `BAAI/bge-m3`, 1024 dimensões |
| 5 | `parte5_carga.py` | valida e carrega no ChromaDB; também orquestra o pipeline inteiro |
| — | `db_manager.py` | escreve o SQLite |
| — | `deduplicacao.py` | remove perfis repetidos antes da carga |

**`parte1_scraping_sigaa.py` está desligada de propósito**
([`parte5_carga.py:181`](../modulo1_etl/parte5_carga.py#L181)). Ela é a
varredura ampla do SIGAA; ligada, o vetor store passa a conter material que a
base estruturada não enxerga, e a comparação entre os pipelines deixa de ser
entre abordagens e passa a ser entre corpora diferentes. Reativar quando a
varredura ampla entrar em escopo.

**Armazenamento híbrido.** Os mesmos dados vão para dois lugares, com papéis
distintos:

- **ChromaDB (vetorial)** — os textos descritivos (formação, áreas de atuação,
  lattes). Otimizado para similaridade semântica: responde "quem trabalha com
  tema X" sem que a palavra X apareça literalmente.
- **SQLite (estruturado)** — tabela genérica `entidades_sigaa`, com
  `tipo_entidade` indexado e `dados_brutos` em JSON (*schema-less*). Responde
  "quantos" e "quais todos" com precisão exata e custo desprezível.

Uma nota de honestidade sobre o SQLite: a escolha original foi motivada pelo
suporte nativo a `json_extract`, mas **o código não usa mais isso**. Hoje só o
filtro por `tipo_entidade` fica no SQL (é indexado) e o casamento do campo é
feito em Python, com normalização de acento e caixa
([`db_manager.py:97-113`](../modulo1_etl/db_manager.py#L97-L113)). A escolha
continua defensável — banco embutido, sem servidor, sem schema —, mas não pela
razão que foi escrita antes.

### 2.2. Módulo 2 — inferência (`modulo2_inferencia/`)

Aqui o LLM não é só um gerador de texto: é quem decide onde buscar. Via
**tool calling**, o modelo lê a pergunta e escolhe a ferramenta.

| arquivo | papel |
|---|---|
| `llm_setup.py` | monta as peças: `ChromaDocumentStore`, embedder, retriever, `OllamaChatGenerator` |
| `tools.py` | schema e implementação das três ferramentas |
| `agent.py` | o laço de decisão, no máximo `MAX_RODADAS_TOOL = 4` rodadas |
| `pipelines.py` | as três abordagens comparadas, isoladas uma da outra |

As três ferramentas registradas hoje:

| ferramenta | vai em | responde |
|---|---|---|
| `buscar_docentes_por_departamento` | SQLite | "quem são todos do departamento X", contagens |
| `buscar_docente_por_nome` | SQLite | "onde o professor Y está lotado" |
| `busca_vetorial_sigaa` | ChromaDB | "quem trabalha com tema X" |

O casamento dos argumentos nas duas ferramentas de SQLite é por **substring,
ignorando caixa e acento** ([`db_manager.py:59`](../modulo1_etl/db_manager.py#L59)).
Isso não é detalhe: o `LIKE` do SQLite só é *case-insensitive* para ASCII, e
`LIKE '%Ciência da Computação%'` devolvia 0 resultados onde
`'%CIÊNCIA DA COMPUTAÇÃO%'` devolvia 6 — o SIGAA grava em caixa alta, o LLM
escreve o argumento em caixa mista com acento, e a tool falhava em quase toda
pergunta acentuada. O agente então respondia, com honestidade, que não havia
docentes naquele departamento.

Duas decisões de resposta que valem registro, porque ambas evitam a resposta
errada convincente:

- **Ambiguidade é relatada, não resolvida.** "Silva" casa com dezenas de
  docentes; a ferramenta devolve o total exato e lista até 15, em vez de
  escolher o primeiro.
- **Base vazia ≠ pessoa ausente.** Antes de dizer "não encontrei", a ferramenta
  consulta `total_de_entidades`. Um banco vazio apresentado como ausência da
  pessoa é a mentira mais convincente que este sistema sabe produzir.

`agent.py` não sabe nada sobre CLI nem sobre rede: recebe a pergunta, devolve a
resposta. É o que permite trocar a interface sem duplicar o laço de decisão.

O `pipelines.py` guarda as três abordagens que a avaliação compara —
`1-vetorial`, `2-estruturado` e `3-agente`. Só a terceira é o sistema; as outras
duas existem como termo de comparação. A do meio é **sem LLM por definição**, o
que a obriga a sair da linguagem natural por um caminho determinístico: ela casa
a pergunta contra os nomes reais de departamento com `rapidfuzz`
(`fuzz.partial_ratio`, `LIMIAR_FUZZY = 70`,
[`pipelines.py:95`](../modulo2_inferencia/pipelines.py#L95)). É deliberadamente
burra — a graça é ver onde o caminho barato empata com o agente e onde quebra.

**Parâmetros em uso** (`.env.example` traz os valores e o porquê de cada um):

| variável | valor | nota |
|---|---|---|
| `MODELO_LLM` | `qwen2.5:32b-instruct-q4_K_M` | escolhido por medição, não por estimativa |
| `MODELO_EMBEDDING` | `BAAI/bge-m3` | `EMBEDDING_DIM=1024` |
| `TOP_K` | 10 | |
| `LIMIAR_DISTANCIA` | 1.24 | **distância**, não similaridade: o filtro é `score <= limiar`, menor é mais parecido. Calibrado em [`calibracao_limiar.md`](calibracao_limiar.md); vazio desliga o filtro |
| `NUM_CTX` | 8192 | o padrão do Ollama (4096) é apertado para `TOP_K=10` |
| `MAX_RODADAS_TOOL` | 4 | |

> ⚠️ O `config.py` traz outros *defaults* (`mistral`,
> `paraphrase-multilingual-MiniLM-L12-v2`). Eles **não** são o que roda: o
> `.env` sobrescreve. Rodar sem `.env` não dá erro — vetoriza com outro modelo,
> em outra dimensão, em silêncio.

### 2.3. Camada de entrega (`interfaces/`)

| interface | estado |
|---|---|
| `cli.py` — REPL de terminal | funciona |
| `rede/` — rede social **simulada**, local | funciona (`http://localhost:5000`) |
| Mastodon / ActivityPub | **não existe** |

Este é o ponto onde a documentação anterior afirmava o que o projeto não faz.
O pacote `interfaces/rede/` diz na própria docstring:

> *"NÃO é o Módulo 3, e o nome do pacote evita de propósito a palavra
> 'federação'. Aqui não há ActivityPub, não há instância remota e não há
> federação nenhuma: há uma imitação local do formato de uma rede social."*

O objetivo dela é metodológico: fazer o agente receber a pergunta do jeito que
ele a receberia em produção — um post com menção, dentro de uma thread — em vez
de um prompt limpo digitado num terminal. **A federação de verdade é escopo do
TCC.**

---

## 3. Decisões de projeto

### 3.1. Por que o banco vetorial sozinho não basta

O RAG clássico busca os *K* trechos mais próximos da pergunta. Isso é
inadequado para **contagens e listagens exaustivas**, por construção: "liste
todos os professores do Departamento de Computação" devolve no máximo `top_k`
trechos, e não o departamento inteiro. Aumentar `top_k` para cobrir todo mundo
esgota a janela de contexto — latência, VRAM, e o *lost in the middle*, em que
o modelo ignora o que está no meio do prompt.

O acoplamento do SQLite resolve: `COUNT` e filtragem com precisão exata, e ao
LLM chega só o resultado consolidado.

Isso deixou de ser argumento teórico. Nas 16 perguntas objetivas da bateria, o
RAG clássico acerta **25,0%**, o acesso determinístico ao banco **81,3%**, e o
agente que escolhe entre os dois **95,8%**. Nos subtipos que dependem de
exaustividade — contagem e listagem — o RAG clássico faz **0%**. Apuração e
ressalvas em [`comparacao_abordagens.md`](comparacao_abordagens.md).

### 3.2. Por que Regex saiu e entrou Tool Calling

No MVP, o roteamento entre busca vetorial e busca exata era uma expressão
regular. Ela existiu de verdade, em `modulo1_etl/teste_llm.py:79`, commit
`5221190` (16 jun 2026):

```python
padroes_exaustivos = r"(todos os|lista de|quais s[aã]o os) professores\s+(do|da|de)?\s*(.*)"
match_intencao = re.search(padroes_exaustivos, pergunta_limpa)
```

Funciona até a primeira pergunta que ninguém previu. A linguagem natural é
variável demais para ser enumerada à mão, e cada nova aba do SIGAA
multiplicaria os padrões — código frágil e acoplado.

O tool calling entrou no commit `0208ef9` (25 jun 2026): registram-se funções
Python como ferramentas e delega-se a interpretação da intenção ao próprio LLM.
Ele avalia a semântica e decide qual chamar e com que argumentos. O ganho é de
escalabilidade: uma aba nova do SIGAA custa **uma ferramenta a mais**, não uma
reescrita da lógica de controle.

O custo dessa decisão também está medido, e não é zero — o agente perde do RAG
puro na precisão das perguntas interpretativas. Ver `relatorio_ic.md` §6.3.

---

## 4. Stack

| tecnologia | papel | por quê |
|---|---|---|
| **Haystack 2.x** | orquestração do pipeline de IA | arquitetura por componentes; suporte a tool calling |
| **ChromaDB** | banco vetorial | persistência local, sem API proprietária |
| **SQLite** | document store genérico | embutido, sem servidor, JSON schema-less |
| **Ollama** | inferência local do LLM | roda modelo grande sem custo de API e sem mandar dado da instituição para fora |
| **BAAI/bge-m3** | embedding | multilíngue, 1024 dimensões |
| **rapidfuzz** | casamento aproximado de nome de departamento | dá ao pipeline `2-estruturado` um caminho sem LLM da pergunta até o parâmetro de query |
| **BeautifulSoup4** | parsing | tolerante ao HTML antigo do SIGAA |
| **httpx + asyncio** | extração concorrente | corta o tempo de scraping |
| **Docker Compose** | infra como código | reprodutibilidade; isola ChromaDB, ETL, agente e rede |
| **pytest** | suíte de testes | 181 testes |

---

## 5. Operação

Tudo passa pelo `rag.sh`. No Windows, `rag.cmd` é um invólucro que acha o Git
Bash e delega ao mesmo script — os comandos são idênticos.

**Pré-requisitos:** Docker, plugin Docker Compose V2, e um `.env` preenchido
(`cp .env.example .env`, depois preencher `DCC_USUARIO`).

### 5.1. O túnel não é opcional

O Ollama **não roda na sua máquina**: roda na máquina da faculdade, alcançada
por túnel SSH. Sem o túnel, tudo que precisa do LLM falha — o agente sobe e não
responde, e os pipelines 1 e 3 da bateria falham inteiros.

```bash
./rag.sh tunel up       # sobe   (também: status | down)
```

Os comandos `agente`, `comparar` e `rede` já tentam levantar o túnel sozinhos, e
**não abortam** se ele falhar: avisam e seguem. A mensagem de erro do túnel é
mais informativa que um timeout lá dentro.

### 5.2. Ciclo normal

```bash
./rag.sh build          # constrói a imagem
./rag.sh etl            # sobe o ChromaDB e roda o ETL completo
./rag.sh agente         # abre o agente interativo no terminal
```

### 5.3. Comandos

| comando | o que faz |
|---|---|
| `build` | constrói a imagem |
| `etl` | sobe o ChromaDB e roda o pipeline ETL completo |
| `agente` | sobe o ChromaDB, levanta o túnel e abre o REPL |
| `comparar` | roda a bateria dos 3 pipelines nas 30 perguntas |
| `testes` | **reconstrói a imagem** e roda o pytest |
| `tunel up\|status\|down` | gerencia o túnel SSH até o Ollama |
| `chroma` | sobe só o ChromaDB, em background |
| `logs` | segue os logs do ETL |
| `status` | mostra os containers rodando |
| `limpar` | derruba tudo e **APAGA** o banco e os volumes (pede confirmação) |
| `rede` | sobe a rede simulada → `http://localhost:5000` |
| `rede-parar` | derruba a página e o bot; os posts ficam em `dados/rede.db` |
| `rede-logs` | segue os logs do worker do bot |
| `semear` | **APAGA** os posts da rede simulada e recria o cenário (pede confirmação) |

### 5.4. Duas armadilhas que já custaram caro

**O `testes` reconstrói a imagem, e isso não é zelo.** `testes/` e o código
**não são volumes montados** — vêm do `COPY` da imagem. Rodar `pytest` sem
rebuild executa a versão anterior à edição e devolve verde de código obsoleto.
Aconteceu em 5 set 2026. Por isso o comando imprime o id da imagem: a saída diz
de onde os testes vieram.

**O `comparar` sobrescreve, se você deixar.** Por padrão ele escreve em
`docs/avaliacao_fase3.md` e **acrescenta** ao `docs/avaliacao_fase3.jsonl`.
Rodar duas baterias diferentes sem trocar os caminhos mistura execuções no
mesmo registro. Use `--saida` e `--registro` para separar:

```bash
docker compose --profile agente run --rm agente python -m interfaces.comparar \
    --saida docs/minha_bateria.md --registro docs/minha_bateria.jsonl
```

O comando avisa onde vai escrever e reclama se o registro já tem execuções.

---

## 6. Onde está cada coisa

| quero | vou em |
|---|---|
| os resultados, o que fecha critério e o que não fecha | [`relatorio_ic.md`](relatorio_ic.md) |
| a comparação das três abordagens nas 30 perguntas | [`comparacao_abordagens.md`](comparacao_abordagens.md) |
| as previsões, commitadas antes de rodar | [`pre_registro_comparacao_30.md`](pre_registro_comparacao_30.md) |
| como cada métrica é apurada, e o que ela não enxerga | [`criterios_avaliacao.md`](criterios_avaliacao.md) |
| os defeitos conhecidos e não corrigidos | [`backlog_avaliacao.md`](backlog_avaliacao.md) |
| a calibração do limiar de distância | [`calibracao_limiar.md`](calibracao_limiar.md) |
| a qualidade dos perfis coletados | [`auditoria_perfis.md`](auditoria_perfis.md) |
| convenções, armadilhas e estado de trabalho | [`../CLAUDE.md`](../CLAUDE.md) |
