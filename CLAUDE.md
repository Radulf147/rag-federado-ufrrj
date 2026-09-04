| 10 | **O SQLite acumulava a cada recarga.** O Chroma era limpo, o SQLite não — a contagem de docentes dobraria, e é ela que responde "quantos docentes tem X". | ✅ `substituir=True` em `salvar_entidades` | ✅ recarregado — Chroma 1302 = SQLite 1302 || 09 | **A guarda do 06 rejeitava perfil legítimo e esparso** — 8 de 30 docentes reais descartados por não terem seção descritiva. | ✅ identidade pela listagem substitui a checagem de chaves | ✅ recarregado || 08 | **Corrida de sessão JSF no scraping concorrente** — cliente HTTP compartilhado entre 8 requisições disputava a sessão do SIGAA. Media 6 de 15 perfis com a pessoa errada e 47% de captura. | ✅ cliente isolado + conferência do nome contra a listagem + repescagem em série | ✅ recarregado — identidade **15 de 15** contra o SIGAA ao vivo (era 3 de 41) || 06 | SIAPE inválida devolvia a home do portal com HTTP 200 | ✅ feito | ✅ recarregado || 05 | Abas não coletadas + mapeamento assistido por LLM | ✅ campos grátis (Lattes, Sala, E-mail) coletados; ❌ outras abas a desenhar | ✅ Lattes **79,6%**, e-mail **92,9%**, sala **27,1%** || 04 | Quanto do corpus vazio é falha nossa vs. ausência real no SIGAA | ✅ `modulo1_etl/auditoria_perfis.py` | ▶ **próximo passo** — rodar agora que o pareamento id↔pessoa é confiável || 03 | Recuperação não discrimina | ✅ boilerplate `"não informada"` fora do texto indexado; ⚠️ limiar de distância pronto mas **desligado** | ✅ boilerplate recarregado; ▶ falta **calibrar o limiar** sobre o corpus novo || 02 | 38% dos chunks não continham o nome do docente; a tool semântica descartava os metadados | ✅ perfil não é mais fatiado + `busca_vetorial_sigaa` devolve nome, departamento e fonte | ✅ recarregado — 1 documento por docente, **0 sem nome** || 01 | Campo "Áreas de interesse" nunca capturado. Causa: o `<dt>` é o único com um `<span class="info">` aninhado, e `get_text(strip=True)` junta os nós **sem separador**. O espaço existe na página; sumia no parser. | ✅ prefixo normalizado (`CAMPOS_DO_PERFIL`) | ✅ recarregado — **653 de 1302** (era 0 de 704) |# Agente RAG Federado — UFRRJ

Guia principal do projeto para sessões de desenvolvimento (Claude Code).
Reflete o estado e as decisões atuais — leia antes de mexer em Docker,
seleção de modelo, ou no escopo do ETL.

## O que é o projeto

Projeto de Iniciação Científica (Ciência da Computação, UFRRJ). Um agente
RAG (Haystack v2 + ChromaDB + Ollama) que responde perguntas sobre dados
institucionais da UFRRJ, com objetivo final de integração a uma rede
social federada (Mastodon/ActivityPub). Orientador: Marcel William Rocha
da Silva.

## ▶ Estado atual (4 set 2026) — leia isto primeiro

A primeira bateria real de comparação dos três pipelines rodou e expôs sete
defeitos; um oitavo — o mais grave — apareceu em 4 set 2026 ao auditar os
perfis. Os sete primeiros estão documentados em dois artefatos publicados:

- **Diagnóstico:** https://claude.ai/code/artifact/0b5cee19-318b-4008-b29c-80c8c0d873dc
- **Plano de correção:** https://claude.ai/code/artifact/3a32e6dc-cc5b-4453-9408-af8d1a124413

A coluna de estado foi **dividida em duas** (Set/2026): antes havia só "Estado",
e escrever "aguarda recarga" num achado cujo **código ainda não foi corrigido**
transmitia que só faltava rodar o ETL. Faltava mais que isso. São dois eixos
independentes — o código já está corrigido? os dados já foram recarregados? —
e um achado só está encerrado quando as duas colunas estão verdes.

| # | Achado | Código | Dados |
|---|---|---|---|
| 01 | Campo "Áreas de interesse" nunca capturado — 0 de 704 perfis. Causa confirmada no HTML real em 4 set 2026: o `<dt>` desse campo é o único com um `<span class="info">` aninhado, e `get_text(strip=True)` junta os nós **sem separador**, produzindo `áreas de interesse(áreas…`. O espaço **existe** na página; some no parser. | ✅ feito — casamento por prefixo normalizado (`CAMPOS_DO_PERFIL`) | ❌ exige recarga |
| 02 | 38% dos chunks não continham o nome do docente; a tool semântica descartava os metadados | ✅ feito — perfil não é mais fatiado (`parte3_chunking`) e `busca_vetorial_sigaa` devolve nome, departamento e fonte | ❌ exige recarga |
| 03 | Recuperação não discrimina (os 10 primeiros para "IA" não continham IA) | ✅ boilerplate `"não informada"` fora do texto indexado; ⚠️ limiar de distância implementado mas **desligado** — calibrar depois da recarga | ❌ exige recarga |
| 04 | Quanto do corpus vazio é falha nossa vs. ausência real no SIGAA | ✅ `modulo1_etl/auditoria_perfis.py` | ✅ **respondido: 0% é falha nossa** — 192 campos com conteúdo real na página, 192 no store |
| 05 | Abas não coletadas + arquitetura de mapeamento assistido por LLM | ✅ **campos grátis feitos** — Currículo Lattes, Sala e E-mail entraram em `CAMPOS_DO_PERFIL`; ❌ as outras abas continuam a desenhar (fase própria) | ❌ exige recarga |
| 06 | SIAPE inválida devolvia a home do portal com HTTP 200 | ✅ feito | ❌ os perfis coletados antes da correção só são revalidados na recarga |
| 07 | `SYSTEM_PROMPT` proibia dizer "não sei" | ✅ feito | n/a — não toca em dado |
| 08 | **Corrida de sessão JSF no scraping concorrente.** `coletar_perfis_async` compartilhava um `httpx.AsyncClient` entre 8 requisições; o portal do SIGAA guarda o docente corrente na sessão do servidor, atrelada ao cookie compartilhado. Media **6 de 15** perfis com a pessoa errada e **47%** de captura. | ✅ feito — cliente isolado por requisição + conferência do nome contra a listagem + repescagem em série | ❌ exige recarga |
| 09 | **A guarda do 06 rejeitava perfil legítimo e esparso.** Docente que não preenche a seção descritiva faz o SIGAA omitir o `<dl>` inteiro; `CHAVES_DE_PERFIL` então descartava a pessoa. Custava **8 de 30** docentes reais. | ✅ feito — identidade pela listagem substitui a checagem de chaves | ❌ exige recarga |

**Regra de trabalho combinada com o orientando:** antes de alterar qualquer
arquivo, apresentar o que será feito e aguardar aval explícito. Um achado por
vez — propor, receber o aval, então implementar.

**Ordem de execução (revista em 4 set 2026, depois do achado 08):**

1. **08 — corrida de sessão no scraping.** Isolar o cookie por requisição e
   conferir o nome do perfil contra o nome da listagem.
2. **02 e 03 no código** — parar de chunkar perfis, a tool devolver metadados,
   limpar o boilerplate `"não informada"` do texto indexado.
3. **Recarga única do ETL**, já contendo 01, 02, 03, 08 e os campos grátis do 05
   (Lattes, sala, e-mail). Tudo que exige recarga entra nesta **única** execução:
   fazer em três passadas é desperdício e carga desnecessária no servidor da
   UFRRJ.
4. **04 — auditoria**, agora com pareamento confiável entre id e pessoa.
5. Calibrar o limiar de similaridade → bateria de roteamento → 05 completo.

> ⚠️ **Por que o 08 passou à frente de tudo.** Enquanto o scraper disputar a
> sessão do SIGAA consigo mesmo, metade dos docentes não é coletada e parte dos
> coletados fica sob a identidade de outra pessoa. Qualquer medição feita sobre
> este corpus — contagem, roteamento, auditoria — mede dado corrompido. Não
> adianta verificar se o agente roteia certo para uma resposta errada.
>
> ⚠️ **A auditoria do 04 desceu para depois da recarga.** Ela compara "o que o
> SIGAA mostra" com "o que capturamos". Rodá-la antes exigiria parear pelo id
> guardado, que é exatamente o que o 08 mostrou não ser confiável. Depois da
> recarga corrigida, o pareamento é válido. O código do 01, corrigido e
> verificado em 4 set, já não bloqueia nada.

### Os três princípios transversais (definidos pelo orientando)

Valem para qualquer solução proposta neste projeto:

1. **Realismo sobre a completude do SIGAA.** O SIGAA é a fonte da verdade e
   assumimos que está completo. Docente que não preencheu o perfil não é falha
   do nosso algoritmo. Nosso papel é lidar bem com o que existe, não compensar
   o que falta.
2. **O teste de correção que importa:** *"se o docente tivesse preenchido o
   dado, nosso sistema o encontraria?"* Se sim, o sistema está correto — mesmo
   que a resposta final esteja incompleta em relação ao mundo real.
3. **Especulação só com respaldo explícito.** Só se afirma que alguém pesquisa
   em uma área quando um dado explícito diz isso. Dedução por proximidade ("é
   da Computação, logo talvez pesquise IA") é proibida. Quando falta dado, as
   saídas aceitáveis são dizer que não encontramos, ou responder apenas com os
   vínculos de fato encontrados.

## Estrutura (atualizada em 4 set 2026)

```
CLAUDE.md                    # este guia
config.py                    # config centralizada — só o Módulo 2 a consome de fato (ver armadilhas)
Dockerfile
docker-compose.yml           # serviços: chromadb, etl, agente
rag.sh                       # entrypoint de tudo: build | etl | agente | comparar | tunel | logs
tunel.sh                     # túnel SSH via ControlMaster: up | manter | status | down | ajuda
requirements.txt

modulo1_etl/                 # scraping → dedup → chunking → embedding → carga
  parte1_scraping_sigaa.py    # varredura ampla do SIGAA — FORA DE ESCOPO na fase atual (§4)
  parte2_scraping_docentes.py # scraping da aba 'docentes' — único escopo ativo agora
  parte3_chunking.py
  parte4_embedding.py
  parte5_carga.py             # entrypoint do pipeline ETL
  db_manager.py               # SQLite; normalizar() resolve a cegueira a acentos
  deduplicacao.py             # 1278 SIAPEs → 703 pessoas; roda dentro de scrape_docentes()
  reconstruir_sqlite.py       # recupera o SQLite a partir dos metadados do Chroma, sem re-scraping

modulo2_inferencia/          # motor de inferência
  llm_setup.py                # componentes Haystack + NUM_CTX/REASONING_EFFORT + _ClienteComThink
  tools.py                    # TOOLS_SCHEMA, criar_tools(), criar_dispatcher()
  agent.py                    # loop do agente (tool calling via Ollama) + SYSTEM_PROMPT
  pipelines.py                # os 3 pipelines da fase de validação (ver §3)

interfaces/
  cli.py                      # REPL — entrypoint do agente interativo
  comparar.py                 # runner em lote da comparação dos 3 pipelines

dados/sigaa.db               # banco estruturado; volume montado, senão o --rm o destrói
docs/                        # volume montado; saída da comparação vai para cá
logs/                        # saída do ETL (fora do versionamento)
```

### ⚠️ Convenção de imports mista — não "corrigir" sem entender

`modulo1_etl/` foi mantido intocado no refactor e usa imports **planos**
entre seus próprios arquivos (ex: `parte2_scraping_docentes.py` faz
`from db_manager import salvar_entidades`).

`modulo2_inferencia/tools.py` usa import **qualificado a partir da raiz**
(ex: `from modulo1_etl.db_manager import buscar_entidades_por_campo`).

As duas convenções coexistem de propósito.

> ⚠️ Corrigido em Set/2026: este guia dizia que a coexistência era resolvida
> por `PYTHONPATH=/app:/app/modulo1_etl` no Dockerfile. **O Dockerfile não
> define `PYTHONPATH` nenhum** — verificado. O que faz funcionar é acidental:
> `python modulo1_etl/parte5_carga.py` põe a pasta do script no `sys.path`
> (daí os imports planos funcionarem), e o `WORKDIR /app` cobre os
> qualificados quando se roda com `-m` a partir da raiz.
>
> Consequência prática: um script novo em `modulo1_etl/` que importe
> `config` ou `modulo1_etl.algo` **falha** se rodado como
> `python modulo1_etl/script.py` — precisa de `python -m modulo1_etl.script`.
> Foi exatamente o que aconteceu com `reconstruir_sqlite.py`.
>
> Adicionar de fato o `ENV PYTHONPATH=/app:/app/modulo1_etl` no Dockerfile
> continua em aberto — faria as duas formas funcionarem e alinharia o
> comportamento com o que este guia sempre afirmou.

## Comandos

Tudo passa pelo `rag.sh` na raiz — ele orquestra Docker e túnel juntos.

| Comando | O que faz |
|---|---|
| `./rag.sh build` | Reconstrói as imagens. **Necessário após qualquer alteração de código** — ver armadilha 1 |
| `./rag.sh chroma` | Sobe só o ChromaDB |
| `./rag.sh etl` | Sobe o Chroma e roda o ETL completo (scraping → dedup → chunking → embedding → carga) |
| `./rag.sh agente` | Levanta o túnel e abre o REPL interativo |
| `./rag.sh comparar` | Levanta o túnel e roda a bateria dos 3 pipelines |
| `./rag.sh logs` | Segue os logs do serviço `etl` |
| `./rag.sh tunel up|manter|status|down|ajuda` | Delega para `tunel.sh` |

Utilitários sem atalho no `rag.sh`. **Sempre com `-m`, a partir da raiz** — a
forma `python modulo1_etl/x.py` quebra o `import config` (ver a convenção de
imports acima):

```bash
docker compose run --rm etl python -m modulo1_etl.deduplicacao
docker compose run --rm etl python -m modulo1_etl.reconstruir_sqlite --dedup
```

## ⚠️ Achado 08 — corrida de sessão JSF no scraping (4 set 2026)

**É o defeito mais grave encontrado até agora, e é nosso.** Descoberto ao rodar
a auditoria do achado 04.

### O que acontece

`coletar_perfis_async` dispara os perfis com `asyncio.gather` sobre **um único
`httpx.AsyncClient` compartilhado**, com `MAX_WORKERS = 8`. O portal público do
SIGAA é JSF e guarda o docente corrente em **estado de sessão no servidor**,
atrelado ao cookie `JSESSIONID` — que é único porque o cliente é único. As oito
requisições concorrentes disputam essa sessão: a última a escrever vence, e as
outras recebem a página **de outra pessoa**, com HTTP 200 e conteúdo perfeitamente
bem formado.

### Medição (15 docentes de Ciência da Computação/IM)

| Como buscar | Perfis com a pessoa errada |
|---|---|
| Cliente compartilhado — como o ETL faz hoje | **6 de 15** |
| Um cliente (cookie) por requisição | **0 de 15** |
| Idem, segunda rodada | **0 de 15** |

Os 6 errados voltaram **todos com o mesmo docente**. Não é ruído aleatório: é o
vencedor da corrida sendo servido a todo mundo.

### O que isso produziu no store

1. **Perfis gravados sob o id de outra pessoa.** `source_url` leva à página
   errada. Numa amostra de 41, a URL guardada devolve outro docente em **93%**.
2. **Duplicatas falsas.** O mesmo docente entrou várias vezes, uma por id que
   perdeu a corrida — com conteúdo *byte a byte idêntico*, porque é literalmente
   a mesma resposta HTTP.
3. **Docentes que nunca foram coletados.** O id de quem perdeu a corrida nunca
   rendeu o perfil dele. Medido contra a listagem ao vivo: **6 de 15** e **8 de
   15** capturados nos dois departamentos de computação — **47%**.

### ⚠️ Isto reinterpreta a "inflação de 82% por duplicatas"

O registro anterior dizia que o SIGAA cadastra a mesma pessoa sob várias SIAPEs,
e que 353 de 370 nomes repetidos tinham perfil byte a byte idêntico. **Perfil
byte a byte idêntico é a assinatura desta corrida**, não de cadastro duplicado —
duplicatas reais do SIGAA teriam pequenas diferenças. A deduplicação estava
tratando o sintoma.

Consequência: **"quantos docentes tem o departamento X?" provavelmente responde
cerca de metade do valor certo**, e a contagem de 703 pessoas é um piso, não o
número. O valor verdadeiro deve estar mais perto das 1278 siapes originalmente
coletadas. Isso precisa ser remedido antes de qualquer bateria de roteamento —
não adianta medir se o agente roteia certo para uma resposta errada.

### Correção desenhada (ainda não aplicada)

1. **Isolar o cookie por requisição** — um `AsyncClient` por perfil, ou desligar
   o compartilhamento de cookies. Mantém a concorrência e elimina a corrida.
   Verificado: 0 erros em duas rodadas.
2. **Conferir o nome contra a listagem.** A listagem por departamento já entrega
   o par (nome, id). Comparar esse nome com o `<h3>` do perfil transforma a
   corrida — e qualquer defeito futuro do mesmo tipo — em falha barulhenta.
   A guarda atual do achado 06 (`f"siape={siape}" in html`) **não pega isto**,
   porque a página servida se autorreferencia com o id pedido.

### Verificação da correção (4 set 2026, 2 departamentos de computação)

| | antes | só com o 08 | com 08 + 09 |
|---|---|---|---|
| Perfis com a pessoa errada | 6 de 15 | 0 de 30 | **0 de 30** |
| Identidade id↔nome confere com a listagem | não verificada | 100% | **100%** |
| Captura contra a listagem ao vivo | 47% | 73% | **100%** |
| Duplicatas remanescentes após a coleta | muitas | — | **0** |

**Zero duplicatas fecha a reinterpretação.** Se a repetição viesse de cadastro
duplicado no SIGAA, ela continuaria aparecendo depois da correção. Não aparece:
30 listados, 30 coletados, 30 pessoas distintas. A "inflação de 82%" era a
corrida de sessão do nosso próprio scraper, do começo ao fim.

### ⚠️ Achado 09 — a guarda do achado 06 rejeita perfil legítimo e esparso

Os 8 perfis não coletados foram recusados com "nenhuma chave de perfil
encontrada". São docentes reais, com nome e departamento corretos — entre eles
BRUNO JOSE DEMBOGURSKI, LEANDRO GUIMARAES MARQUES ALVIM, GIZELLE KUPAC VIANNA.
Eles apenas não preencheram a seção descritiva, e o SIGAA então **omite aquele
`<dl>` inteiro**, deixando só o bloco de contato:

```
['endereço eletrônico', 'endereço profissional', 'sala', 'telefone/ramal']
```

`CHAVES_DE_PERFIL` exige "descrição pessoal", "formação acadêmica" ou "áreas de
interesse", então a página é descartada inteira. Isso contradiz o **princípio 1**
do projeto: docente que não preencheu o perfil não é falha do nosso algoritmo, e
perder a pessoa é muito pior que guardar um registro magro — nome e departamento
são exatamente o que as perguntas de contagem e listagem precisam.

A guarda existia porque não havia testemunha melhor de "esta página é um perfil
de verdade". Agora há: **o nome prometido pela listagem**. Se a listagem disse
que o id é do BRUNO e o `<h3>` diz BRUNO, é o perfil do BRUNO — com ou sem seção
descritiva. A verificação de identidade é estritamente mais forte que a de
chaves, e torna a segunda desnecessária quando a primeira está disponível.

**Corrigido em 4 set 2026:** `_motivo_de_rejeicao` recebe o nome esperado e,
quando ele confere, aceita a página sem exigir chaves de perfil.
`CHAVES_DE_PERFIL` permanece como fallback para quando a listagem não trouxer
nome (HTML do SIGAA mudou). Medido depois: **100% de captura**, 30 de 30.

### O que eu havia escrito antes, e estava errado

A primeira redação deste achado dizia que a `siape` "caducou" e que o SIGAA
teria remapeado os identificadores. **Falso.** Os ids são estáveis entre sessões
independentes e resolvem para a pessoa certa quando buscados em série. A
evidência (93% de divergência) estava certa; a causa que eu atribuí a ela, não.

## Verificação de 02 e 03 (4 set 2026, ponta a ponta, 2 departamentos)

Scraping → chunking, com todo o código corrigido:

| | antes | depois |
|---|---|---|
| Perfis capturados | 47% | **30 de 30 (100%)** |
| Chunks por perfil | vários | **1** — perfil não é mais fatiado |
| Chunks sem o nome do docente no texto | 38% | **0** |
| Chunks com o boilerplate "não informada" | 36% dos campos | **0** |
| Perfis com "Áreas de interesse" | 0 | **15 de 30** |

Exemplo de perfil esparso, que antes era descartado inteiro e agora entra:

```
Docente: MARCELO DIB CRUZ. Departamento: DEPARTAMENTO DE COMPUTAÇÃO. Telefone: 677
```

O ramal "677" também é um ganho: o corte antigo `len(val) > 3` descartava ramal
curto legítimo — medido, 7 de 40 perfis. `_e_placeholder` cobre o que aquele
corte queria pegar (vazio, "-", "n/a") sem jogar fora dado real.

### ⚠️ O limiar de distância está DESLIGADO, e a direção da comparação importa

`LIMIAR_DISTANCIA` existe em `tools.py` mas vem vazio por padrão. Duas razões:
calibrar sobre o corpus atual seria calibrar sobre dado corrompido, e o valor
certo depende da distribuição que só existe depois da recarga.

O `score` do `ChromaEmbeddingRetriever` é **distância, não similaridade** —
menor é mais parecido. Medido em 4 set 2026 com bge-m3: "inteligência
artificial" recuperou os melhores em ~1.02; "culinária japonesa medieval", que
não tem relação nenhuma com o corpus, ainda devolveu 5 documentos, o melhor em
~1.37. Escrever o filtro como `score >= limiar` descartaria exatamente os
relevantes e devolveria só o ruído, **sem erro nenhum**. O filtro está escrito
como `score <= limiar`.

## O SQLite acumulava a cada recarga (achado 10, 4 set 2026)

Encontrado ao preparar a recarga, antes de dispará-la.

`parte5_carga.py` chama `carregar_documentos(..., limpar_antes=True)`, então **o
ChromaDB é limpo** antes de receber a carga nova. O SQLite não tinha nada
equivalente: `salvar_entidades` fazia `INSERT` puro e `init_db` usa
`CREATE TABLE IF NOT EXISTS`.

Consequência se a recarga tivesse rodado assim: o vetor store ficaria com os
~1400 docentes novos, e o SQLite com os 703 antigos **mais** os 1400 — e é o
SQLite que responde "quantos docentes tem o departamento X". As duas metades do
RAG discordariam, e o lado estruturado, que é o determinístico, seria o errado.

Corrigido com `substituir=True` em `salvar_entidades`, que apaga as linhas
daquele `tipo_entidade` antes de inserir. Uma execução completa do ETL é um
**retrato completo**, não um incremento. `reconstruir_sqlite.py` recebeu o mesmo
tratamento pelo mesmo motivo.

## ▶ Recarga concluída — nova linha de base (4 set 2026, 14h22)

```
                                    antes        depois
docentes                             704          1302      (+85%)
duplicatas descartadas na coleta     575             1
captura contra a listagem            47%          100%      (1303 de 1305)
identidade confere com o SIGAA      3 de 41      15 de 15
documentos no Chroma                1130          1302      (1 por docente)
registos no SQLite                   703          1302      (as duas metades batem)
```

**Cobertura por campo**, sobre os 1302:

| Campo | | Campo | |
|---|---|---|---|
| E-mail | 92,9% | Formação | 55,5% |
| Telefone | 92,7% | Áreas de interesse | **50,2%** (era 0) |
| Currículo Lattes | 79,6% | Endereço | 40,2% |
| Perfil | 38,2% | Sala | 27,1% |

**O número de docentes quase dobrou, e este é o dado que importa mais.** Toda
contagem registrada antes desta data está errada por um fator próximo de dois —
inclusive as que apareciam como "corrigidas" depois da deduplicação. As 703
pessoas eram 54% do corpus real.

**A duplicata única é a prova final do achado 08.** Antes: 575 descartadas. Se a
repetição viesse de cadastro duplicado no SIGAA, ela sobreviveria à correção da
corrida. Sobrou uma.

> Duas perdas conhecidas e aceitas: 2 de 1305 perfis caíram por
> `ConnectTimeout`. Não é defeito de lógica, é rede — e aparece contado no log,
> que é o comportamento desejado. Refazer esses dois é barato se algum dia
> importar.

## ▶ Achado 04 respondido — o vazio é da fonte, não nosso (4 set 2026)

`docs/auditoria_perfis.md`, 40 docentes sorteados com semente fixa, comparados
campo a campo contra a página ao vivo do SIGAA.

```
campos com conteúdo real na página ....... 192 de 320 possíveis
desses, ausentes do nosso store .......... 0  (0%)
campos que são placeholder "não informada"  96  (30%)
```

**Em todos os oito campos, "parser captura" é igual a "está no store".** Não há
nada que a página mostre e nós percamos. A resposta ao princípio 2 — *se o
docente tivesse preenchido, nós encontraríamos?* — é sim, sem exceção na
amostra.

O que resta vazio se divide em duas coisas, e nenhuma é nossa:

- **96 campos (30%) são placeholder.** O docente não preencheu e o SIGAA gravou
  "não informada". Desde o achado 03 esse texto fica fora do índice, como deve.
- **8 dos 40 perfis não têm bloco descritivo nenhum.** Perfil, Formação, Áreas
  de interesse e Lattes aparecem como ausentes nos mesmos 8 docentes: o SIGAA
  omite o `<dl>` inteiro quando nada foi preenchido. São exatamente os perfis
  que o achado 09 resgatou de serem descartados — hoje entram com nome,
  departamento e contato, que é o que dá para ter.

### Consequência para o plano

**Investir mais em extração nesta aba não rende nada.** O corpus está no limite
da fonte. Os próximos ganhos só podem vir de:

1. **Calibrar o limiar de distância** (segunda metade do achado 03) — agora há
   um corpus correto sobre o qual medir a distribuição.
2. **As outras abas do SIGAA** (achado 05), que exigem requisições novas e
   arquitetura própria.

> ⚠️ Os dois artefatos publicados linkados no topo deste guia (diagnóstico e
> plano de correção) são de 3–4 set e **não contêm os achados 08, 09 e 10**, que
> são os mais graves. Tratar este arquivo como a fonte da verdade; os artefatos
> ficaram históricos.

## Armadilhas conhecidas

Cada uma destas já produziu — ou produziria — um resultado **plausível e
errado**, que é o modo de falha que este projeto trata como inaceitável.
Estão aqui porque nenhuma delas dá erro: todas passam e devolvem algo.

### 1. O código roda da IMAGEM, não do disco — e nada reconstrói sozinho

O `Dockerfile` faz `COPY` de `config.py`, `modulo1_etl/`,
`modulo2_inferencia/` e `interfaces/` para dentro da imagem. **Não há volume
de código no `docker-compose.yml`** — só `./dados`, `./logs` e `./docs`.
Editar um arquivo e rodar `./rag.sh etl` executa a versão **antiga**, sem aviso
nenhum.

Medido em 4 set 2026: a imagem `rag-federado-ufrrj-etl` era de **03/09 01:38**
e a `rag-federado-ufrrj-agente` de **04/09 00:32** — um dia de diferença entre
as duas. A imagem do ETL não continha sequer `deduplicacao.py` e
`reconstruir_sqlite.py`, arquivos criados depois dela.

**Consequência direta para o plano:** rodar a recarga única do ETL sem
reconstruir executaria o scraper anterior às correções — sem validação de SIAPE
(achado 06), sem deduplicação, sem o casamento por prefixo (achado 01) — e
produziria um corpus que *parece* correto. Queimaria a recarga e, pior, geraria
dado em que todo mundo confiaria.

> **Sempre `./rag.sh build` antes de `etl`, `agente` ou `comparar`**, até
> que os comandos passem a reconstruir sozinhos. Tempo de build é irrelevante
> neste projeto; rodar código velho não é.

### 2. Não existe suíte de testes

Não há nenhum `test_*.py` no repositório e `pytest` **não está** no
`requirements.txt`. Rodar `pytest` coleta zero testes e sai com sucesso — o
que se lê facilmente como "está tudo passando". Não está: não há o que passar.
A verificação hoje é manual, pelo relatório da comparação e pela inspeção dos
achados.

### 3. `config.py` não governa o Módulo 1

Apesar do docstring de `config.py` dizer que centraliza tudo, o ETL relê as
mesmas variáveis por conta própria, com **defaults diferentes dos do `.env`**:

```
parte4_embedding.py:37-38   MODELO_EMBEDDING, EMBEDDING_DIM
parte5_carga.py:37-39,197   CHROMA_HOST, CHROMA_PORT, EMBEDDING_DIM, CHROMA_REMOTE
```

O default embutido é `paraphrase-multilingual-MiniLM-L12-v2` com `dim=384`,
enquanto o projeto usa `BAAI/bge-m3` com `dim=1024`. Se o `.env` não for
carregado, o ETL vetoriza com o modelo errado, numa dimensão errada, **sem
erro** — e o resultado só aparece como recuperação ruim, indistinguível de um
problema de qualidade de dado.

### 4. `DB_PATH` fora do Docker aponta para um banco que não existe

O default em `config.py` é `"sigaa.db"`, caminho **relativo ao diretório de
trabalho**. O `docker-compose.yml` sobrepõe com `/app/dados/sigaa.db` nos dois
serviços, então dentro do container está certo. Rodando na mão fora do Docker,
sem `DB_PATH` no ambiente, o SQLite abre um arquivo vazio novo — e a busca
estruturada responde, com toda a honestidade, que **não há docentes**.

### 5. Este guia não é carregado se a sessão abrir na pasta errada

O `CLAUDE.md` está em `ufrrj/rag-federado-ufrrj/`. Abrir a sessão em
`ufrrj/` (o diretório pai, que só contém essa subpasta) faz o guia **não ser
lido** — a sessão começa sem nenhuma das decisões registradas aqui. Aconteceu em
4 set 2026. Abra sempre em `rag-federado-ufrrj/`.

### 6. `dados/` não está versionado nem ignorado

O `.gitignore` cobre `logs/` mas não `dados/`, que aparece como
`untracked`. Um `git add -A` distraído commita o `sigaa.db` binário.

## 1. Infraestrutura e Deploy

**Execução:** Docker roda inteiramente na máquina local do desenvolvedor,
com privilégios de administrador — sem restrição de usuário sem-admin,
sem workaround de ambiente sem Docker/sem sudo. `Dockerfile`,
`docker-compose.yml`, `rag.sh` e `.dockerignore` ficam na raiz do projeto
(build context = raiz, necessário porque `requirements.txt`, `config.py`,
`modulo2_inferencia/` e `interfaces/` vivem fora de `modulo1_etl/`).

Serviços do compose: `chromadb` (vetor DB), `etl` (roda uma vez, embedding
local), `agente` (REPL, comando `python -m interfaces.cli`, perfil
`agente`).

**O `agente` não roda o Ollama dentro do container.** A aplicação local
faz requisições para o Ollama, que roda na máquina da faculdade — nunca
localmente. Container fala com ele via
`OLLAMA_HOST=http://host.docker.internal:<porta>`.

> Corrigido em Set/2026: o serviço `agente` declarava
> `OLLAMA_HOST=http://host-gateway:11434` em `environment:`, que **sobrepõe o
> `env_file:`** — o valor do `.env` era ignorado, e o container tentava um host
> não resolvível numa porta errada. A linha foi removida para o `.env` valer.
> Se precisar sobrepor `OLLAMA_HOST` de novo, lembre dessa precedência.

### Acesso ao Ollama da faculdade

A máquina `invaders.dcc.ufrrj.br` roda o Ollama. Acesso via túnel SSH,
usando `pacman` (`www.dcc.ufrrj.br`) como jump host — não requer login
direto na invaders, só o túnel:

```
ssh <usuario>@www.dcc.ufrrj.br -L 9999:invaders.dcc.ufrrj.br:11434
```

Com o túnel aberto, `http://localhost:9999` (do lado de quem abriu o
túnel) fala com o Ollama da invaders. Dentro de um container Docker no
Windows/Mac, isso vira `http://host.docker.internal:9999` (sem precisar de
`extra_hosts`; no Linux precisaria mapear `host.docker.internal:host-gateway`).

Verificar disponibilidade sem precisar de permissão nenhuma (é só uma
chamada HTTP):
```powershell
Invoke-RestMethod -Uri "http://localhost:9999/api/version"
```

### Túnel automatizado — `tunel.sh` (Set/2026)

Abrir o túnel à mão obrigava a estar no PC, o que inviabilizava despachar
trabalho pelo telefone. `tunel.sh` faz isso sem interação:

```
./rag.sh tunel up | status | down
```

`./rag.sh agente` e `./rag.sh comparar` já chamam `tunel up` sozinhos (não
abortam se falhar — o erro do túnel é mais informativo que um timeout lá
dentro).

**Exige chave SSH, não senha.** Com senha o `ssh` abriria um prompt que trava
para sempre numa sessão automatizada; por isso o script usa `BatchMode=yes`,
que transforma "pediria senha" em falha em ~2s com mensagem clara. O setup da
chave é uma vez só, precisa do PC, e está documentado em `./tunel.sh ajuda`.
O usuário do DCC vem de `DCC_USUARIO` no `.env`.

O `status` não checa "existe processo ssh" — checa se o Ollama de fato
responde através da porta, que é a única prova que importa.

**O ciclo de vida usa ControlMaster, não busca de PID.** Corrigido em Set/2026:
a versão anterior achava o processo pelo `netstat -ano`, e isso era falso — o
netstat reportava para a porta 9999 um PID (24568) que **não existia** no
`tasklist`, porque o `ssh -f` bifurca e o PID associado ao socket fica obsoleto.
Consequência: `down` não derrubava nada e **ainda assim reportava sucesso**, e
processos ssh órfãos se acumulavam (havia dois, um com mais de 20 h).

Agora o `tunel.sh` pergunta ao próprio ssh pelo socket de controle
(`ssh -O check` / `-O exit`), e o `down` **verifica antes de declarar sucesso**.

> ⚠️ O caminho do socket **não pode conter espaço** — o ssh do Windows quebra
> com `keyword controlpath extra arguments at end of line`. Isso exclui o
> `~/.ssh` deste perfil de usuário ("Raul Nascimento"), daí o socket ficar em
> `/tmp/rag_tunel_dcc`. Sobrepor com `CONTROL_PATH` se necessário.

O `status` distingue três estados que antes eram um só: master ssh vivo,
encaminhamento funcionando, Ollama respondendo.

**`./tunel.sh manter`** supervisiona e reabre sozinho. O `ssh -f` com
`ServerAlive` derruba o túnel quando a rede cai, mas não o levanta de volta —
aconteceu na prática: uma oscilação de internet deixou o túnel morto e
silencioso até alguém rodar `up` à mão. Para operação pelo telefone isso é
inaceitável, porque o comando remoto falha e não há ninguém no PC para
perceber. Deixe o `manter` rodando numa janela, ou registre-o como tarefa
agendada no logon para não depender de nada.

### Hardware alvo: máquina da faculdade (invaders)

**RTX 5070, 16GB VRAM.** Essa é a única especificação de hardware que
importa para a escolha de modelo — a máquina local do desenvolvedor só
roda Docker/orquestração, não faz inferência, então suas specs não entram
na equação de qual modelo cabe ou não.

Modelos já pulados na invaders (checar com
`curl http://localhost:9999/api/tags` ou `Invoke-RestMethod` no
PowerShell — o `curl` do PowerShell é na verdade `Invoke-WebRequest` e não
aceita a sintaxe do curl real):
- `gpt-oss-safeguard:120b`
- `bge-m3:latest`
- `nomic-embed-text:latest`
- `qwen2.5:32b-instruct-q4_K_M`
- `llama3:latest`
- `gpt-oss:latest` (= gpt-oss:20b)
- `mistral:latest`

Modelos adicionais podem ser pulados remotamente sem login na invaders,
via API do Ollama através do túnel (`POST /api/pull`) — mas isso baixa
pro disco de uma máquina compartilhada da faculdade, então vale confirmar
com o orientador/DCC antes de puxar modelos grandes.

## 2. Escolha do modelo

**Critério: melhor qualidade de resposta possível, considerando VRAM
(16GB da RTX 5070) **e** a RAM do sistema da invaders como recurso
disponível — offload parcial de camadas pra CPU/RAM é aceitável, não é
motivo de descarte. Latência e tempo de inferência são irrelevantes —
não entram na decisão.**

> Fator ainda em aberto: não sabemos a quantidade exata de RAM da
> invaders — suposição de trabalho é 16GB ou 32GB (tamanhos comuns pra
> uma máquina com RTX 5070 dedicada), mas isso não foi confirmado, só
> estimado. Na prática, isso não deve mudar a escolha entre os dois
> candidatos abaixo: o offload extra que o `qwen2.5:32b` precisaria
> (~4GB) é pequeno o bastante pra caber mesmo no cenário mais
> conservador (16GB total, dividido com SO e outros processos da
> máquina compartilhada). Só vira decisivo se surgir candidato bem
> maior que os dois listados — aí sim vale confirmar de verdade antes.

### ✅ DECIDIDO (Set/2026): `qwen2.5:32b-instruct-q4_K_M`

Medido, não estimado. O `gpt-oss:20b` **não emite resposta final** numa
fração grande das chamadas: despeja tudo no canal de raciocínio e devolve
`content` vazio com `done_reason='stop'` e `eval_count>0`.

| pergunta | qwen2.5:32b | gpt-oss (melhor config) |
|---|---|---|
| objetiva (SQLite) | 0/3 vazios, acertou 3/3 | 0/3 vazios, acertou 3/3 |
| interpretativa (RAG) | **0/3 vazios** | **2/3 vazios** |

Sem `think`, o gpt-oss dá 4/4 vazios; com `think=high`, 4/4; `low`, 2/4;
`off`, 1/4. Não é falta de contexto — com `num_ctx=32768` foram 3/3 vazios
com `prompt_eval` de apenas ~2000 tokens.

Os critérios desta seção tornam a escolha direta: latência é explicitamente
irrelevante e offload para RAM é aceitável, o que anula as duas vantagens do
gpt-oss (caber inteiro na VRAM, ser mais rápido). Sobra confiabilidade, e um
modelo que não responde a 2 de 3 perguntas interpretativas é inutilizável
num agente.

O texto abaixo é o registro da análise anterior, mantido porque documenta o
raciocínio — mas a decisão está tomada.

**Candidatos, entre as opções já disponíveis na invaders:**

- **`gpt-oss:20b`** (tag `gpt-oss:latest`) — cabe inteiro em 16GB de
  VRAM, sem precisar de offload. Nativamente quantizado em MXFP4
  (~13GB de peso), formato desenhado especificamente pro envelope de
  16GB. Tool calling nativo (relevante porque o agente depende disso —
  `agent.py` passa `tools=TOOLS_SCHEMA` pro Ollama). Contexto de 128K
  tokens — folga generosa pro `TOP_K` de chunks recuperados via RAG.
  Effort de raciocínio configurável (baixo/médio/alto) — mecanismo
  nativo de "trocar tempo por qualidade".
- **`qwen2.5:32b-instruct-q4_K_M`** — volta a ser candidato válido agora
  que offload pra RAM é aceitável: o arquivo de pesos ocupa ~20GB, ou
  seja, só uns ~4GB precisam sair da VRAM e ir pra RAM — offload leve,
  não catastrófico, sobretudo sem restrição de latência. Já está
  pulado na invaders, sem custo de novo download. 32B de parâmetros
  densos (vs. 3.6B ativos por token do gpt-oss, que é MoE) — não dá pra
  afirmar de antemão qual dos dois responde melhor no domínio específico
  deste projeto (dados institucionais em português) sem testar; vale
  comparar os dois na fase de validação (seção 3), não só cravar um.

`MODELO_EMBEDDING=BAAI/bge-m3` (dim=1024) permanece — não é afetado pela
escolha de LLM, e já é a opção de maior qualidade disponível pro
embedding nesse projeto.

```
MODELO_LLM=qwen2.5:32b-instruct-q4_K_M
MODELO_EMBEDDING=BAAI/bge-m3
EMBEDDING_DIM=1024
OLLAMA_HOST=http://host.docker.internal:9999
CHROMA_REMOTE=True
TOP_K=10

NUM_CTX=8192              # janela de contexto; default do Ollama (4096) é apertado para TOP_K=10
REASONING_EFFORT=auto     # auto | off | low | medium | high — só afeta modelos com raciocínio
DCC_USUARIO=seu.usuario   # usado por tunel.sh
```

**Pendência de código — RESOLVIDA (Set/2026).** `llm_setup.py` agora passa
`generation_kwargs={"num_ctx": ..., "reasoning_effort": ...}` ao
`OllamaChatGenerator`, lidos de `NUM_CTX` / `REASONING_EFFORT` no `.env`
(defaults 8192 e `medium`, no mesmo padrão `os.getenv` do `config.py`).
Moram em `llm_setup.py` e não em `config.py` porque só o Módulo 2 gera texto.

> Default de `NUM_CTX` é 8192, não 131072: 128K só existe no gpt-oss, que
> não é mais o modelo escolhido, e 8192 dá folga confortável para TOP_K=10.
>
> **`reasoning_effort` estava inerte e foi consertado.** A suspeita se
> confirmou: dentro de `options` o Ollama ignora em silêncio; o controle real
> é o parâmetro `think`, no topo do payload. Agora `REASONING_EFFORT` é
> traduzido para `think` por `_ClienteComThink`, um adaptador em
> `llm_setup.py` que envolve o cliente do Ollama — a `ollama-haystack 2.2.0`
> não expõe `think` no `run()`, e saltar dela para a 6.x no meio da validação
> era risco maior que o adaptador. Remover o adaptador quando a integração
> for atualizada.
>
> O default é `auto`, que **não envia `think` nenhum**. Isso é deliberado:
> passar `think` para um modelo sem raciocínio configurável — como o
> `qwen2.5`, o modelo atual — é erro, não no-op.

## 3. Validação e estratégia de testes (fase atual)

**Escopo restrito:** apenas dados da aba "docentes"
(`modulo1_etl/parte2_scraping_docentes.py`), exatamente como está
estruturado hoje. `parte1_scraping_sigaa.py` (scraping mais amplo do
SIGAA) não entra nessa fase.

**A validar:** comparação de três pipelines diferentes para responder às
mesmas perguntas sobre docentes, documentando resultado de cada um:

1. **Busca semântica pura (Vector Store)** — só retrieval via
   `ChromaEmbeddingRetriever`, sem tool calling, resposta gerada
   diretamente sobre os chunks recuperados.
2. **Perguntas objetivas via banco estruturado (SQL/estruturado)** —
   resolução direta contra `db_manager.py`/SQLite, sem passar pelo
   vetor store.
3. **Cenário atual com Tool Calling** — o agente decide entre chamar
   ferramentas de busca estruturada ou semântica, conforme já implementado
   em `modulo2_inferencia/agent.py` + `tools.py`.

Cada pipeline deve ser testado com o mesmo conjunto de perguntas
específicas sobre docentes, e os resultados documentados lado a lado
(qualidade da resposta, se usou a fonte certa de dado, se alucinou).
Esse documento de testes ainda não existe — é o próximo artefato a
produzir nessa fase, antes de qualquer refatoração do `modulo1_etl`.

### ⚠️ O critério de sucesso mudou (Set/2026) — mede-se ROTEAMENTO

A redação original desta seção media a qualidade isolada de cada pipeline.
**Estava errada**, e o orientando corrigiu a premissa: o objetivo da comparação
nunca foi deixar os três pipelines bons. Ela existe para justificar a decisão de
arquitetura — mostrar por que mesclamos a abordagem vetorial e a objetiva.

Logo, o pipeline vetorial falhar em contagem **não é defeito, é o resultado
esperado**, e é o que dá sentido ao tool calling. Um pipeline vetorial que
contasse certo tornaria a arquitetura desnecessária. O mesmo vale para o
pipeline objetivo falhar em perguntas genéricas.

O que se mede é a **acurácia de roteamento**. O conjunto de perguntas ganha um
rótulo de rota esperada, definida pela natureza da pergunta e não pela resposta:

```
estruturada  contagem, listagem, vínculo docente↔departamento
semantica    conteúdo do perfil: formação, áreas de interesse, atuação
nenhuma      fora de escopo, ou dado que sabidamente não existe na base
ambigua      exige os dois caminhos (ex.: "quem da Matemática pesquisa
             estatística?") — a classe mais informativa, e a que distingue
             um roteador de verdade de um classificador de palavra-chave
```

Três métricas, não uma:

| Métrica | Definição | Por que |
|---|---|---|
| Acurácia de roteamento | % com rota escolhida == esperada | É a tese da arquitetura |
| Estabilidade | Em N execuções, % que roteia igual todas as vezes | O LLM é estocástico; medir uma vez esconde um roteador que acerta 60% |
| Acurácia condicional | Dado roteamento correto, a resposta é certa | Separa erro de roteamento de erro de resposta |

**Critério de encerramento da fase 3:** roteamento ≥ 95%, estabilidade ≥ 90%,
acurácia condicional ≥ 95% nas objetivas. Nas interpretativas o critério não é
acerto, é **ausência de afirmação sem respaldo explícito** — zero tolerância,
verificada manualmente.

Os três pipelines continuam sendo rodados, mas como **linha de base**: a
evidência de que cada caminho sozinho falha na classe do outro. Isso vira o
argumento do relatório final, não o critério de aprovação.

> O conjunto precisa crescer de 7 para ~30 perguntas para que porcentagens
> signifiquem algo. Só o agente precisa das N repetições. Com N=3 são ~90
> execuções — cerca de 1 h de máquina. Tempo de execução é irrelevante neste
> projeto; o que importa é a bateria sobreviver a ele (ver §1, infraestrutura).

### Qualidade dos dados — três defeitos que invalidavam a medição

Encontrados ao rodar a primeira bateria de verdade (Set/2026). Todos
corrigidos; registrados porque cada um produzia resultado *plausível* e
errado, que é o modo de falha difícil de perceber.

**1. Duplicação de docentes (inflação de 82%).** ⚠️ **A interpretação abaixo foi revista e a revisão está confirmada — ver achado 08.** Depois de corrigir a corrida, a coleta dos dois departamentos de computação devolveu **0 duplicatas** em 30 docentes. O perfil byte a byte idêntico é assinatura da corrida de sessão do nosso próprio scraper, não de cadastro duplicado no SIGAA. A deduplicação continua correta como mecanismo, mas estava mascarando perda de docentes. O SIGAA registra a mesma
pessoa sob vários SIAPEs — 1278 SIAPEs para 703 pessoas, com 353 dos 370
nomes repetidos tendo perfil byte a byte idêntico. "Quantos docentes tem a
Matemática?" respondia 56 quando a resposta é 29. Os três pipelines
concordariam no 56 e os três estariam errados. Corrigido em
`modulo1_etl/deduplicacao.py`, aplicado dentro de `scrape_docentes()` —
o único ponto que alimenta o SQLite e o vetor store ao mesmo tempo.

**2. Busca estruturada cega a acentos.** O `LIKE` do SQLite é
case-insensitive só para ASCII: `LIKE '%Ciência da Computação%'` devolvia 0
e `'%CIÊNCIA DA COMPUTAÇÃO%'` devolvia 6. Como o SIGAA grava em CAIXA ALTA e
o LLM escreve o argumento em caixa mista com acento, a tool falhava em quase
toda pergunta — e o agente respondia, honestamente, que não havia docentes
no departamento. Corrigido com `normalizar()` em `db_manager.py`.

**3. Chunks duplicados desperdiçando o TOP_K.** Consequência do (1) no vetor
store: 2055 chunks para 1152 conteúdos distintos. Numa busca com TOP_K=10
chegavam 4 cópias do mesmo perfil, ocupando o lugar de 4 pessoas diferentes.
O store foi limpo com `python -m modulo1_etl.deduplicacao`, e o ETL agora
deduplica antes do chunking — a duplicata nem chega a ser vetorizada.

### ⚠️ O pipeline 3 estava inerte até Set/2026 — releia testes anteriores

`agent.py` passava as tools em `generation_kwargs`, que a integração joga
inteiro no dict `options` da chamada ao Ollama. Os schemas viravam uma
"option" chamada `tools` (ignorada pelo Ollama) e o parâmetro `tools=` de
verdade ia como `None`. **Nenhuma tool era anunciada ao modelo**, ele nunca
emitia tool call, e `agent.py` retornava cedo em `if not msg_resposta.tool_calls`
— sem tocar no SQLite nem no ChromaDB. As tools de `tools.py` nunca chegaram
a executar.

Corrigido: `criar_tools()` em `tools.py` converte `TOOLS_SCHEMA` em objetos
`haystack.tools.Tool`, e `agent.py` os passa no parâmetro `tools=`. Isso
trouxe `jsonschema` para o `requirements.txt` (dependência de `Tool`).

**Consequência para a documentação:** qualquer anotação de teste sobre o
comportamento do agente feita antes disso mediu um LLM puro sem RAG, não o
pipeline 3. Descartar esses resultados.

### O tool calling virou iterativo (4 set 2026)

`agent.py` fazia exatamente **uma** rodada de ferramentas: PASSO A pedia,
PASSO B executava, PASSO C gerava a resposta final **sem anunciar tools**. O
agente nunca conseguia usar o *resultado* de uma ferramenta para escolher a
próxima.

Isso contradizia o próprio `SYSTEM_PROMPT`, que manda "primeiro o recorte
estruturado, depois o semântico". A instrução só funcionava porque o modelo
emitia as duas chamadas na mesma resposta — acerto de primeira, não
encadeamento. E era justo a **classe ambígua**, a mais informativa da métrica de
roteamento, que dependia dessa sorte estrutural.

Agora `processar_pergunta` roda um laço de até `MAX_RODADAS_TOOL` (padrão 4,
via env). Duas propriedades deliberadas:

- **A última rodada é feita sem tools.** O teto encerra a conversa forçando uma
  resposta em texto, em vez de truncá-la no meio. Por isso o valor precisa ser
  ≥ 2, e há um `raise` explícito se alguém puser 1 — com 1, a única rodada
  seria a última, o agente não receberia ferramenta nenhuma e responderia de
  cabeça *parecendo funcionar*, que é exatamente o defeito de Set/2026.
- **Chamada repetida com argumentos idênticos vira erro explícito** devolvido ao
  modelo, em vez de queimar rodada em silêncio.

`pipelines.py` não precisou mudar: ele já varria o histórico inteiro atrás de
`tool_call_results`, então passou a capturar as ferramentas de todas as
rodadas de graça.

### Como rodar a comparação

```
docker compose --profile agente run --rm agente python -m interfaces.comparar
```

- `modulo2_inferencia/pipelines.py` — os três pipelines, todos com a mesma
  assinatura `(componentes, pergunta) -> ResultadoPipeline`. O pipeline 3
  delega para `agent.processar_pergunta` em vez de reimplementar o loop, para
  a comparação medir o agente real. Histórico novo a cada pergunta, senão uma
  contamina a seguinte.
- `interfaces/comparar.py` — roda o conjunto de perguntas (editável em
  `PERGUNTAS`) e escreve `docs/testes_pipelines.md`. O `docs/` virou volume
  montado no `docker-compose.yml`, senão o relatório morreria no container.
- O runner preenche automaticamente só a **fonte** de cada resposta (é
  verificável: de qual banco o dado saiu). **Qualidade** e **Alucinou?** ficam
  em branco para preenchimento manual — nota automática de qualidade seria
  fabricar o resultado do experimento.

**Registro bruto — `docs/testes_pipelines.jsonl`.** Uma linha por
(pergunta × pipeline × repetição), gravada assim que a célula termina. **Não é
backup do markdown: é o dado primário**, e o markdown é uma projeção dele. Duas
razões: a métrica de estabilidade precisa saber a rota escolhida em CADA
repetição, e o markdown (uma resposta por pergunta por pipeline) não representa
isso; e uma bateria interrompida deixa dado aproveitável em vez de nada. Cada
linha carrega carimbo de configuração — modelo, `TOP_K`, `NUM_CTX`,
`REASONING_EFFORT` e o SHA do `SYSTEM_PROMPT` — para que registros de
configurações diferentes nunca sejam comparados por engano.

**Portão de saúde antes de cada pergunta.** Se o Ollama não responde, espera até
`ESPERA_OLLAMA` segundos (padrão 180); se voltar, segue; se não, **aborta com
código 1** e marca o relatório parcial como interrompido. Antes disso, um Ollama
fora do ar não interrompia nada: cada célula virava `FALHOU` pelo try/except, a
bateria concluía com código 0 e escrevia um relatório que parecia completo e não
valia nada. Quedas e tempo de espera são contabilizados no cabeçalho do
relatório — retry que esconde instabilidade contamina a medição de qualidade com
um problema de infraestrutura invisível.

> Escolha de projeto no pipeline 2: ele é "sem LLM" por definição, então
> precisa de um jeito determinístico de sair da linguagem natural e chegar num
> parâmetro de query. Usa `rapidfuzz` (já era dependência declarada e não
> estava sendo usada em lugar nenhum) casando a pergunta contra os
> departamentos reais do banco, com limiar 70. É deliberadamente burro — a
> graça é ver onde esse caminho barato empata com o agente e onde quebra.


## 4. Refatoração e futuro (próxima fase — bloqueada pela fase 3)

**O que está bloqueado é o ESCOPO DE DADOS, não o código.** Corrigido em
Set/2026: este guia dizia "não mexer no `modulo1_etl`", o que foi lido como
proibição de tocar nos arquivos. Não é isso. Correções de bug em `modulo1_etl`
são não só permitidas como necessárias — a fase 3 depende delas, e duas delas
(acentuação na busca e deduplicação) estavam silenciosamente invalidando os
resultados dos testes.

O que continua fora de escopo é **ativar a varredura ampla do SIGAA**: deixar
de raspar apenas a aba "docentes" e passar a raspar o site inteiro. O
`parte1_scraping_sigaa.py` existe e está pronto, mas a chamada
`scrape_sigaa()` foi removida de `parte5_carga.py` — misturar as duas fontes
fazia o vetor store conter material que a base estruturada não enxerga, o que
invalida a comparação entre os pipelines. Reativar aquela linha é o gesto que
inicia esta fase.

Não iniciar a varredura ampla antes da fase 3 estar concluída.
