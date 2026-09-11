# Agente RAG para dados institucionais da UFRRJ — relatório da Iniciação Científica

**Raul Montes Rosales do Nascimento** · Ciência da Computação — IM/UFRRJ
Orientador: **Prof. Marcel William Rocha da Silva**

---

> ### ⚠️ Este documento é um retrato, não um documento vivo
>
> **Estado em 11 set 2026**, branch `main`, commit `06fa23d`.
>
> O projeto tem seis documentos de trabalho que continuam sendo editados. Este
> aqui **não compete com eles**: ele é o ponto de entrada para quem vai ler o
> trabalho de fora, e está preso a um commit. Se o projeto andar, este arquivo
> não fica errado — vira histórico, como tudo em `docs/historico/`.
>
> **Nenhum número aqui é novo.** Cada um traz o arquivo de onde saiu. Se um
> número não tem medição atrás, ele não está neste documento.
>
> **Escopo:** apenas a aba **Docentes** do SIGAA. A expansão para as demais abas
> existe, está em desenvolvimento no branch `master`, e aparece só na última
> seção.

---

## 1. O que é o projeto

Um agente que responde perguntas em linguagem natural sobre os docentes da
UFRRJ, usando como fonte o **portal público do SIGAA**. O título da IC é
*"RAG para recuperação de informação em redes sociais federadas"*, e o que está
sendo pesquisado é a **recuperação** — a federação em si é escopo do TCC, no ano
seguinte.

Construído com Haystack v2, ChromaDB e Ollama. O modelo de linguagem roda na
máquina da faculdade (`invaders.dcc.ufrrj.br`, RTX 5070), alcançada por túnel SSH.

---

## 2. O problema, e por que ele não tem uma solução só

O agente recebe dois tipos de pergunta que exigem coisas incompatíveis:

> *"Quantos docentes tem o Departamento de Matemática?"*

Tem **uma** resposta, exata, verificável: 44. Errar por um já é errar. Isso é
consulta a banco de dados.

> *"Quem pesquisa inteligência artificial?"*

Não tem resposta exata. A informação existe como **texto livre** que cada
professor escreveu no perfil, com as palavras que ele escolheu. Isso é busca
semântica.

**A aposta do projeto é que um agente consegue decidir sozinho, pergunta a
pergunta, qual dos dois caminhos usar** — e encadeá-los quando a pergunta exige
os dois. Se essa decisão falha, o resto não importa: o sistema entrega uma
resposta bem escrita e completamente errada.

Por isso a métrica central não é "a resposta está boa". É **acurácia de
roteamento**.

---

## 3. Como o sistema é feito

### 3.1 Módulo 1 — ETL (`modulo1_etl/`)

Extrai os dados do SIGAA e os carrega nas duas bases.

| arquivo | o que faz |
|---|---|
| `parte2_scraping_docentes.py` | percorre a listagem de docentes por departamento e coleta cada perfil. **É o único escopo ativo.** |
| `parte3_chunking.py` | monta o texto que vai ser indexado. Hoje **não fatia** o perfil: um documento por docente |
| `parte4_embedding.py` | vetoriza com `BAAI/bge-m3` (1024 dimensões) |
| `parte5_carga.py` | entrada do pipeline; carrega ChromaDB e SQLite |
| `db_manager.py` | acesso ao SQLite. `normalizar()` resolve a cegueira a acentos |
| `deduplicacao.py` | descarta repetição antes da vetorização |
| `auditoria_perfis.py` | compara o que capturamos com a página ao vivo |
| `parte1_scraping_sigaa.py` | varredura ampla do SIGAA — **existe e está desativada**, fora do escopo desta fase |

O resultado são **duas bases que se complementam**:

```
SQLite   ->  nome, departamento, siape           1302 docentes
ChromaDB ->  o texto do perfil, vetorizado       1302 documentos
```

As duas batem em 1302, e isso é verificação, não coincidência: o ETL trata cada
execução como **retrato completo**, apagando antes de inserir. Ver o achado 10
em `CLAUDE.md`.

> ⚠️ **O arquivo do SQLite tem 1374 linhas, não 1302** — as outras 72 são
> departamentos, escritos pelo trabalho de expansão. O diretório de dados é
> volume montado e **não muda quando se troca de branch**, então o banco usado
> nas medições deste relatório contém dado que este relatório não descreve.
>
> Isso foi verificado, não suposto: as três ferramentas consultam
> `buscar_entidades_por_campo("docente", …)` e `total_de_entidades("docente")`,
> e o filtro por tipo está no SQL. **As 72 linhas não alcançam nenhuma resposta
> do agente.** No ChromaDB nem chegam — um departamento não tem texto livre, e
> indexar um nome de departamento é exatamente o defeito que a seção 6.1
> descreve.

### 3.2 Módulo 2 — inferência (`modulo2_inferencia/`)

| arquivo | o que faz |
|---|---|
| `tools.py` | as **três ferramentas** que o agente pode chamar, e o texto que as descreve para o modelo |
| `agent.py` | o laço de decisão: o modelo pede ferramenta, o código executa, o modelo decide de novo (até 4 rodadas) |
| `llm_setup.py` | monta os componentes Haystack e traduz `REASONING_EFFORT` |
| `pipelines.py` | os **três caminhos** que a validação compara |

As três ferramentas:

    buscar_docentes_por_departamento ... conta e lista — vai ao SQLite
    buscar_docente_por_nome ............ vínculo de uma pessoa — SQLite
    busca_vetorial_sigaa ............... o texto do perfil — ChromaDB

> As **descrições** dessas ferramentas são o artefato mais medido do projeto. O
> roteamento de 97,8% é propriedade daqueles textos, não do código: cada uma foi
> reescrita para consertar uma falha de roteamento medida.

### 3.3 A rede simulada (`interfaces/rede/`)

Um site local que simula uma rede social: o usuário publica um post mencionando
o bot, e ele responde dentro da thread.

**Não é o módulo de federação.** Não há ActivityPub nem instância remota. Existe
porque a pergunta, no uso real, não chega limpa — ela chega como menção dentro de
uma conversa:

```
@raul:  quantos professores tem o Departamento de Computação?
@bia:   @ufrrj e a Matemática, tem mais ou menos que esse?
```

*"esse"* não significa nada sozinho. Sem um cenário que force isso, a pesquisa
mediria um chatbot de terminal, que é outra coisa.

Três regras que o módulo não pode quebrar, e a primeira é a que importa:
**nunca publicar resposta quando o agente falhou**. Post do bot é
indistinguível de outro para quem lê; texto de fallback com cara de resposta é
o resultado plausível e errado.

### 3.4 Instrumentos de medição (`interfaces/`, `modulo2_inferencia/`)

| arquivo | mede |
|---|---|
| `conjunto_avaliacao.py` | as **30 perguntas** e a rota esperada de cada uma |
| `comparar.py` | roda as três pipelines e calcula as três métricas |
| `medir_recuperacao.py` | recall@10 da busca semântica |
| `medir_hibrido.py` | semântica × busca por palavra × híbrido |
| `calibrar_limiar.py` | o corte de distância da busca semântica |
| `repontuar.py` | repontua uma bateria já gravada, sem chamar o modelo |

---

## 4. O método

Três regras, e elas são o que separa este trabalho de uma demonstração.

**1. Pré-registro.** A rota esperada de cada pergunta foi escrita e **commitada
antes** de qualquer execução. O mesmo vale para as previsões: antes de cada
rodada eu escrevo o que espero que aconteça e **o que me derrubaria**. O
histórico do git prova a ordem. Sem isso, "o resultado confirmou a hipótese" não
é verificável.

**2. O inimigo é o resultado plausível e errado**, não o resultado ruim. Um
sistema que falha alto é fácil de consertar. Um que devolve 56 quando a resposta
é 29 passa despercebido e contamina tudo depois. `CLAUDE.md` mantém uma seção de
**armadilhas** — cada uma já produziu, ou produziria, exatamente isso.

**3. Testar o que mede, não só o que é medido.** São **181 testes**, e cada um é
a memória de um defeito real. Quatro deles existem porque o instrumento de
medição já quebrou a bateria no meio.

---

## 5. ✅ O que está medido e fecha critério

### 5.1 As três métricas da fase 3

Fonte: **`docs/relatorio_fase5.md`**. 30 perguntas pré-registradas, 3 pipelines,
3 repetições no agente.

| critério | limiar | medido | |
|---|---|---|---|
| Acurácia de roteamento | ≥ 95% | **97,8%** | ✅ |
| Estabilidade | ≥ 90% | **93,3%** | ✅ |
| Condicional objetiva | ≥ 95% | **[95,83% ; 100%]** | ✅ robusta |

**O intervalo da terceira merece explicação, porque é o resultado mais forte do
conjunto.** Em 2 dos 48 itens o verificador não conseguiu decidir com segurança.
Em vez de escolher um lado, o relatório reporta os dois extremos: contando os
dois como errados dá 95,83%, como certos dá 100%. As duas pontas ficam acima do
critério, então **o veredito não depende de nenhuma escolha minha.**

> ⚠️ `docs/avaliacao_fase3.md` mostra **91,7%** nessa linha. Aquele é o valor
> **automático**, antes da auditoria — o instrumento conta item ambíguo como
> reprovado, porque é o único palpite conservador que um cálculo sabe dar. O
> resultado da fase é o intervalo.

### 5.2 A bateria foi refeita e reproduziu

Fonte: **`docs/comparacao_abordagens.md`**, rodada de 10 set, mesmo código.

| | 5 set | 10 set |
|---|---|---|
| roteamento | 97,8% | **97,8%** |
| estabilidade | 93,3% | **96,7%** |
| condicional automática | 91,7% | **95,8%** |

O pipeline determinístico deu **13/16 nas duas rodadas**, e as checagens que não
dependem do modelo bateram exatamente. Toda a variação está na checagem que
depende do modelo.

### 5.3 A comparação das três abordagens

Fonte: **`docs/comparacao_abordagens.md`**. É a resposta à pergunta do
orientador: *"há apenas uma abordagem implementada"*.

| grupo | o que mede | RAG clássico | só banco | agente |
|---|---|---|---|---|
| 16 objetivas | resposta exata | 25,0% | 81,3% | **95,8%** |
| 6 sem dado | recusou ou inventou | **100%** | 83,3% | **100%** |
| 7 semânticas | precisão dos nomes | **72,4%** | 16,7% | 36,0% |

⚠️ **Não existe um número único "acurácia nas 30"**, e isso é deliberado. O
gabarito das objetivas **é** a resposta; o das semânticas é um substituto ("quem
escreveu a palavra no perfil"). Somar os dois produziria um percentual que
parece homogêneo e não é.

**O caso que resume a arquitetura.** Perguntado *"quantos docentes tem o
Departamento de Matemática?"*, o RAG clássico respondeu **"há 8 docentes"**,
listando os oito nomes que apareceram nos 10 trechos recuperados. A resposta é
**44**. Ele não errou a busca — contou o que coube na janela.

**O contraste na ambiguidade.** Existem dois departamentos de Geografia:
`DEPARTAMENTO DE GEOGRAFIA` (16 docentes) e `DEPARTAMENTO DE GEOGRAFIA/IM` (14).
A consulta direta ao banco somou os dois e respondeu **30**. O agente respondeu:

> *"O SIGAA lista dois departamentos relacionados à Geografia: DEPARTAMENTO DE
> GEOGRAFIA: 16 docentes. DEPARTAMENTO DE GEOGRAFIA/IM: 14 docentes. Poderia
> especificar qual deles você está buscando?"*

### 5.4 Zero afirmações inventadas sobre pessoas

Fonte: **`docs/comparacao_abordagens.md`**, grupo E. 120 execuções com modelo de
linguagem. **Nenhuma citou um docente que não estivesse no contexto recuperado**
e não tivesse vindo da própria pergunta.

Isso só pôde ser verificado porque o registro passou a guardar o texto do
contexto (`10129ed`, 10 set). Antes, o critério existia e **não era auditável**.

### 5.5 A qualidade dos dados

Fonte: **`docs/auditoria_perfis.md`**, 40 docentes sorteados com semente fixa,
comparados campo a campo contra a página ao vivo.

```
campos com conteúdo real na página ....... 192 de 320
desses, ausentes do nosso store .......... 0  (0%)
```

**Não há nada que a página mostre e nós percamos.**

### 5.6 A escolha do modelo foi medida, não estimada

Fonte: **`CLAUDE.md` §2**. O `gpt-oss:20b` não emite resposta final em **2 de 3**
perguntas interpretativas — devolve conteúdo vazio. O `qwen2.5:32b` dá **0 de 3**
vazios. Por isso ele é o modelo em uso.

### 5.7 A busca semântica se justifica

Fonte: **`docs/backlog_avaliacao.md` item 7**. Comparação entre busca semântica,
busca por palavra e híbrido, em 6 temas, com as paráfrases escritas antes de
rodar:

```
                     frase exata    paráfrase
semântica              23/133        24/133
busca por palavra      50/133        14/133
híbrido (RRF)          49/133        23/133
```

Na coluna da esquerda a busca por palavra ganha **por construção** — o gabarito
*é* o casamento literal. A coluna que vale é a da direita: quando o usuário não
digita as palavras exatas que o professor escreveu, **a semântica ganha**.

---

## 6. ⚠️ O que está medido e NÃO fecha

Esta seção existe porque o trabalho não termina em 17 de setembro, e omitir isto
tornaria as seções anteriores menos confiáveis, não mais.

### 6.1 A busca semântica acha pouco — recall@10 mediana de 14%

Fonte: **`docs/backlog_avaliacao.md` item 7**.

Critério: entre os docentes que escreveram um tema no próprio perfil, quantos
aparecem nos 10 primeiros resultados. A mediana é **14%**.

O diagnóstico está feito: o ranking é decidido pelo **tamanho do documento**.
Perfil quase vazio, cujo único texto é o nome do departamento, é curto — e
documento curto ganha na comparação de similaridade, ocupando a vaga de quem
escreveu.

**556 dos 1302 docentes (42,7%) não têm nenhum texto descritivo no perfil.**

> Pelo princípio 1 do projeto, isso é da fonte, não nosso: o SIGAA é a verdade e
> docente que não preencheu não é falha do algoritmo. A premissa da proposta é
> que o preenchimento seja obrigatório — é o mínimo. Mas o número afeta toda
> medida de cobertura, e por isso está aqui.

### 6.2 A correção conhecida está SUSPENSA

Reindexar usando só o texto descritivo, sem o cabeçalho institucional, leva a
mediana de **14% para 27%** — medido, com controle separando as duas causas do
ganho.

**Não foi adotada.** No teste ao nível da resposta, o agente passou a dizer que
um docente real *"pode não ser docente desta instituição"*. O ganho é verdadeiro
e o efeito colateral também, e o segundo é pior que o primeiro.

Fonte: **`docs/backlog_avaliacao.md` item 7**, seção "Teste no nível da RESPOSTA".

### 6.3 O agente perde do RAG puro na metade interpretativa

**36,0% contra 72,4%** de precisão. Fonte: **`docs/comparacao_abordagens.md`**,
grupo C.

A causa está identificada e catalogada como **item 3 do backlog**: nome de
departamento temático atrai consultas sobre o tema. Perguntado *"que professores
atuam na área de formação de professores?"*, o agente responde com docentes do
`DEPARTAMENTO DE FORMAÇÃO DOCENTE/IM` — o nome do departamento casa com o tema,
o perfil das pessoas não.

Numa execução ele diz em voz alta:

> *"docentes que possuem interesses relacionados à didática **ou áreas
> correlatas**"*

Dedução por proximidade, que o princípio 3 do projeto proíbe explicitamente.

**A armadilha está no dado, e pega os três caminhos por vias diferentes** — a
consulta direta ao banco cai nela também, casando
`DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE` numa
pergunta sobre movimentos sociais.

### 6.4 Dois defeitos diagnosticados e não corrigidos

Fonte: **`docs/backlog_avaliacao.md`**, itens 8 e 9.

- **O agente reescreve a pergunta e piora a própria busca.** Com a pergunta
  inteira, um docente vem em 1º lugar; com a reescrita que o próprio agente
  emitiu, sai dos 10 primeiros.
- **`buscar_docente_por_nome` casa substring contígua.** "Leandro Alvim" não
  encontra "LEANDRO GUIMARAES MARQUES ALVIM" — falha exatamente na forma como se
  chama um professor.

### 6.5 O instrumento tem cegueiras declaradas

Fonte: **`docs/criterios_avaliacao.md`** e **item 11 do backlog**.

O critério de atribuição **só examina nomes de fora do departamento**. Atribuição
falsa sobre alguém de dentro é invisível para ele, por desenho — e o código diz
isso em voz alta, num comentário chamado *"limite honesto"*.

**Medido na prática**, na pergunta `amb-06`:

> *"Algum professor de Engenharia Agrícola e Ambiental trabalha com
> agroecologia?"*

O departamento tem **31 docentes**. No corpus, **12 pessoas** escreveram
"agroecologia" no perfil — e **nenhuma delas é desse departamento**. A resposta
certa é **"nenhum"**.

| | o que respondeu | com o tema no perfil | o critério aprovou? |
|---|---|---|---|
| RAG clássico | *"não encontrei menção explícita a nenhum professor…"* | — | sim |
| só banco | listou os **31** docentes do departamento | 0 | **sim** |
| agente r1 | 1 nome | 0 | **sim** |
| agente r2 | 2 nomes | 0 | **sim** |
| agente r3 | 3 nomes | 1 — mas de **outro** departamento | **sim** |

**O critério aprovou as cinco execuções**, inclusive a que despejou um
departamento inteiro. Ele está certo em aprovar: atribuição departamental é tudo
o que ele julga, e todas as atribuições estavam corretas. O que é falso não é a
lotação das pessoas — é o *"trabalha com agroecologia"*, e isso o critério não
olha.

E a única que acertou foi o **RAG clássico**, dizendo que não encontrou. É a
mesma assimetria da seção 6.3: na metade interpretativa, o agente não está à
frente do caminho mais simples.

---

## 7. ✗ O que eu afirmei e se mostrou falso

**Esta seção é parte do método, não um apêndice de vergonha.** Um trabalho em que
nenhuma previsão falha é um trabalho cujas previsões não eram arriscadas.

### 7.1 Sobre o corpus

| afirmei | é | como caiu |
|---|---|---|
| 704 docentes | **1302** | o scraper perdia metade por corrida de sessão |
| "inflação de 82% por duplicatas do SIGAA" | eram **do nosso scraper** | depois de corrigir, 30 listados → 30 pessoas distintas |
| "a SIAPE caducou, o SIGAA remapeou ids" | **falso** | ids são estáveis quando buscados em série |

O primeiro é o mais grave: **toda contagem registrada antes de 4 set 2026 está
errada por um fator próximo de dois.** A evidência que eu tinha estava certa; a
causa que atribuí a ela, não.

### 7.2 Previsões pré-registradas que erraram

| # | previ | aconteceu |
|---|---|---|
| 9 | a mediana do recall dobraria (barra: 28%) | **27%** — não atendida por um ponto |
| 12 | o híbrido nunca ficaria abaixo da melhor das duas | ficou abaixo **nas duas** partes |
| 13 | o modo de falha seria inventar | foi **negar que o docente existe** — minha rubrica não tinha essa categoria |
| 23 | diferença menor que 15 pontos na precisão interpretativa | **36,4 pontos**, e com o agente pior |
| 25 | alguém inventaria áreas de interesse num perfil vazio | **ninguém inventou** |

Nenhuma foi reescrita depois do resultado. Estão em
`docs/pre_registro_troca_colecao.md` (13),
`docs/pre_registro_comparacao_30.md` (23 e 25) e no item 7 de
`docs/backlog_avaliacao.md` (9 e 12), todos commitados antes das respectivas
rodadas.

> ⚠️ **Esta lista é recortada, e o recorte é declarado.** As rodadas feitas
> durante a expansão para as outras abas **não entram — nem as previsões que
> erraram, nem as que acertaram.** Elas medem um agente com cinco ferramentas, e
> ficam registradas no `master`.
>
> O recorte é simétrico: nada da expansão foi mantido neste relatório, em
> nenhuma direção. Uma lista de erros silenciosamente encurtada seria pior que
> uma lista curta declarada.

### 7.3 Sobre o próprio instrumento

- **A primeira versão do calibrador de limiar mediu a coisa errada** e
  recomendou 1.10. Ela calculava precisão sobre os 1302 documentos filtrados por
  distância, ignorando que em produção o retriever devolve 10. O número saía, e
  descrevia um sistema que não existe.
- **O pipeline com ferramentas esteve inerte** por um período: as ferramentas
  iam no parâmetro errado da chamada ao Ollama e **nenhuma era anunciada ao
  modelo**. Toda anotação sobre o comportamento do agente feita antes disso
  mediu um modelo puro sem RAG.
- **A documentação do projeto continha quatro afirmações falsas** até 10 set —
  incluindo uma contagem de testes desatualizada e referências a dois arquivos
  que não existem mais, num arquivo que se declara "a única fonte da verdade".
  Corrigidas em `ef7eab3`.

---

## 8. ○ O que nunca foi medido

Declarado para que nada acima seja lido como mais amplo do que é.

| | situação |
|---|---|
| **Federação (Mastodon/ActivityPub)** | **não iniciada.** É escopo do TCC |
| **Contexto de thread melhora a resposta?** | há um interruptor (`--sem-contexto`) para negar a hipótese, e um caso observado ao vivo — mas **nenhuma medição formal** |
| **Injeção de prompt pelo post citado** | o texto de terceiro entra delimitado e o delimitador do usuário é neutralizado. Isso **não elimina** injeção. Item 6 do backlog |
| **Anáfora obrigatória, negação, calibração de ressalva** | pré-registrados em `docs/pre_registro_fase4.md`, **não executados** |
| **Qualidade de texto da resposta** | nenhuma métrica. Nota automática de qualidade seria fabricar o resultado |
| **Concordância entre avaliadores** | juiz único. Não há segundo avaliador |

---

## 9. Próximo passo, em desenvolvimento

No branch `master`, **não medido e fora do escopo deste relatório**.

O projeto foi construído sobre uma suposição nunca escrita: existe **um** tipo de
entidade, o docente. Estender para as outras abas do portal público — cursos,
componentes curriculares, ações de extensão — exigiu derrubar essa suposição.

A saída foi um **registro de tipos**: cada tipo de entidade declara suas buscas
num único lugar, e as ferramentas do agente são geradas a partir dele. Tipo novo
passou a custar **uma entrada**, em vez de edições em cinco arquivos.

O primeiro tipo novo, `departamento`, entrou assim — e os **72 departamentos** já
estão coletados.

> ⚠️ **Isso muda o sistema medido, e o argumento não depende de medir.** O
> agente passa a anunciar cinco ferramentas, **duas delas aceitando o mesmo tipo
> de argumento e devolvendo coisas diferentes** — um nome de departamento serve
> tanto para "quantos docentes tem" quanto para "a que instituto pertence". O
> texto que o modelo lê para decidir é outro, e uma acurácia de roteamento medida
> sobre um conjunto de ferramentas não descreve outro.
>
> A bateria foi de fato repetida com as cinco, e **os números mudaram**. Eles não
> aparecem aqui de propósito: são métrica de outro sistema, e este relatório se
> apresenta como o estado final da aba Docentes. Quem quiser vê-los, estão no
> `master`.
>
> Por isso **toda medição citada neste relatório foi feita no `main`**, com três
> ferramentas.

---

## 10. Onde está cada coisa

| arquivo | o que é |
|---|---|
| `CLAUDE.md` | o guia do projeto — estado, decisões e as armadilhas conhecidas |
| `docs/relatorio_fase5.md` | fecha as métricas da fase 3, com os intervalos de robustez |
| `docs/comparacao_abordagens.md` | as três abordagens nas 30 perguntas |
| `docs/criterios_avaliacao.md` | o método de avaliação e os limites que ele declara |
| `docs/backlog_avaliacao.md` | os 11 defeitos encontrados, e o que foi corrigido |
| `docs/calibracao_limiar.md` | como o corte da busca semântica foi escolhido |
| `docs/auditoria_perfis.md` | o que o SIGAA mostra × o que capturamos |
| `docs/pre_registro_*.md` | o que foi previsto antes de cada rodada |
| `docs/historico/` | versões superadas, preservadas de propósito |

Para reproduzir qualquer medição, os comandos estão em `CLAUDE.md` §3. Os
registros brutos (`.jsonl`) acompanham cada relatório e carregam o carimbo da
configuração — modelo, `TOP_K`, `NUM_CTX` e o SHA do prompt de sistema — para que
rodadas de configurações diferentes nunca sejam comparadas por engano.
