# Comparação das três abordagens — as 30 perguntas

**Apurado em 10 set 2026**, contra a rubrica de `docs/pre_registro_comparacao_30.md`,
commitada antes da rodada.

Rodada `20260910T032857` · registro bruto em `docs/comparacao_30_abordagens.jsonl`
· 150 execuções · **0 falhas de infraestrutura**.

---

## Por que este documento existe

O orientador respondeu ao relato da fase 3:

> *"Uma questão importante para a apresentação seria fazer uma comparação entre
> diferentes abordagens. Pelo resultado que você relatou há apenas uma abordagem
> implementada."*

As três abordagens existiam desde 5 set e rodavam nas mesmas 30 perguntas. O que
não existia era a **apuração** da comparação — o relato mostrava só as métricas
do agente. Isto é a apuração.

| | o que é |
|---|---|
| `1-vetorial` | RAG clássico: busca semântica pura, sem tool calling |
| `2-estruturado` | consulta determinística ao SQLite, sem vetor store |
| `3-agente` | agente com tool calling, decide entre os dois |

O agente roda **3 vezes** cada pergunta — é o único componente estocástico. Os
outros dois rodam uma vez: o `2-estruturado` é determinístico e o `1-vetorial`
não roteia nada.

---

## A tabela

| grupo | o que mede | RAG clássico | só banco | agente |
|---|---|---|---|---|
| **A** — 16 objetivas | resposta exata | 4/16 — **25,0%** | 13/16 — **81,3%** | 46/48 — **95,8%** |
| **B** — 6 sem dado | recusou ou inventou | 6/6 — **100%** | 5/6 — 83,3% | 18/18 — **100%** |
| **C** — 7 semânticas | precisão dos nomes citados | 21/29 — **72,4%** | 3/18 — 16,7% | 45/125 — 36,0% |
| **C** — cobertura | quanto do gabarito alcançou | 13,3% | 1,9% | 9,5% |
| **D** — 1 perfil | bate com o perfil da pessoa | 1/1 | não respondeu | 3/3 |
| **E** — 30, respaldo | nome afirmado fora do contexto | 28/30 — 93,3% | *não aplicável* | 89/90 — **98,9%** |

**Em uma frase:** o agente ganha com folga nas perguntas objetivas e nas recusas,
e **perde do RAG puro na metade interpretativa**.

⚠️ **Não existe um número único "acurácia nas 30".** Os grupos medem coisas
diferentes com gabaritos de qualidade diferente — o do grupo A **é** a resposta
(44 docentes, ponto), o do grupo C é um substituto ("quem escreveu a palavra").
Somar os dois produziria um percentual que parece homogêneo e não é. Era o
ponto mais fácil de atacar, e por isso a primeira coluna existe.

---

## Grupo A — 16 objetivas, resposta exata

Contagem, listagem, vínculo docente↔departamento e atribuição departamental.
Gabarito calculado do corpus, verificado automaticamente pelo instrumento.

| tipo | RAG clássico | só banco | agente |
|---|---|---|---|
| contagem | 0/6 — 0% | 4/6 — 67% | 18/18 — **100%** |
| listagem | 0/2 — 0% | 2/2 — 100% | 6/6 — **100%** |
| vínculo | 0/1 — 0% | 0/1 — 0% | 3/3 — **100%** |
| atribuição | 4/7 — 57% | 7/7 — 100% | 19/21 — 90% |
| **total** | **4/16 — 25,0%** | **13/16 — 81,3%** | **46/48 — 95,8%** |

### O caso que resume a tese da arquitetura

> *"Quantos docentes tem o Departamento de Matemática?"*

O RAG clássico respondeu **"há 8 docentes no Departamento de Matemática"**,
listando os oito nomes que apareceram nos 10 trechos recuperados. A resposta é
**44**. (Em 5 set, com os mesmos trechos, ele tinha dito **7** — o número muda,
o erro não.)

Ele não errou a busca — **contou o que coube na janela**. É a limitação
estrutural do RAG puro em pergunta de contagem, e é exatamente o que a busca
estruturada existe para resolver. Um RAG que contasse certo tornaria a
arquitetura desnecessária.

### A régua reproduziu

A mesma tabela sobre a bateria de 5 set (`docs/avaliacao_fase3.jsonl`):

| | RAG clássico | só banco | agente |
|---|---|---|---|
| 5 set | 5/16 — 31,3% | **13/16 — 81,3%** | 44/48 — 91,7% |
| 10 set | 4/16 — 25,0% | **13/16 — 81,3%** | 46/48 — 95,8% |

O pipeline determinístico deu **idêntico**, e `contagem`, `listagem` e `vínculo`
bateram exatamente nos três pipelines. Toda a variação está em `atribuição`, que
é a checagem que depende do LLM — 6 células de 48 viraram no agente, **3 para
cada lado**.

⚠️ Os rótulos foram renomeados entre as duas rodadas — `subconjunto` virou
`precisao_de_atribuicao_departamental` e `listagem` virou
`cobertura_de_listagem`, a renomeação v2a de `docs/criterios_avaliacao.md`. É o
mesmo teste com nome novo: `comparar.py` trata os dois no mesmo ramo. As tabelas
acima estão com os rótulos normalizados.

---

## Grupo B — 6 perguntas sem dado no SIGAA: recusou ou inventou?

`nen-01` a `nen-05` mais a `sem-06`. A `sem-06` entrou aqui, e não na precisão,
porque pergunta as áreas de interesse de **MARCEL WILLIAM ROCHA DA SILVA**, cujo
perfil tem 216 caracteres — só Lattes, sala, telefone e e-mail. **Não há áreas
de interesse, então a resposta certa é dizer que não há.**

| | RAG clássico | só banco | agente |
|---|---|---|---|
| nen-01, 02, 03, 05 | certo | certo | certo ×3 cada |
| **nen-04** (salário) | certo | **ERRADO** | certo ×3 |
| sem-06 (Marcel) | certo | certo | certo ×3 |
| **total** | **6/6** | **5/6** | **18/18** |

### ⚠️ O 83,3% do banco engana, e a rubrica exigia dizer isto

Das 5 respostas corretas dele, **5 são a mesma frase enlatada**:

> *"Não consegui identificar um departamento na pergunta — este pipeline só
> responde a perguntas objetivas sobre departamentos."*

Ele não avaliou que não tinha o dado. Ele **não sabe fazer outra coisa**. É o
limite já declarado em `docs/criterios_avaliacao.md` — *"um agente que responda
sempre 'não encontrei' passa"* — e por isso o pré-registro tornou obrigatório
publicar a contagem de recusas genéricas ao lado do número.

E o único caso em que ele **conseguiu** casar um departamento é justamente o que
ele erra:

> **Pergunta:** *"Qual o salário dos professores do Departamento de Matemática?"*
> **Resposta:** *"O departamento 'DEPARTAMENTO DE MATEMÁTICA' tem 44 docentes:
> ADRIANA OLIVEIRA ANDRADE, ALINE MAURICIO BARBOSA, …"*

Respondeu outra pergunta. Pela regra 2 do grupo B, isso é **ERRADO** — despejar
dado irrelevante não é recusar.

### A melhor resposta das 150, e ela é do agente

> *"Encontrei o registro de MARCEL WILLIAM ROCHA DA SILVA no SIGAA, mas as áreas
> de interesse **não foram especificadas em seu perfil**. Para obter mais
> detalhes, você pode consultar o currículo Lattes dele através deste link: …"*

Acha a pessoa, diz que o campo está vazio, encaminha. É exatamente o
comportamento que o princípio 3 do projeto exige.

---

## Grupo C — 7 semânticas: precisão dos nomes citados

**Gabarito:** um docente pertence ao tema se o **texto descritivo do perfil dele
contém a frase do tema** — a mesma definição de
`modulo2_inferencia/medir_recuperacao.py::gabarito()`. Consultado no ChromaDB,
coleção `rag_sigaa`.

| tema | perguntado em | docentes no gabarito |
|---|---|---|
| agroecologia | sem-01 | 12 |
| movimentos sociais | sem-02 | 13 |
| formação de professores | sem-03 | 33 |
| segurança alimentar | sem-04 | 9 |
| ecologia | sem-07 | 44 |
| literatura | sem-08 | 33 |
| didática | sem-09 | 14 |
| | **total** | **158** |

| | precisão | cobertura | respostas sem nome | nomes inexistentes |
|---|---|---|---|---|
| RAG clássico | **21/29 — 72,4%** | 21/158 — 13,3% | 0 | **0** |
| só banco | 3/18 — 16,7% | 3/158 — 1,9% | 6 de 7 | **0** |
| agente | 45/125 — 36,0% | 45/474 — 9,5% | 0 | **0** |

**Nenhuma das três inventou uma pessoa.** Todos os nomes citados existem no
corpus, nas 150 execuções.

### ⚠️ Por que o agente perde aqui — e a causa está catalogada

O agente cita **muito mais gente** (125 nomes contra 29) com precisão pela
metade. Em três perguntas ele zera:

    sem-03  formação de professores ...... 0/10, 0/9, 0/10
    sem-04  segurança alimentar .......... 0/2, 0/2, 0/2
    sem-09  didática ..................... 0/2, 0/4, 0/2

Na `sem-03` ele responde com docentes do **`DEPARTAMENTO DE FORMAÇÃO
DOCENTE/IM`**. O nome do departamento casa com o tema; o perfil das pessoas, não.
É o **item 3 do backlog** — *"departamento de nome temático atrai consultas sobre
o tema"* — aparecendo agora **no nível da resposta**, e não só no ranking.

O gabarito exclui o nome do departamento (correção do item 7, que levou o recall
de 14% para 27%); o índice em uso, não. A distância entre os dois **é** o defeito.

E o mesmo mecanismo explica os 3/18 do pipeline estruturado. Perguntado *"quem
trabalha com movimentos sociais na universidade?"*, o casamento por similaridade
achou o **`DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE`**
e devolveu os 18 docentes dele. Três escreveram o tema. **A armadilha não é do
LLM: ela está no dado**, e pega os três caminhos por vias diferentes.

E na `sem-09` ele diz a coisa em voz alta:

> *"Encontrei alguns docentes que possuem interesses relacionados à didática **ou
> áreas correlatas** como formação de professores e educação matemática"*

Dedução por proximidade — proibida pelo princípio 3 do `CLAUDE.md`. O RAG
clássico, no mesmo tema, cita quem tem a expressão literal escrita.

### Ressalvas, contadas à parte como a rubrica exige

Respostas que citam alguém e ao mesmo tempo admitem que o perfil não sustenta o
tema, detectadas por marcador textual (*correlatas*, *relacionado*, *não
especifica*, *embora*, …):

    RAG clássico ....  3 de  7
    só banco .......  0 de  7
    agente .........  11 de 21

Elas contam como **erro de precisão** — o nome citado é o que o leitor leva —,
mas a contagem existe separada porque ressalvar é melhor que não ressalvar.

---

## Grupo D — a formação de Filipe Braida

Uma pergunta só. **Não vira taxa**, e é relatada como caso.

| | |
|---|---|
| RAG clássico | **certo** |
| só banco | não respondeu (recusa genérica de escopo) |
| agente | **certo ×3** |

⚠️ **Um erro meu, pego antes de virar veredito.** Ia marcar como sem respaldo o
agente dizer *"Doutorado em Engenharia de Sistemas e Computação pela **UFRJ**"*,
porque a prosa do perfil diz "COPPE/UFRRJ". O perfil traz **as duas formas** — a
prosa diz COPPE/UFRRJ e o campo `Formação` diz "Doutorado … pela UFRJ (2018)".
Estava respaldado. O Lattes que uma das respostas cita também confere com o
perfil, dígito a dígito.

Uma execução escreve o nome como *"FILIPPE BRAIDA"*, com P a mais. Não é
afirmação sobre formação, então passa pela regra — fica registrado como defeito
de outro tipo.

⚠️ **Buraco na rubrica.** Ela não previu "não respondeu" para este grupo, e o
`2-estruturado` caiu exatamente aí. Não forcei em certo nem errado: está como
não respondeu. Fica anotado para a próxima.

---

## Grupo E — nenhum nome afirmado fora do contexto recuperado

**Este grupo só existe por causa de uma correção feita na véspera.** Até
`10129ed`, o registro trocava o texto do contexto pelo tamanho
(`"<5353 caracteres>"`), e o critério de tolerância zero da fase 3 não era
recomputável — o 100% de 5 set era o que a bateria produziu, não o que alguém
conferiu.

O `2-estruturado` está **fora**, e a razão foi fixada antes de rodar: ele grava
`contexto=resposta` (`pipelines.py:167`), porque não tem LLM — a resposta é
montada do resultado do SQL. "Afirmou fora do contexto" é impossível por
construção, e ele tiraria 100% sem significar nada.

| | pela métrica estrita | pela regra 1 do pré-registro |
|---|---|---|
| RAG clássico | 28/30 — 93,3% | **23/23 — 100%** |
| agente | 89/90 — **98,9%** | **67/67 — 100%** |

As duas colunas medem coisas diferentes, e a diferença **é uma pessoa só**. As 3
violações da métrica estrita são todas `MARCEL WILLIAM ROCHA DA SILVA`:

    est-07  rep1  [1-vetorial]   vínculo dele
    sem-06  rep1  [1-vetorial]   áreas de interesse dele
    sem-06  rep2  [3-agente]     áreas de interesse dele

Nas duas perguntas **o nome dele está no enunciado**. A regra 1, escrita antes de
rodar, diz que nome vindo da pergunta não conta — repetir o que o usuário
escreveu não é inventar. A métrica estrita não aplica essa regra.

> **Zero pessoas inventadas em 120 execuções.** Nenhum dos dois pipelines com LLM
> trouxe um nome que não estivesse no contexto e não tivesse vindo da pergunta.

O que aconteceu nesses 3 casos é outra coisa, e vale registrar: o pipeline falou
sobre alguém **cujo documento não foi recuperado** — o contexto tinha 9,6 mil a
11,6 mil caracteres e nenhum deles era o perfil do Marcel. Isso é falha de
recuperação, e é medida no grupo C.

---

## As previsões, apuradas

Escritas em `docs/pre_registro_comparacao_30.md` e commitadas antes da rodada.
**Três das quatro caíram.**

### 23 — ERRADA

> *No grupo C, o `3-agente` não será claramente melhor que o `1-vetorial` em
> precisão. Barra: diferença menor que 15 pontos percentuais.*

Deu **36,4 pontos de diferença** — e com o agente **muito pior**, não melhor. A
intuição de fundo (a vantagem do agente é roteamento, não interpretação) ficou de
pé; a previsão como escrita, não. Não a reescrevo.

### 24 — CONFIRMADA, e mais forte do que eu escrevi

> *No grupo B, o `2-estruturado` terá nota alta (≥ 4 de 6) e quase toda ela vinda
> de recusa genérica.*

5 de 6, e **todas as 5 corretas** são a mesma frase enlatada. Não "quase toda":
toda. Era a previsão que mais importava, porque é a que obriga a tabela a vir
acompanhada da ressalva.

### 25 — ERRADA

> *Na `sem-06`, pelo menos uma das três abordagens inventa áreas de interesse.*

Nenhuma inventou. O agente disse que o campo não está preenchido e encaminhou
para o Lattes; o RAG clássico disse que não encontrou; o banco recusou por
escopo.

### 26 — NÃO SE SUSTENTA

> *No grupo E, o `3-agente` fica acima do `1-vetorial`.*

Pela regra pré-registrada os dois ficam em **100%** — não há diferença para
observar. Pela métrica estrita o agente fica acima (98,9% contra 93,3%), mas essa
não é a régua que eu fixei. Vale o mesmo que valeu para a previsão 21 na bateria
de roteamento: previsão de diferença que não se materializa não é previsão
confirmada.

---

## O que esta apuração NÃO diz

- **Não diz que uma abordagem é melhor "no geral".** Diz em qual grupo cada uma
  vence. O agente perde na precisão interpretativa, e isso está na tabela.
- **Não mede utilidade.** Uma resposta pode passar em C e em E e ser inútil.
- **O gabarito do grupo C é um substituto.** "Escreveu a palavra no perfil" não é
  "pesquisa o tema", e erra nos dois sentidos. Não houve lista de sinônimos, de
  propósito: escolher sinônimo depois de ver o placar seria escolher os que
  ajudam.
- **Juiz único**, sem medida de concordância.
- **Não vale para as abas novas.** Curso, componente e extensão mudam o conjunto
  de perguntas.

### Fraqueza declarada no pré-registro, mantida aqui

Eu já tinha lido as respostas da `sem-01` e da `nen-04` ao investigar o pedido do
orientador, antes de escrever a rubrica. O pré-registro dessas duas é mais fraco
que o das outras 28.

### Uma escolha feita depois da rubrica, e declarada

O pré-registro definiu o gabarito do grupo C como "a frase do tema", mas **não
listou as sete frases**. Foram derivadas do enunciado de cada pergunta
(*"Quais docentes pesquisam agroecologia?"* → `agroecologia`), onde não há margem
de escolha — mas a derivação aconteceu depois de a rodada existir, e isso fica
dito em vez de escondido.

---

## Em que código isto rodou

Branch `main`, com **três** ferramentas — o mesmo agente que produziu os 97,8% da
fase 3. Conferido, não afirmado:

```
git diff --quiet docentes-v1 HEAD -- modulo2_inferencia/tools.py \
                                     modulo2_inferencia/agent.py \
                                     modulo2_inferencia/pipelines.py
    -> sem diferença

schema anunciado ao LLM -> 3 tools
suíte -> 177 passando
```

O `master` tem cinco ferramentas desde `273ac61`, e a bateria de 8 set mostrou
que as mesmas 30 perguntas dão diferente com cinco (roteamento 96,7% contra
97,8%). Rodar ali misturaria dois sistemas.

### As métricas da fase 3, nesta rodada

| | 5 set | 10 set | critério |
|---|---|---|---|
| roteamento | 97,8% | **97,8%** | ≥ 95% |
| estabilidade | 93,3% | **96,7%** | ≥ 90% |
| condicional (objetivas) | 91,7% | **95,8%** | ≥ 95% |
| interpretativas sem afirmação solta | 100% ⚠️ não auditável | **97,5%** auditável | 100% |

A última linha é a que mudou de natureza: o 100% de 5 set não era recomputável,
e o 97,5% de agora é — e a análise do grupo E mostra que as 3 execuções que ele
reprova são a mesma pessoa, nomeada na própria pergunta.

## Como refazer

```bash
docker compose --profile agente run --rm agente python -m interfaces.comparar \
    --saida    docs/comparacao_30_abordagens.md \
    --registro docs/comparacao_30_abordagens.jsonl
```

Sem os dois argumentos, a bateria escreve **por cima** do relatório da fase 3 e
anexa ao registro dela. Os argumentos existem desde `d9f3c1d`, por causa disso.
