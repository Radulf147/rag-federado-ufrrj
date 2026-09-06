# Fase 5 — repontuação da bateria `624c82234acd`

**Sem conclusão sobre a fase 3.** Este documento traz números e discordâncias. O
veredito da fase é decisão do orientando, depois de ler.

Toda repontuação é sobre as **respostas já gravadas**, sem chamar o LLM. Duas
execuções do mesmo checker sobre o mesmo JSONL devolvem o mesmo veredito, e
qualquer diferença entre rótulos é atribuível ao checker e a mais nada.

## Carimbos

| | v1 | v2a oficial | v2a anáfora |
|---|---|---|---|
| `checker_sha1` | `6a6e7721905a` | `54e26dd6a11c` | `54e26dd6a11c` |
| `desempate_anaforico` | — | `false` | `true` |
| `veredito_oficial` | — | **sim** | não |
| docentes no corpus | 1302 | 1302 | 1302 |
| `prompt_sha1` da bateria | `624c82234acd` | `624c82234acd` | `624c82234acd` |

`sha256` do corpus (pares nome+departamento, normalizados e ordenados) idêntico
nos três arquivos. O `checker_sha1` do v2a oficial e da variante é o **mesmo** de
propósito: é o mesmo código, e o que muda é um argumento — a flag está carimbada
em campo próprio.

O **gate de reprodução** do v1 passou: 44/48, subconjunto 17/21, e **zero itens
divergentes** do veredito gravado pela bateria, conferido item a item.

---

## 1. Matriz de transição v1 → v2a (48 itens)

| de | para | n | itens |
|---|---|---|---|
| passa | APROVA | 44 | — |
| reprova | APROVA | 2 | `amb-02#1`, `amb-06#1` |
| reprova | AMBÍGUO | 2 | `amb-01#3`, `amb-04#3` |
| **passa** | **REPROVA** | **0** | **impossível por construção** |

⚠️ **A célula vazia não é achado empírico.** O v1 reprova um item se e somente se
ele tem ao menos um nome fora do elenco; logo todo item que passou no v1 tem
**zero** nomes fora do elenco; e o v2a só examina nomes fora do elenco. Num item
sem nenhum, não há o que examinar. **As reprovações do v2a são subconjunto
próprio das do v1**, em qualquer corpus e qualquer configuração.

Consequência: **nenhum contrapeso pode existir dentro da v2a.** A mudança é
relaxamento estrito, e a subida de 44 para 46 não pode ser lida como validação
da regra — só as sintéticas do gold set provam que ela reprova.

## 2. As três pontuações, por categoria

| | total | contagem | cobertura | vínculo | atribuição |
|---|---|---|---|---|---|
| **v1** | 44/48 | 18/18 | 6/6 | 3/3 | 17/21 |
| **v2a oficial** | **46/48** | 18/18 | 6/6 | 3/3 | **19/21** |
| v2a anáfora | 47/48 | 18/18 | 6/6 | 3/3 | 20/21 |

Vereditos da categoria de atribuição:

```
v1            17 APROVA ·  4 REPROVA ·  0 AMBIGUO
v2a oficial   19 APROVA ·  0 REPROVA ·  2 AMBIGUO
v2a anáfora   20 APROVA ·  0 REPROVA ·  1 AMBIGUO
```

A variante anafórica muda **um** item — `amb-04#3` —, exatamente o previsto. O
`amb-01#3` permanece ambíguo porque ali a causa é negação, não anáfora.

### Causas dos AMBÍGUOS, separadas

```
negacao            1   amb-01#3   DIOGENES FERREIRA FILHO
anafora            3   amb-04#3   VALERIA · ROSEMARY · CARMEN
vizinho_de_lista   0
```

**Contaminação por vizinho de lista: zero ocorrências.** Isto é **resultado, não
ausência** — a estrutura foi testada (caso `(d)` do gold set, que existe
exatamente para documentá-la e produz AMBÍGUO), e ela não ocorre no dado real. A
regra é conservadora nessa estrutura, e a estrutura não apareceu.

## 3. Intervalos de robustez — medidos

Política de denominador **(a)**, fixada antes de rodar: AMBÍGUO fora do
denominador, com o intervalo `[todos ambíguos reprovados ; todos aprovados]`
sempre reportado.

| | intervalo medido | previsto | atravessa 95%? | veredito |
|---|---|---|---|---|
| Categoria de atribuição | **[90,48% ; 100,00%]** | [90,48% ; 100%] | **sim** | **NÃO CONCLUSIVA** |
| Condicional objetiva | **[95,83% ; 100,00%]** | [95,83% ; 100%] | **não** | **ROBUSTA** |

Os dois bateram exatamente o pré-registro.

A condicional objetiva fica **acima de 95% independentemente de como os dois
ambíguos forem resolvidos** — não depende da política de denominador, nem do
desempate anafórico, nem de qualquer escolha minha. A categoria isolada, não: ela
vai de 90,48% a 100% conforme a resolução, e por isso **não sustenta afirmação
sozinha**.

## 4. Sensibilidade — dois níveis contra janela plana

Operacionalização alternativa: janela plana de W caracteres nas duas direções,
aceitando todo departamento conhecido dentro dela.

| W | concorda | discordâncias |
|---|---|---|
| 150 | 18 de 21 | `amb-02#1` APROVA→REPROVA · `amb-04#3` AMBÍGUO→APROVA · `amb-06#1` APROVA→**AMBÍGUO** |
| 250 | 18 de 21 | idem |
| 400 | 18 de 21 | `amb-02#1` APROVA→AMBÍGUO · `amb-04#3` AMBÍGUO→APROVA · `amb-06#1` APROVA→**AMBÍGUO** |
| 800 | 19 de 21 | `amb-02#1` APROVA→AMBÍGUO · `amb-06#1` APROVA→**AMBÍGUO** |

**A janela plana erra a `amb-06` em todos os quatro W**, e erra pelo motivo já
conhecido: puxa o departamento do vizinho de lista, que fica a 4 caracteres,
enquanto o parêntese do próprio nome começa 37 depois. É a mesma falha que a
métrica de "mais próximo" cometeu na Fase 2 e que levou a rejeitar a
proximidade — aqui ela aparece reproduzida em quatro configurações.

A `amb-02#1` também se desfaz: com W pequeno a declaração fica longe demais dos
últimos nomes da lista, e com W grande entram dois departamentos.

**As duas operacionalizações não são equivalentes**, e a divergência não é ruído:
ela se concentra exatamente nos itens cuja estrutura o desenho de dois níveis foi
feito para tratar.

## 5. `respaldo_de_citacao` — métrica exploratória

**Não julga a fase 3** e **não é contrapeso da v2a** — a v2a é relaxamento
estrito e contrapeso interno a ela é impossível. Isto mede outra coisa: é proxy
parcial e offline de `nomes_sem_respaldo`, congelada por falta do contexto
persistido.

### Agregado nas 92 citações dos 21 itens

```
SEM RESPALDO ..... 33     perfil sem campo descritivo; não há o que casar
COM RESPALDO ..... 43     perfil substantivo e a frase do tema aparece nele
INCONCLUSIVO ..... 16     perfil substantivo, frase não encontrada — NÃO é falha

intervalo [43 ; 59] de 92
```

### Por item

| item | citados | SEM | COM | INCONCL | intervalo |
|---|---|---|---|---|---|
| `amb-01#1` | 6 | 0 | 6 | 0 | [6; 6] |
| `amb-01#2` | 5 | 0 | 5 | 0 | [5; 5] |
| `amb-01#3` | 6 | 0 | 6 | 0 | [6; 6] |
| `amb-02#1` | 10 | 9 | 1 | 0 | [1; 1] |
| `amb-02#2` | 0 | 0 | 0 | 0 | [0; 0] |
| `amb-02#3` | **35** | **23** | **4** | **8** | **[4; 12]** |
| `amb-03#1` | 1 | 0 | 1 | 0 | [1; 1] |
| `amb-03#2` | 1 | 0 | 1 | 0 | [1; 1] |
| `amb-03#3` | 4 | 1 | 1 | 2 | [1; 3] |
| `amb-04#1` | 3 | 0 | 3 | 0 | [3; 3] |
| `amb-04#2` | 4 | 0 | 4 | 0 | [4; 4] |
| `amb-04#3` | 5 | 0 | 5 | 0 | [5; 5] |
| `amb-05#1..3` | 1 cada | 0 | 1 | 0 | [1; 1] |
| `amb-06#1` | 4 | 0 | 3 | 1 | [3; 4] |
| `amb-06#2` | 1 | 0 | 0 | 1 | [0; 1] |
| `amb-06#3` | 1 | 0 | 0 | 1 | [0; 1] |
| `amb-07#1..3` | 1 cada | 0 | 0 | 1 | [0; 1] |

### A previsão arriscada: **CONFIRMADA**

Pré-registrado: *"a maioria dos não-COM-RESPALDO cai em SEM RESPALDO, não em
INCONCLUSIVO. Se muitos caírem em INCONCLUSIVO, o casamento de palavra é que
está grosseiro, e a métrica entra no texto como limitação em vez de medida."*

```
dos 49 não-COM-RESPALDO:   33 SEM RESPALDO   ·   16 INCONCLUSIVO
```

Entra como **MEDIDA**, pelo critério pré-registrado.

### Uma divergência contra o previsto, registrada

Previ **10 COM RESPALDO** na `amb-02#3`; o medido é **4**. A previsão foi
calculada com a medição frouxa anterior, que não separava o texto descritivo dos
campos institucionais. **O pré-registro não foi reescrito** — previsão corrigida
depois do resultado não é previsão. A divergência aparece aqui, e a direção dela
importa: a medição antiga **superestimava** a qualidade da resposta.

## 6. As métricas discordam, nos dois sentidos, no mesmo conjunto

Não é hipótese sobre cegueira de métrica. É demonstração, nos mesmos 21 itens.

| | atribuição departamental | respaldo de citação |
|---|---|---|
| **`amb-01`** | **AMBÍGUO** (`#3`) — o instrumento não consegue julgar | **6/6, 5/5, 6/6** — respaldo total nas três |
| **`amb-02#3`** | **APROVA com folga** — 35 de 35, zero intrusos | **[4; 12] de 35** — 23 sem respaldo nenhum |

**`amb-01`** é a resposta que pega um falso positivo da recuperação, exclui em voz
alta e nomeia o departamento real do excluído — e todas as suas citações têm
evidência própria no perfil. A métrica oficial a marca como não-julgável.

**`amb-02#3`** despeja o departamento inteiro, acerta toda atribuição por
construção, e 23 dos 35 citados não têm uma palavra sobre o tema em lugar nenhum.
A métrica oficial a aprova com folga.

Nos dois casos as duas métricas apontam para lados opostos, e **nenhuma das duas
está errada**: elas medem coisas diferentes. O que o par mostra é que a nota de
atribuição, sozinha, não ordena as respostas por qualidade — e que a soma delas
também não, porque não há como somar.

## 7. Iterações — o custo de fechar

**Gold set: três iterações.**

| | resultado | causa |
|---|---|---|
| 1 | 43 passa, 1 falha | fixture `(d)` não implementava a condição do rótulo |
| 2 | 48 passa, 1 skip | skip de teste que esperava flag `recall` inexistente |
| 3 | 102 passa, 0 falha | suíte inteira |

A única falha de veredito foi de **fixture** — terceira categoria, que não estava
prevista no protocolo (não era bug de implementação nem discordância regra/rótulo:
regra certa, rótulo certo, texto errado). O v2a acertou os outros 11 casos de
primeira, incluindo os dois AMBÍGUO e as duas reprovações sintéticas.

**Validação cega: uma execução, sem iteração.**

```
SUBSTANTIVOS (o que conta) ..... 10 de 10
INSTITUCIONAIS (parte fácil) ... 10 de 10
INCONCLUSIVO identificados ..... 2 de 2
```

A amostra **nunca virou dado de treino** — o classificador rodou uma vez contra
ela e não foi ajustado depois. Critério pré-registrado (≥9 de 10 nos substantivos
E os dois INCONCLUSIVO): **atendido**.

Vale registrar a linha de base: entre os 10 substantivos os rótulos cegos deram 8
COM RESPALDO e 2 INCONCLUSIVO, então **um classificador que respondesse "COM
RESPALDO" para tudo acertaria 8 de 10**. Foi por isso que o critério exigiu 9 e
mais os dois casos difíceis.

## 8. Um conserto posterior ao resultado, declarado

`_causa_do_ambiguo` classificava o `DIOGENES` como `anafora` porque procurava
marcador anafórico em **qualquer** peça da resposta, e a palavra "ELES" aparece
em outro ponto do texto. Consertado com precedência de negação sobre anáfora, e a
busca de anáfora localizada nas peças que contêm um departamento em escopo.

**O conserto foi feito depois de ver a saída, e não afeta veredito nenhum:**
`_causa_do_ambiguo` vive em `interfaces/repontuar.py`, não é chamada por
`_conferir`, e serve só para a tabela da seção 2. Os vereditos antes e depois são
idênticos — 19 APROVA · 0 REPROVA · 2 AMBÍGUO. Registrado como iteração separada.

## 9. O que não é recomputável, e continua congelado

`atribuicao_ok` e `nomes_sem_respaldo` — a verificação de tolerância zero das
interpretativas — **não podem ser reconferidos**, porque o JSONL grava o contexto
recuperado apenas como tamanho (`"<8254 caracteres>"`). Os valores da bateria
permanecem: **100%**, sem possibilidade de auditoria.

É a terceira cobrança da mesma lacuna nesta fase. Registrada em
`docs/backlog_avaliacao.md`.
