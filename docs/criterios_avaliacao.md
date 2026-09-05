# Critérios de avaliação — precisão de atribuição departamental

**Data:** 5 de setembro de 2026.
**Escreve-se antes do código, de propósito.** A regra existe primeiro em prosa,
com as previsões registradas, para que a implementação possa ser julgada contra
ela em vez de a regra ser julgada contra o que a implementação produziu.

## Registro de honestidade: esta mudança foi decidida DEPOIS de ver o resultado

A bateria `624c82234acd` fechou com a condicional objetiva em **91,7%** (44/48),
abaixo do critério de 95%. Todas as 4 falhas estavam em subconjunto (17/21). Ao
revisar essas 4 falhas na mão, encontrei que o checker reprovava respostas
corretas. **Foi essa revisão — motivada pela reprovação — que originou a regra
abaixo.**

Isso é exatamente a prática que invalida avaliação: mexer no instrumento depois
de ver o número, numa direção que melhora o número. Fica registrado em vez de
escondido, e três salvaguardas acompanham a mudança:

1. O veredito do checker anterior está **congelado** em
   `resultados/v1_624c82234acd.json`, com gate de reprodução item a item.
2. A regra nova **tem que poder reprovar**. A previsão sobre a `amb-06` abaixo é
   o teste disso.
3. A cláusula de recall (v2b), que é construto novo e não conserto, fica
   **separada** e não é comparada ao critério de 95%.

## Por que a v1 estava errada

`interfaces/comparar.py`, bloco `subconjunto`:

```python
elenco = {_normalizar(n) for nomes in verdade["departamentos"].values() for n in nomes}
intrusos = [n for n in afirmados if n not in elenco]
ok = not intrusos
```

Pertinência de conjunto pura. **O checker nunca olha o texto da resposta** — só a
lista de nomes detectados. É estruturalmente incapaz de distinguir estes dois
comportamentos, que ele pontua igual:

> *"Note que DIOGENES FERREIRA FILHO foi listado mas ele pertence ao
> Departamento de Ciências Econômicas e Exatas/ITR, não ao Departamento de
> Matemática."* — o agente **pegou** um falso positivo da recuperação e corrigiu

> um nome de outro departamento citado sem qualquer ressalva, como se fosse do
> departamento perguntado

O primeiro é o melhor comportamento possível e é o que o princípio 3 do projeto
pede. O v1 chama os dois de intruso.

## Duas versões, e só uma julga a fase 3

Descoberto na Fase 0: `ok = not intrusos` é **precision-only**. O checker nunca
penalizou omissão — um agente que responda "não encontrei" sempre passa. A
instrução inicial ("ausência de nome do conjunto esperado continua reprovando")
descrevia comportamento que nunca existiu. Portanto:

| | o que é | julga a fase 3? |
|---|---|---|
| **v2a** | v1 + cláusula do rótulo. **Conserto** de um defeito. | **Sim** |
| **v2b** | v2a + cláusula de recall. **Construto novo.** | **Não** |

A v2b é exploratória, reportada em separado, e o critério de 95% **não se aplica
a ela** — comparar um construto novo a um limiar calibrado sobre outro construto
seria comparar coisas diferentes.

---

# A categoria foi RENOMEADA (5 set 2026)

`subconjunto` → **`precisao_de_atribuicao_departamental`**

**Motivo.** O nome antigo prometia o que a checagem nunca pôde entregar. Em
pergunta ambígua **não existe gabarito para o filtro semântico**: a base sabe em
que departamento cada docente está, e não sabe o que cada um pesquisa. A
checagem nunca verificou se os citados realmente pesquisam o tema — só se o
agente afirmou vínculo departamental falso.

Chamar isso de "subconjunto" convida a ler 17/21 como *"em 17 casos o agente
acertou quem pesquisa o tema"*, que é uma afirmação que o dado não sustenta. O
nome novo diz o que é medido e, por consequência, o que não é.

A `amb-02` é a demonstração: dez docentes citados como trabalhando com
movimentos sociais, dez atribuições departamentais **corretas** — e nove deles
sem uma palavra sobre o tema no perfil. A checagem aprova, e está certa em
aprovar, porque atribuição departamental é tudo o que ela julga.

---

# Política de denominador — fixada ANTES de rodar

Itens marcados **AMBÍGUO** mudam a aritmética. As três opções, com os números
já calculados sobre a projeção do v2a (19 passa, 1 reprova, 1 ambíguo):

| política | categoria | condicional geral |
|---|---|---|
| **(a)** AMBÍGUO fora do denominador | 19/20 = **95,00%** | 46/47 = **97,87%** |
| **(b)** AMBÍGUO como aprovado | 20/21 = **95,24%** | 47/48 = **97,92%** |
| **(c)** AMBÍGUO como reprovado | 19/21 = **90,48%** | 46/48 = **95,83%** |

## Adotada: (a), com reporte obrigatório da contagem de AMBÍGUOS ao lado da nota

**Instrumento que não consegue julgar não deve fingir que julgou.** Contar
ambíguo como aprovado (b) é absolvição por incapacidade de medir; contar como
reprovado (c) é condenação pelo mesmo motivo. Nenhuma das duas é sobre o agente.

## Salvaguarda: o intervalo de robustez, não um limiar arbitrário

A fraqueza de (a) é deixar o instrumento **encolher o próprio denominador** — um
checker que marcasse metade dos itens como ambíguos exibiria uma nota alta sobre
uma base minúscula. A defesa não é um teto de percentual escolhido a dedo, e sim
esta regra:

> Reportar sempre o **intervalo [pior caso, melhor caso]**, calculado resolvendo
> TODOS os ambíguos como reprovados e depois todos como aprovados. O veredito só
> é declarado **robusto** quando as duas pontas caem do mesmo lado do critério.
> Se o intervalo atravessa o critério, o veredito é **NÃO CONCLUSIVO**, e a nota
> pontual de (a) não pode ser apresentada como aprovação.

Aplicada à projeção atual — e este é o motivo de a política valer a pena:

- **Condicional geral: [95,83% ; 97,92%]** — as duas pontas acima de 95%.
  **Robusto: passa, independentemente de como o ambíguo for resolvido.**
- **Categoria isolada: [90,48% ; 95,24%]** — atravessa o critério.
  **Não conclusiva sozinha.**

A afirmação forte que sobrevive é sobre a condicional geral, e ela não depende de
nenhuma escolha de política. É por isso que o intervalo é obrigatório: ele
distingue um resultado que se sustenta de um que depende da régua.

---

# Regra v2a — a cláusula do rótulo

## Princípio

**A checagem avalia a VERDADE da resposta, não o seu formato.** Citar um docente
de fora do departamento não é defeito se a resposta disser de onde ele é e
estiver certa ao dizer.

## Os três braços

1. Nome fora do elenco **não conta como intruso** se, e somente se, a resposta
   declarar vínculo departamental para ele **E** esse vínculo bater com a base.
2. Nome fora do elenco **sem declaração de vínculo**: **reprova**. Ausência de
   evidência reprova; a regra não pode ser satisfeita por vagueza.
3. Nome fora do elenco **com vínculo declarado que não bate com a base**:
   **reprova**. É alucinação com aparência de rigor — pior que o intruso
   silencioso, porque veste a roupa da precisão.

A cláusula se aplica **só a nomes fora do elenco**. Nome dentro do elenco nunca é
avaliado por vínculo declarado.

## Como o vínculo declarado é extraído

Casamento determinístico de string contra o **conjunto fechado** dos nomes de
departamento existentes na base — os 67 valores distintos do campo
`departamento` em `entidades_sigaa`. **Nada de heurística de paráfrase, nada de
LLM, nada de inferência semântica.** "Área de Letras" não é nome de
departamento; "Departamento de Letras/IM" é.

### Comparação de nome de departamento

**Igualdade** do nome completo após normalização (caixa, acento, espaço).
**Proibido containment ou fragmento.** `por_departamento()` já casa por
fragmento para montar o elenco, e reaproveitar isso faria `LETRAS` casar dentro
de `LETRAS E COMUNICAÇÃO SOCIAL` — juntando dois departamentos distintos.

**Os sufixos `/IM` e `/ITR` são discriminantes e não são removidos na
normalização.** `DEPARTAMENTO DE LETRAS/IM` e `DEPARTAMENTO DE LETRAS E
COMUNICAÇÃO SOCIAL` são departamentos diferentes, com 24 e 30 docentes.

### Escopo — dois níveis, sem parâmetro ajustado ao dado

Uma janela em caracteres foi **considerada e recusada**. A proposta era 400 nas
duas direções; as respostas têm 826 a 1058 caracteres, então a janela cobriria o
documento inteiro e a regra viraria *"algum departamento correto aparece em
algum lugar"*. Análise de sensibilidade não conserta uma regra vazia.

**Nível 1 — atribuição local.** Nome e departamento no mesmo parêntese ou na
mesma frase, com delimitador de sentença — não contagem de caracteres.

> `- ANELISE DIAS (DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE)`

**Nível 2 — declaração com escopo, BIDIRECIONAL.** Uma declaração departamental
governa um nome — **antes ou depois dele** — se e somente se **nenhum outro nome
de departamento conhecido aparecer entre a declaração e o nome**.

> ⚠️ A primeira redação dizia "governa os nomes que **a seguem**". Estava
> incompleta, e a incompletude foi encontrada ao medir a `amb-04`: a declaração
> que absolve as três docentes está em **anáfora**, na posição 640, depois dos
> nomes em 539/565/583. Com escopo só para a frente, a `amb-04` falharia por
> buraco na regra e não por defeito do caso — o pior tipo de reprovação.

> *"encontrei diversos docentes vinculados ao 'DEPARTAMENTO DE EDUCAÇÃO DO
> CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE' que trabalham com essa temática:"*
> seguido de dez nomes em lista — a declaração governa os dez, porque nenhum
> outro departamento aparece entre ela e eles.

**Ambiguidade não se resolve por proximidade.** Se mais de um departamento
conhecido ficar em escopo para o mesmo nome, o item é marcado **AMBÍGUO** e vai
para revisão manual. O checker não escolhe o mais próximo. O relatório informa
quantos itens caíram nisso.

### Desempate anafórico — adotado como SECUNDÁRIO, não como regra principal

Decidido **antes de contar quantos itens caem em AMBÍGUO** e antes de rodar.

Uma declaração departamental precedida imediatamente por um marcador anafórico
resolve o escopo **para trás** e desempata. É sinal sintático determinístico —
não semântica, não proximidade.

**Lista FECHADA**, fixada aqui:

```
estes docentes      estas docentes
estes professores   estas professoras
eles                elas
```

O marcador tem de ser imediatamente anterior à declaração departamental, na
mesma frase.

⚠️ **Ele NÃO é a regra principal.** O veredito primário de cada item é o da regra
pura, sem desempate; o desempate entra como **variante reportada ao lado**. Duas
razões:

1. O marcador foi formulado depois de ver a `amb-04`. Mesmo sendo defensável
   sozinho, promovê-lo a regra principal faria a única mudança que converte um
   AMBÍGUO em APROVA nascer do caso que ela beneficia.
2. Reportar os dois vereditos custa uma coluna e mostra exatamente quanto do
   resultado depende dessa escolha.

**A Fase 5 reporta os vereditos COM e SEM o desempate anafórico, item a item.**

Aplicado à `amb-04`, cuja declaração é *"**Estes docentes** estão listados no
Departamento de Letras/IM"*: o marcador está presente, então na variante com
desempate o item vira **APROVA**; no veredito primário permanece **AMBÍGUO**.

### Homônimo

Se `departamento_de(nome)` devolver mais de um departamento, o vínculo declarado
é aceito se casar com **qualquer um** deles, e o item vai marcado como ambíguo no
relatório.

⚠️ **Para o único homônimo do corpus a regra é infalsificável.** Existe um só:
`FERNANDA SILVA FERREIRA CHAER`, em `DEPARTAMENTO DE ADMINISTRAÇÃO E TURISMO/IM`
e `DEPARTAMENTO DE CIÊNCIAS ADMINISTRATIVAS`. Qualquer um dos dois que a resposta
declare será aceito, e não há como a declaração estar errada. Verificado: ela
**não aparece** em nenhuma das 21 respostas de subconjunto nem em nenhum dos 7
elencos. **A cláusula é inerte nesta repontuação.**

---

# Regra v2b — a cláusula de recall (construto novo)

Nome presente no elenco e ausente da resposta conta como **omissão** e reprova.

**Isto não é conserto, é competência nova.** A v1 e a v2a medem só precisão: não
citar ninguém de fora. A v2b passa a medir também cobertura: citar quem devia.

**O critério de 95% não se aplica.** Ele foi fixado observando um construto
precision-only; aplicá-lo a um construto que soma recall compararia coisas
diferentes. A v2b é reportada em separado, como exploração.

## Onde a v2b se aplica: `listagem`, não `subconjunto`

Em **subconjunto** o recall não faz sentido. O elenco é o **departamento
inteiro** — 44 docentes na `amb-01`, 35 na `amb-02`, 31 na `amb-06` — mas a
pergunta é *"quem do Departamento de Matemática pesquisa estatística?"*, cuja
resposta certa é um punhado. Exigir os 44 reprovaria toda resposta correta.

Em **listagem** o elenco **é** a resposta: *"quais docentes pertencem à
Bioquímica?"* pede os 11, e omitir um é erro. É ali que a v2b tem sentido, e
ali ela será rodada na Fase 5 — reportada em separado, sem julgar a fase 3.

⚠️ **Isto é a auditoria simétrica que faltava neste trabalho.** Até aqui todas as
mudanças de checagem só podiam transformar `reprova` em `passa`. A v2b em
listagem é o **único caminho pelo qual um `passa` pode virar `reprova`** —
listagem está hoje em 6/6 sob uma regra que nunca puniu omissão, e essas 6
nunca foram testadas contra o critério de cobertura. O resultado será reportado
explicitamente, **inclusive se as 6 sobreviverem**.

---

# Previsões registradas antes de codificar

## `amb-04` — AMBÍGUO sob v2a. Previsão substituída DUAS vezes.

A previsão anterior dizia que a `amb-04` continuaria reprovada porque "área de
Letras" não é nome de departamento do conjunto fechado. **Ela partiu de revisão
manual truncada minha**: li os primeiros ~620 caracteres da resposta e não li o
parágrafo seguinte, que diz:

> *"Estes docentes estão listados no **Departamento de Letras/IM**, que parece
> ser um departamento afiliado ou parte do mesmo grupo."*

`DEPARTAMENTO DE LETRAS/IM` é o departamento real de `VALERIA ROSITO FERREIRA`,
`CARMEN PIMENTEL` e `ROSEMARY GONCALO AFONSO`. A declaração existe, é explícita
e está correta.

### Mas a previsão corrigida também falhou. O veredito é AMBÍGUO.

Medidas as posições no texto normalizado (892 caracteres), **existem apenas dois
nomes de departamento conhecidos na resposta inteira**, e os três nomes estão
exatamente **entre** eles:

```
pos   29   DEPARTAMENTO DE LETRAS E COMUNICACAO SOCIAL   (o perguntado, errado para elas)
pos  539   VALERIA ROSITO FERREIRA
pos  565   CARMEN PIMENTEL
pos  583   ROSEMARY GONCALO AFONSO
pos  640   DEPARTAMENTO DE LETRAS/IM                     (o real delas)
```

Com escopo bidirecional e a guarda de "nenhum outro departamento intervém",
**os dois ficam em escopo para cada um dos três nomes** — nada intervém em
nenhuma das direções. Pela regra de ambiguidade, **o item é marcado AMBÍGUO e vai
para revisão manual.** Não passa nem reprova automaticamente.

Isso é o comportamento correto da regra, não uma falha dela. Um leitor humano
sabe que *"Estes docentes estão listados no Departamento de Letras/IM"* se refere
para trás, aos três; a regra, sem semântica e por decisão de projeto, não sabe.
Resolver por proximidade daria a resposta certa aqui e é **proibido** — foi
justamente a operacionalização recusada.

**Previsão final registrada: `amb-04` → AMBÍGUO sob v2a.** Se o código aprovar ou
reprovar automaticamente, o código está errado.

**Nenhuma cláusula de vagueza foi criada para preservar a reprovação.** Ajustar a
regra para salvar uma previsão é o erro que motivou este trabalho inteiro.

### Defeito residual, fora desta regra

*"que parece ser um departamento afiliado ou parte do mesmo grupo"* é afirmação
**organizacional sem respaldo**. Verificado na base: `dados_brutos` contém apenas
`departamento`, `nome` e `siape` — **não há campo nenhum que ligue departamentos
entre si**. Existem três departamentos com "Letras" no nome (`LETRAS/IM` 24
docentes, `LETRAS E COMUNICAÇÃO SOCIAL` 30, `DIREITO, HUMANIDADES E LETRAS/ITR`
23), sem relação registrada. O hedge do agente é **especulação pura**, não
verdade hedgeada — contraria o princípio 3.

Pertence a `atribuicao_ok` / `nomes_sem_respaldo`, que estão **congelados** (o
JSONL não guarda o contexto recuperado). Registrado como observação qualitativa
no relatório e em `docs/backlog_avaliacao.md`. **Fora da regra de subconjunto.**

## ⚠️ LIMITAÇÃO ESTRUTURAL: a v2a não reprova NADA em dado real

Projeção final sobre os 21 itens, depois de medir todos os casos:

```
19 passa · 0 reprova · 2 AMBÍGUO (amb-01, amb-04)
```

**Nenhuma das quatro falhas do v1 sobrevive como reprovação.** A célula
`passa → reprova` da matriz de transição estará vazia, e a célula
`reprova → passa` terá duas. A mudança é **inteiramente unidirecional** neste
conjunto de dados.

*"A regra poderia reprovar"* é afirmação teórica. O terceiro braço existe no
texto e nenhum caso real o aciona. Um capítulo de avaliação **não se sustenta
sobre uma regra que, no dado observado, só absolve** — por mais bem justificada
que cada absolvição individual seja.

As duas únicas fontes possíveis de reprovação neste trabalho passam a ser:

1. o **caso sintético** do gold set (departamento declarado incorreto);
2. a **v2b em listagem**, que é a auditoria simétrica e pode converter `passa`
   em `reprova`.

Isso fica escrito para que ninguém leia a subida de 91,7% como validação da
regra. A regra foi validada contra casos construídos; o dado real só a exercitou
no sentido que a favorece.

## `amb-06` — a previsão FALHOU. O caso passa.

⚠️ **Registro de erro meu, o segundo por leitura parcial.** Previ que a `amb-06`
reprovaria pelo terceiro braço, com `ADRIANA DE MAGALHAES CHAVES MARTINS`
declarada em departamento errado. **Falso.** A resposta diz:

> `- ADRIANA DE MAGALHÃES CHAVES MARTINS (DEPARTAMENTO DE CIÊNCIAS SOCIAIS)`

que **é** o departamento real dela. A `amb-06` passa sob v2a.

**A origem do erro é instrutiva e justifica a recusa da proximidade.** Minha
medição anterior usava "departamento conhecido mais próximo no texto
normalizado": `AGROTECNOLOGIAS` termina 4 caracteres antes do nome dela, na
linha de cima, enquanto o parêntese dela começa 37 caracteres depois. A
heurística de proximidade elegeu o departamento da linha anterior e devolveu
"não bate". **O escopo por frase, sobre o texto cru, devolve o certo.**

### Armadilha de implementação, encontrada antes de existir código

`_normalizar` colapsa quebras de linha (`" ".join(...split())`). Dividir o texto
em frases **depois** de normalizar funde os itens de uma lista com o parágrafo
seguinte, e o resultado é lixo: a primeira medição atribuiu 1 departamento à
`amb-04` e 3 à `amb-06`, ambos artefato. **A segmentação em frases tem de
acontecer no texto CRU, com a quebra de linha como fronteira, normalizando cada
pedaço depois.**

## `amb-06` — o que a previsão pretendia testar (mantido para registro)

⚠️ **Rotulada como TESTE DE IMPLEMENTAÇÃO, não pré-registro cego.** Foi formulada
depois de ver o dado.

`ADRIANA DE MAGALHAES CHAVES MARTINS` é declarada na resposta como sendo do
`DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE`. Seu departamento real é
**`DEPARTAMENTO DE CIÊNCIAS SOCIAIS`**.

**Se o código não reprovar a `amb-06` por esse motivo específico, o código está
errado.** Não basta reprovar: tem de reprovar pelo braço 3, com o departamento
declarado e o real registrados no detalhe.

Isto é o que garante que a v2a **não é unidirecional por construção**. Ela
absolve `amb-01` e `amb-04` e mantém `amb-06` reprovada — por razão melhor que a
do v1, que reprovava por pertinência de conjunto e não por rótulo falso.

---

# `amb-01` — AMBÍGUO, aceito. Sem regra de resgate.

> *"Note que DIOGENES FERREIRA FILHO foi listado mas ele pertence ao
> Departamento de Ciências Econômicas e Exatas/ITR, **não ao Departamento de
> Matemática**."*

Dois departamentos conhecidos na mesma frase → dois em escopo → **AMBÍGUO**.

**O instrumento não julga negação, por decisão de projeto.** "Não ao
Departamento de X" é, para a regra, uma menção a X como qualquer outra. E isso
atinge exatamente o comportamento que motivou toda esta correção: o agente
pegando um falso positivo da recuperação e corrigindo-o em voz alta.

**Nenhuma regra de negação foi criada para salvar este caso.** Já foram
acrescentados dois mecanismos estruturais depois de ver os dados — o escopo
bidirecional e o desempate anafórico. Um terceiro, que resgatasse justamente o
caso emblemático, seria ajuste por acúmulo: cada peça defensável sozinha, o
conjunto moldado para produzir o veredito desejado.

Marcadores de **negação** ("não", "não é do", "ao contrário de") vão para o
**pré-registro da próxima bateria** — documentados agora, não aplicados nesta.

A consequência é aceita: a categoria fica **NÃO CONCLUSIVA** e a condicional
objetiva passa **robusta** em [95,83% ; 100%]. A fase 3 fecha assim.

# Status do desempate anafórico: **ADOTADO como secundário**

Para não deixar dúvida, os dois vereditos da `amb-04`:

| | veredito |
|---|---|
| Regra primária, sem desempate | **AMBÍGUO** |
| Variante com desempate anafórico | **APROVA** |

A resposta traz *"**Estes docentes** estão listados no Departamento de
Letras/IM"* — o marcador está na lista fechada e é imediatamente anterior à
declaração. **O veredito oficial da fase 3 é o primário (AMBÍGUO);** a variante
é reportada ao lado, item a item, na Fase 5.

# Verificações registradas

## Nomes curtos — buraco existe, mas é inerte aqui

`nomes_do_corpus()` descarta nomes com 10 caracteres ou menos, para evitar falso
positivo de substring. Isso torna quatro docentes **invisíveis ao detector**:
`AHYAS SISS`, `ANA ARAI`, `JOYCE SATO`, `SOFIA EDER`. O corte só pode **inflar** a
nota, nunca reduzi-la.

**Medido: zero ocorrências dos quatro nomes nas 150 respostas gravadas.** O
buraco não afeta esta repontuação, e afeta v1, v2a e v2b igualmente.

## Nomes citados que o detector NÃO reconhece — a verificação irmã

Nome invisível nunca vira intruso: só pode **inflar** a nota. Extraídos
candidatos a nome próprio das 21 respostas e confrontados com os 1297.

**Buraco estrutural do corpus — desprezível:**

```
nomes que são prefixo de outro nome do corpus ....  0 de 1297  (0,0%)
pares com variante de um caractere ...............  1
     TATIANA DE OLIVEIRA PINTO  ~  TATIANE DE OLIVEIRA PINTO
```

**Um caso real de nome invisível, e ele é do gerador:**

| citado | corpus | onde |
|---|---|---|
| `LUIZ CARLOS ALVES DE MELO` | `LUIS CARLOS ALVES DE MELO` | `amb-04#1` |

⚠️ **É a corrupção de nomes sobrevivendo ao conserto do `SYSTEM_PROMPT`.** Em
listagem a regra pegou (6/6); aqui ela escapa, porque o nome corrompido
simplesmente **desaparece da detecção** em vez de virar erro.

**Diagnosticado — e a diagnose foi possível.** O documento no Chroma, que é o
que qualquer recuperação entrega, traz:

```
meta nome_docente : 'LUIS CARLOS ALVES DE MELO'
content           : 'Docente: LUIS CARLOS ALVES DE MELO. Departamento: ...'
```

**O ETL gravou `LUIS`; o agente escreveu `LUIZ`. A corrupção é do gerador, não
da extração.** A distinção foi possível sem o contexto persistido porque o
documento-fonte continua no Chroma inalterado — logo, esta **não** é uma quarta
cobrança da lacuna do contexto.

**Nenhum dos nomes invisíveis muda veredito nesta rodada.** Tanto
`LUIS CARLOS ALVES DE MELO` quanto `LEANDRO AZEVEDO LAPA E SILVA` pertencem ao
departamento **perguntado** nas respectivas perguntas — estão *dentro* do
elenco. Se fossem detectados, não seriam intrusos. O buraco existe e, aqui, não
inflou nada.

### Falso alarme desfeito: o caso `LEANDRO`

`LEANDRO AZEVEDO LAPA` foi reportado como nome não reconhecido em `amb-02#3`.
**Era erro da minha regex de extração**, que capturou um prefixo.
`nomes_afirmados` detecta `LEANDRO AZEVEDO LAPA E SILVA` corretamente, e ele
está **dentro** do elenco de Ciências Sociais. Não é caso do terceiro braço, e a
`amb-02#3` cita 27 docentes, **todos** de Ciências Sociais, com zero intrusos.

## Não determinismo do comportamento de rotular

As 21 células de subconjunto são 7 perguntas × 3 repetições. **Confirmado: cada
uma das 4 falhas caiu em exatamente 1 de 3 repetições.**

```
amb-01 [..X]   amb-02 [X..]   amb-03 [...]   amb-04 [..X]
amb-05 [...]   amb-06 [X..]   amb-07 [...]
```

O comportamento de citar docente de fora — e rotulá-lo — **aparece em uma
execução de três**. Nas outras duas o agente fica dentro do departamento e não
tem o que rotular.

⚠️ **A métrica de estabilidade não capta isso**, porque mede consistência de
**rota**, não de conteúdo. As sete perguntas rotearam `ambigua` nas três
repetições e contam como 100% estáveis, enquanto o conteúdo varia de forma
material entre execuções. É uma lacuna do instrumento, não do agente.

## Correção a uma premissa: o gabarito da `amb-02` **não é vazio**

A `amb-02` foi descrita como tendo "gabarito vazio (nenhum docente do
departamento perguntado)". **Não é o caso.** `por_departamento("CIÊNCIAS
SOCIAIS")` devolve `{'DEPARTAMENTO DE CIÊNCIAS SOCIAIS': 35}` — 35 docentes,
não ambíguo.

O que está vazio é o **recorte semântico**: a resposta afirma que nenhum dos 35
apareceu associado ao tema de movimentos sociais. O elenco existe; a interseção
com o tema é que o agente reportou como vazia. **Isso muda a natureza da decisão
em aberto** — não é "o gabarito não tinha ninguém", é "o agente disse que
ninguém do gabarito servia, e ofereceu dez de outro departamento, rotulados".

---

# `amb-02` — decidida: passa sob v2a

São **dez** nomes, não nove (o "nove" foi erro de contagem em prosa; o registro
sempre trouxe dez). **Nenhum deles está dentro do elenco de Ciências Sociais**, e
`ADRIANA DE MAGALHAES CHAVES MARTINS` — que é de Ciências Sociais — **não está
entre eles**. Todos os dez são de
`DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE`, declarado
na resposta e conferido contra a base.

A atribuição departamental é verdadeira, e é só isso que a regra julga. **Passa.**

## Observação separada: é colisão de nome, e a resposta é vazia no conteúdo

O departamento oferecido chama-se
`DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, **MOVIMENTOS SOCIAIS** E DIVERSIDADE` — o
nome contém a expressão exata da pergunta. Verificado nos perfis do Chroma,
removendo o nome do departamento do texto antes de procurar o tema:

```
com tema próprio no perfil ....  1 de 10   (MARCELO HENRIQUE BEZERRA RAMOS, perfil de 1933 chars)
sem tema nenhum ...............  9 de 10   (perfis de 151 a 278 chars — nome, departamento, contato)
```

**Nove dos dez perfis não mencionam movimentos sociais em lugar algum além do
nome do próprio departamento.** São os perfis esparsos que o achado 09 resgatou:
têm nome, departamento e contato, e nada mais. A recuperação casou com o **nome
do departamento**, não com a pesquisa de ninguém.

A resposta afirma que esses docentes *"trabalham com essa temática"*. Para nove
deles, a única evidência é que o departamento onde trabalham tem essas palavras
no nome. **Correta na forma, vazia no conteúdo.**

A checagem de subconjunto não julga isso, nem deve tentar — ela não tem gabarito
semântico. Registrado aqui, no relatório da Fase 5 e em
`docs/backlog_avaliacao.md`. É parente do achado 03 (recuperação que não
discrimina) sobrevivendo à calibração do limiar, porque a distância até um
documento cujo *nome de departamento* casa com a consulta é genuinamente curta.

---

# Protocolo do gold set (Fase 3)

**Meus rótulos manuais não são confiáveis como asserção.** Duas das quatro
descrições que produzi — `amb-04` e `amb-06` — estavam materialmente erradas por
leitura truncada. Portanto, antes de montar o gold set:

- reler as 21 respostas de subconjunto **por inteiro**, sem truncar;
- cada rótulo vem acompanhado do **trecho exato citado** que o justifica **e** da
  consulta à base para cada nome envolvido;
- divergência entre descrição anterior e texto completo é **reportada antes de
  usar**.

Já verificado: `amb-01` e `amb-02` conferem com o texto completo.

O caso (c) do gold set — nome de fora com departamento incorreto — **não precisa
mais ser fabricado**: a `amb-06` é instância real. A real entra, e a sintética
permanece, para cobrir o caso de departamento inteiramente inventado (que não
existe na base) além do caso de departamento trocado.

# Sensibilidade a reportar na Fase 5

Além da regra de dois níveis, rodar a janela plana em **150 / 250 / 400 / 800**
caracteres e reportar a concordância **item a item**. Concordância total = duas
operacionalizações independentes chegando ao mesmo veredito. Divergência = o
lugar exato onde a regra depende da operacionalização, e isso precisa ser visto.
