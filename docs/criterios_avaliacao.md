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

## ⛔ A v2b-como-recall-em-listagem MORREU: é no-op

**A premissa era falsa, e o erro é meu.** Escrevi na Fase 0 que *"o checker
nunca penalizou omissão"* depois de ler apenas o ramo `subconjunto`, e
generalizei para o instrumento inteiro. O ramo `listagem` sempre puniu:

```python
esperados = [n for nomes in verdade["departamentos"].values() for n in nomes]
faltando  = [n for n in esperados if _normalizar(n) not in resposta_norm]
ok        = not faltando
```

Isso é recall puro. Rodar uma "v2b" ali produziria os mesmos números com outro
`checker_sha1` e nenhuma informação nova.

**A frase "a auditoria simétrica que faltava neste trabalho" está retirada.** Ela
não faltava: já existia e já passou.

### O que ficou subvalorizado por causa do meu erro

**6/6 em listagem significa zero omissões em 63 nomes exigidos** — `est-03` pede
10, `est-06` pede 11, três repetições cada. E o salto foi de **3/6 → 6/6**, pelo
conserto da corrupção de nomes. **A métrica tem sensibilidade real e demonstrada:
ela reprovou quando o agente errava a grafia e aprovou quando parou.** Eu a
tratei como se nunca tivesse mordido; ela mordeu, e o registro mostra a mordida.

## Cegueiras espelhadas — achado

Conferido o ramo `listagem` inteiro: ele calcula `faltando` e **nada mais**. Não
confere intrusos.

| categoria | mede | é cega para |
|---|---|---|
| `precisao_de_atribuicao_departamental` | quem foi citado indevidamente | quem faltou |
| `cobertura_de_listagem` | quem faltou | quem foi citado a mais |

Cada uma é exatamente cega para o que a outra mede. **Uma resposta que listasse a
universidade inteira passaria em `cobertura_de_listagem`** — espelho exato do
despejo dos 35 da `amb-02#3`, que passa em precisão.

Por consistência com a renomeação já feita, `listagem` →
**`cobertura_de_listagem`**. O nome antigo sugeria uma avaliação completa da
listagem; o que existe é metade dela.

## Em `subconjunto` o recall continua não fazendo sentido

O elenco de pergunta ambígua é o **departamento inteiro** — 44 na `amb-01`, 35 na
`amb-02`, 31 na `amb-06` — mas a pergunta pede um recorte temático. Exigir os 44
reprovaria toda resposta correta. Isto permanece válido.

---

---

# `respaldo_de_citacao` — métrica nova, exploratória

## O nome, e por que não "precisão temática"

"Precisão temática" prometeria julgar se o docente **de fato** pesquisa o tema —
o que exigiria gabarito semântico que não existe. `respaldo_de_citacao` diz o
que é medido: **existe, no corpus, respaldo para esta citação?** É pergunta
sobre a fonte, não sobre a pessoa.

## Três classes, não duas

Classificar em "tem o tema / não tem" seria repetir o erro que originou este
trabalho, um nível acima: cobrar forma e chamar de verdade. Ausência de casamento
de palavra tem duas causas incompatíveis, e juntá-las inventa um número.

| classe | definição |
|---|---|
| **SEM RESPALDO** | o perfil não tem conteúdo substantivo. Não há o que casar — não é falso negativo, é ausência. |
| **COM RESPALDO** | perfil substantivo **e** evidência do tema encontrada. |
| **INCONCLUSIVO** | perfil substantivo, evidência não encontrada. **Não conta como falha** — é onde a crueza do casamento de palavra pode estar agindo. |

Reportado como **intervalo `[com respaldo ; com respaldo + inconclusivo]`**, a
mesma disciplina aplicada aos AMBÍGUOS: o que o instrumento não consegue decidir
aparece como largura, não como veredito.

## O corte de "conteúdo substantivo" é ESTRUTURAL, fixado antes de rodar

Não é limiar de caracteres calibrado no dado. É a presença de campo descritivo,
e os campos vêm de `CAMPOS_DO_PERFIL` no ETL:

```
DESCRITIVOS (substantivo)    Perfil · Formação · Áreas de interesse
INSTITUCIONAIS (não conta)   Docente · Departamento · Currículo Lattes
                             Telefone · E-mail · Sala · CEP · Endereço
```

> Um perfil tem **conteúdo substantivo** se, e somente se, contém pelo menos um
> dos três campos descritivos.

Exemplo de SEM RESPALDO, real e integral:

```
Docente: ROBSON MARIANO DA SILVA. Departamento: DEPARTAMENTO DE COMPUTAÇÃO.
Currículo Lattes: link não informado Telefone: 26821469 E-mail: robsonms@ufrrj.br
```

Não há afirmação possível sobre a pesquisa dessa pessoa a partir deste documento.
Que ele seja recuperado por uma consulta temática é o defeito; que nenhuma
palavra do tema case com ele não é surpresa nem erro de medição.

`Currículo Lattes` é institucional apesar de soar acadêmico: aparece em 52,3% dos
perfis, e quando vazio traz o literal `link não informado`. É ponteiro, não
conteúdo.

## Este NÃO é o contrapeso da v2a

Dito claramente para ninguém vender como tal: `respaldo_de_citacao` mede **outra
coisa**. A afirmação honesta sobre a v2a permanece intacta — relaxamento
estrito, contrapeso interno impossível, e o veredito da fase 3 repousa sobre
ela.

O que esta métrica é: **proxy parcial e offline de `nomes_sem_respaldo`**, a
verificação de tolerância zero que ficou congelada por falta do contexto
persistido. Parcial porque `nomes_sem_respaldo` perguntava "este nome estava no
contexto recuperado?" e esta pergunta "o corpus sustenta esta citação?" — a
segunda é mais fraca, porque o documento existir no corpus não prova que ele
chegou ao agente.

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
`reprova → passa` terá duas.

### E ela é IMPOSSÍVEL, não empiricamente vazia

A formulação anterior — *"vazia, e não por acaso de construção"* — estava
errada. É **exatamente** por construção.

O v2a é **relaxamento estrito** do v1. Demonstração em duas linhas:

1. O v1 reprova um item se, e somente se, ele tem ao menos um nome fora do
   elenco. Logo **todo item que passou no v1 tem zero nomes fora do elenco.**
2. O v2a só examina nomes fora do elenco. Num item sem nenhum, não há o que
   examinar, e ele passa necessariamente.

As reprovações do v2a são **subconjunto próprio** das do v1. `passa → reprova`
não é raro: é impossível. Nenhuma resposta, nenhum corpus, nenhuma configuração
produz essa transição.

**Consequência que precisa estar escrita: nenhum contrapeso pode existir dentro
da v2a.** Não é questão de faltar caso real — é impossibilidade estrutural. Se
houver contrapeso neste trabalho, ele vem de **outra métrica**, medindo outra
coisa. O veredito da fase 3 repousa sobre um relaxamento estrito, e isso é dito
sem atenuação.

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

# ⚠️ CASO CENTRAL: a `amb-02` responde duas coisas contraditórias, e as duas passam

Mesma pergunta — *"Quais docentes de Ciências Sociais trabalham com movimentos
sociais?"* — mesmo corpus, mesma configuração, execuções diferentes.

**Repetição 1:**

> *"De acordo com a consulta no SIGAA da UFRRJ, **nenhum docente específico do
> DEPARTAMENTO DE CIÊNCIAS SOCIAIS apareceu** diretamente associado ao tema de
> movimentos sociais. No entanto, encontrei diversos docentes vinculados ao
> 'DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE' que
> trabalham com essa temática:"* — seguido de 10 nomes, nenhum de Ciências
> Sociais.

**Repetição 3:**

> *"Para responder à sua pergunta [...] primeiramente **identifiquei os docentes
> vinculados ao Departamento de Ciências Sociais**. No entanto, ao buscar
> especificamente por 'movimentos sociais', encontrei resultados para o
> Departamento de Educação do Campo, Movim[entos Sociais e Diversidade]..."* —
> seguido de **35 nomes, todos de Ciências Sociais**.

Uma diz que **nenhum** docente do departamento apareceu. A outra lista **o
departamento inteiro**. São afirmações factuais contraditórias sobre o mesmo
corpus, e **ambas são APROVADAS sob v2a** — porque as duas acertam toda
atribuição departamental que fazem.

É a demonstração concreta do que motivou renomear a categoria. `precisão de
atribuição departamental` é literalmente tudo o que está sendo medido, e duas
respostas que se contradizem podem ambas ter precisão perfeita.

## Evidência temática: o filtro semântico não está funcionando em nenhuma das duas

Teste idêntico nas duas — remover o nome do departamento do perfil antes de
procurar o tema:

| | citados | com evidência própria | sem nenhuma | veredito v2a |
|---|---|---|---|---|
| `amb-02#1` | 10 | 1 | 9 | ✅ aprovada |
| `amb-02#3` | **35 de 35** | 4 | **31** | ✅ aprovada |

Os perfis sem evidência na `#3` vão de 114 a 782 caracteres, mediana **166** —
nome, departamento, contato, e nada mais.

## O incentivo perverso, e ele não é hipotético: foi observado

Sob **precisão pura**, a estratégia ótima é **listar o departamento inteiro e
mais ninguém**. Isso garante zero intrusos, portanto 100%, e produz a resposta
menos útil possível — o despejo do departamento sem filtro nenhum, para uma
pergunta que pedia um recorte temático.

**A `amb-02#3` é exatamente essa estratégia, executada.** 35 de 35, nota
perfeita, e 31 dos 35 citados sem respaldo próprio para o tema — 23 SEM RESPALDO
e 8 INCONCLUSIVO, pela medição rigorosa de `respaldo_de_citacao`.

A métrica não é apenas silenciosa quanto a recall. **Ela premia o despejo.**
Dito na forma mais crua possível:

> **3 certos e 1 errado reprova; 35 indiscriminados aprova.**

Esta é a razão mais forte para a v2b existir, e a razão mais forte para não
tratar a subida da condicional como sinal de qualidade do agente.

## E a resposta que parece mais criteriosa é a menos ancorada

A `#1` traz ressalva — *"nenhum docente específico do Departamento de Ciências
Sociais apareceu"* — e por isso soa mais cuidadosa que a `#3`, que simplesmente
despeja. **Proporcionalmente, é a pior das duas:**

| | com evidência temática própria |
|---|---|
| `amb-02#1` | 1 de 10 |
| `amb-02#3` | 4 de 35 |

**A resposta que hedgeia é a menos ancorada.** Quase tudo o que ela oferece
depois da ressalva não tem respaldo nenhum além do nome do departamento. O
hedge não é evidência de cautela — aqui ele precede a afirmação menos
sustentada das duas, e nenhum instrumento deste projeto captura isso.

## Auditoria de atribuição interna — feita, e não achou nada

Nomes **dentro** do elenco nunca eram checados pela regra (a cláusula do rótulo
só se aplica a nomes de fora). Auditados os 21 itens à procura de atribuição
falsa sobre gente de dentro:

```
nomes de dentro com atribuição departamental explícita ....  3
destes, divergentes da base ...............................  0
```

**Não há instância real do terceiro braço escondida ali.** A limitação
permanece como escrita — a regra não tem dentes *nestes dados* —, e não a versão
mais grave ("tem dentes e não alcança onde a mordida existe"). O ponto cego
existe por desenho e continua registrado; ele simplesmente não está ocupado.

# Pré-registro do reporte da v2b — fixado ANTES de rodar

A v2b em listagem é o único caminho pelo qual um `passa` pode virar `reprova`.
Se ela derrubar itens, o relatório fica com *"métrica oficial passa, métrica
exploratória reprova"*, e a tentação de enterrar a segunda é óbvia. Portanto,
comprometido por escrito antes de ver o número:

1. **O resultado da v2b entra no texto com o mesmo destaque, qualquer que
   seja** — inclusive, e principalmente, se derrubar as 6/6 de listagem.
2. Ela **não julga a fase 3**: o veredito oficial continua sendo o da v2a, pelo
   motivo já registrado (construto novo, critério calibrado sobre outro
   construto).
3. Se a v2b reprovar, isso **não** reabre a fase 3 — e **não** é apresentado
   como "defeito exploratório sem importância". É reportado como o que é: uma
   competência que o instrumento oficial nunca mediu, e sobre a qual o sistema
   não tem nota.
4. Se as 6 sobreviverem, isso também é reportado explicitamente, e **não** como
   confirmação de qualidade — 6 itens é amostra pequena e a v2b em listagem tem
   as mesmas limitações de qualquer checagem por casamento de nome.

## Critério de aceite da validação cega — fixado ANTES de executar

Concordância reportada como **fração bruta**, nunca percentual: com n=10 cada
discordância vale 10 pontos, e ponto decimal ali é falsa precisão.

As duas metades da amostra são reportadas **em separado**, porque medem coisas
diferentes:

- **10 institucionais** — a parte **fácil**. Não há texto onde procurar; qualquer
  classificador acerta. Concordância aqui não é evidência de nada e não entra no
  critério.
- **10 substantivos** — onde a decisão entre COM RESPALDO e INCONCLUSIVO exige
  julgar conteúdo. **É só isto que conta.**

### O critério tem de bater a linha de base trivial, e ela é alta

Entre os 10 substantivos, meus rótulos cegos deram **8 COM RESPALDO e 2
INCONCLUSIVO**. Logo um classificador que respondesse "COM RESPALDO" para tudo,
sem ler nada, acertaria **8 de 10**. Essa é a linha de base do inútil, e
qualquer critério abaixo dela mede a distribuição da amostra, não o
classificador.

> **MEDIDA** — entra no texto como medição se, e somente se:
> **(a) ≥ 9 de 10** nos substantivos, **E**
> **(b) os DOIS casos INCONCLUSIVO identificados corretamente.**
>
> **LIMITAÇÃO** — em qualquer outro caso, inclusive 9 de 10 obtido errando um
> INCONCLUSIVO. Errar os dois INCONCLUSIVO com 8 de 10 é exatamente o
> classificador trivial, e ele não vira medida por acidente aritmético.

A condição (b) existe porque as duas classes não têm o mesmo peso probatório: os
INCONCLUSIVO são a razão de a classificação ter três classes em vez de duas. Um
classificador que nunca os produz não implementou a regra — implementou a versão
de duas classes que foi explicitamente rejeitada.

# PRÉ-REGISTRO DAS PROJEÇÕES — escrito antes de rodar a Fase 4

Se o resultado divergir, a divergência aparece contra previsão registrada, e não
contra explicação construída depois de ver o número.

## Projeção da v2a nos 21 itens de atribuição

```
19 APROVA · 0 REPROVA · 2 AMBÍGUO (amb-01, amb-04)
```

Item a item, contra o v1:

| item | v1 | v2a previsto | por quê |
|---|---|---|---|
| `amb-01` ×2 | passa | **APROVA** | sem nomes fora do elenco |
| `amb-01` #3 | reprova | **AMBÍGUO** | 2 departamentos na mesma frase (negação) |
| `amb-02` ×2 | passa | **APROVA** | sem nomes fora / todos de dentro |
| `amb-02` #1 | reprova | **APROVA** | Nível 2, declaração governa os 10 |
| `amb-03` ×3 | passa | **APROVA** | sem nomes fora |
| `amb-04` ×2 | passa | **APROVA** | sem nomes fora |
| `amb-04` #3 | reprova | **AMBÍGUO** | 2 em escopo, nada intervém |
| `amb-05` ×3 | passa | **APROVA** | sem nomes fora |
| `amb-06` ×2 | passa | **APROVA** | sem nomes fora |
| `amb-06` #1 | reprova | **APROVA** | 3 nomes de fora, parêntese imediato, os 3 batem |
| `amb-07` ×3 | passa | **APROVA** | sem nomes fora |

**Intervalos de robustez previstos:**

```
categoria     [90,48% ; 100%]     atravessa 95% -> NÃO CONCLUSIVA
condicional   [95,83% ; 100%]     ambas acima   -> ROBUSTA, passa
```

Na variante com desempate anafórico, `amb-04#3` vira APROVA e a categoria fica
`20 APROVA · 0 REPROVA · 1 AMBÍGUO`.

## Projeção do `respaldo_de_citacao` na `amb-02`

Com base no que já foi medido por casamento de palavra — e a classificação em
três só pode **mover casos de "sem tema" para SEM RESPALDO ou INCONCLUSIVO**,
nunca criar COM RESPALDO novo:

| | citados | COM RESPALDO previsto | o resto |
|---|---|---|---|
| `amb-02#1` | 10 | **1** | 9 divididos entre SEM RESPALDO e INCONCLUSIVO |
| `amb-02#3` | 35 | **10** | 25 divididos entre SEM RESPALDO e INCONCLUSIVO |

Previsão adicional, mais arriscada e por isso mais informativa: **a maioria dos
9 e dos 25 cai em SEM RESPALDO, não em INCONCLUSIVO** — as medianas de 166 e de
tamanho semelhante indicam perfis sem campo descritivo. Se muitos caírem em
INCONCLUSIVO, o casamento de palavra é que está grosseiro, e a métrica entra no
texto como limitação em vez de medida.

# ⚠️ O hedge não é avaliado em direção nenhuma

Dois casos opostos, nenhum dos dois capturado por qualquer instrumento deste
projeto.

**Hedge honesto, invisível.** Na `amb-06`, sobre `HENRIQUE VIEIRA DE MENDONCA`,
o agente escreveu que os temas dele *"podem ter conexões indiretas com a
agroecologia"*. Está **certo**: o perfil, lido inteiro (978 caracteres), fala de
tratamento de resíduos, microalgas, bioenergia e wetlands, e não menciona
agroecologia. **O agente sinalizou corretamente a própria incerteza.** Sob v2a a
resposta é impecável — a atribuição departamental está certa. Sob
`respaldo_de_citacao` ele cai em INCONCLUSIVO. Em nenhuma das duas o acerto de
calibração aparece.

**Hedge vazio, também invisível.** Na `amb-02#1` o agente ressalvou que *"nenhum
docente específico do Departamento de Ciências Sociais apareceu"* e ofereceu dez
substitutos — dos quais **1 de 10** tem evidência própria do tema, contra 10 de
35 da `#3` que não ressalvou nada. A resposta que **soa** mais criteriosa é a
**menos ancorada**, e nenhum instrumento registra isso tampouco.

## A conclusão é sobre o escopo do instrumento inteiro

**O sistema não é avaliado pela calibração das próprias ressalvas — nem quando
elas são honestas, nem quando são vazias.** Um agente que hedgeia corretamente e
um que hedgeia para se cobrir recebem a mesma nota, e um que não hedgeia nunca
também.

Isto **não é limitação de uma métrica**; é limitação de escopo de todas elas.
Roteamento mede a ferramenta escolhida; atribuição mede o departamento;
cobertura mede a omissão; respaldo mede a fonte. **Nenhuma mede a relação entre
a confiança expressa e a evidência disponível** — que é, num sistema cujo
princípio 3 proíbe afirmar sem respaldo explícito, provavelmente a competência
mais próxima do que o projeto diz querer.

Vai para o **pré-registro da próxima bateria**, junto com os marcadores de
negação e anáfora. Não é implementado aqui.

# ⚠️ Medição improvisada — CINCO ocorrências, quatro direções

**Cinco vezes nesta fase uma medição minha produziu resultado errado**, e as
cinco entraram em decisão antes de serem conferidas:

| # | onde | direção do erro | o que produziu |
|---|---|---|---|
| 1 | li ~620 dos 898 chars da `amb-04` | leu de menos | previsão pré-registrada errada |
| 2 | medi a `amb-06` por proximidade em texto normalizado | mediu por proximidade | segunda previsão errada |
| 3 | regex de extração capturando prefixo de nome | leu demais | falso alarme do `LEANDRO`, um turno inteiro |
| 4 | dump de perfis truncado em 520 chars | leu de menos | `MARCOS BACIS CEDDIA` quase rotulado errado |
| 5 | casamento de tema sem separar campo descritivo | mediu frouxo | **superestimou a qualidade da `amb-02#3`** |

## A quinta é a mais séria, porque sustentou um argumento

A medição frouxa dava **10 de 35 "com tema próprio"** na `amb-02#3`. A rigorosa —
extraindo só Perfil, Formação e Áreas de interesse, sem o nome do departamento e
sem os campos institucionais — dá **4 COM RESPALDO, 8 INCONCLUSIVO, 23 SEM
RESPALDO**.

Ela **superestimou a qualidade da resposta**, e foi ela que sustentou o argumento
do despejo neste documento. O argumento sobrevive — na verdade fica mais forte,
porque 4 de 35 é pior que 10 de 35 —, mas ele esteve apoiado num número errado
por vários turnos, e o erro foi para o lado que **enfraquecia** a própria tese
que eu estava defendendo. Números corrigidos onde aparecem; o **pré-registro fica
como foi escrito**, porque previsão reescrita depois do resultado não é
previsão — a divergência entre o previsto (10) e o medido (4) é reportada na
Fase 5.

## Isto não é uma lista de descuidos

São cinco erros de **medição**, em quatro direções diferentes — ler de menos,
ler demais, medir por proximidade, medir frouxo —, num trabalho cuja tese é que
o instrumento de medição precisa do mesmo rigor que o objeto medido. É a tese
aplicada a quem a escreve, e ela não passou de primeira em nenhuma das cinco.

Vale registrar o que cada uma tem em comum: **nenhuma deu erro.** Todas
produziram um número plausível, no formato esperado, que só se revelou errado
quando alguém pediu a evidência por trás dele. É exatamente o modo de falha que
o `CLAUDE.md` define como inaceitável para o sistema — e ele apareceu cinco
vezes no aparato que julga o sistema.

Os textos
aqui (respostas de 800 a 2000 caracteres, perfis de até 2400) estão sempre perto
do tamanho em que se corta para caber na tela, e a evidência decisiva tem o
hábito de ficar depois do corte. Nos quatro casos o trecho que mudava a conclusão
estava fora do que eu tinha lido.

> **Regra daqui em diante:** todo script de medição declara, junto do resultado,
> **o tamanho do texto lido e se houve truncamento**. Se o dado for maior que o
> lido, o resultado sai marcado como parcial.

O caso 3 mostra que isso vale também para o inverso: a regex leu texto **demais**
no sentido errado — capturou um pedaço de um nome maior — e produziu um achado
inexistente que consumiu um turno inteiro. Truncar e transbordar são o mesmo
erro: decidir sobre um recorte que não é a coisa.

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
