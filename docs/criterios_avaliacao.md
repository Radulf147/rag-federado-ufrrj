# Critérios de avaliação — a checagem de subconjunto

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

**Nível 2 — declaração com escopo.** Uma declaração departamental governa os
nomes que a seguem **se e somente se nenhum outro nome de departamento
conhecido aparecer entre a declaração e o nome**.

> *"encontrei diversos docentes vinculados ao 'DEPARTAMENTO DE EDUCAÇÃO DO
> CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE' que trabalham com essa temática:"*
> seguido de dez nomes em lista — a declaração governa os dez, porque nenhum
> outro departamento aparece entre ela e eles.

**Ambiguidade não se resolve por proximidade.** Se mais de um departamento
conhecido ficar em escopo para o mesmo nome, o item é marcado **AMBÍGUO** e vai
para revisão manual. O checker não escolhe o mais próximo. O relatório informa
quantos itens caíram nisso.

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

⚠️ **Há um problema conceitual conhecido na v2b, e ele precisa ser resolvido
antes de qualquer uso sério.** O elenco de uma pergunta ambígua é o
**departamento inteiro** — 44 docentes na `amb-01`, 35 na `amb-02`, 31 na
`amb-06`. Mas a pergunta é *"quem do Departamento de Matemática pesquisa
estatística?"*, e a resposta certa é um subconjunto pequeno. Exigir os 44
reprovaria toda resposta correta. **A v2b como escrita mede a coisa errada em
pergunta ambígua**; ela só faz sentido onde o elenco é a resposta inteira, que é
o caso de `listagem`, não de `subconjunto`. Fica registrado e reportado, não
aplicado como critério.

---

# Previsões registradas antes de codificar

## `amb-04` — passa sob v2a. **Previsão original substituída.**

A previsão anterior dizia que a `amb-04` continuaria reprovada porque "área de
Letras" não é nome de departamento do conjunto fechado. **Ela partiu de revisão
manual truncada minha**: li os primeiros ~620 caracteres da resposta e não li o
parágrafo seguinte, que diz:

> *"Estes docentes estão listados no **Departamento de Letras/IM**, que parece
> ser um departamento afiliado ou parte do mesmo grupo."*

`DEPARTAMENTO DE LETRAS/IM` é o departamento real de `VALERIA ROSITO FERREIRA`,
`CARMEN PIMENTEL` e `ROSEMARY GONCALO AFONSO`, e é o nome de departamento
conhecido mais próximo de cada uma. **A declaração existe, é explícita e está
correta. A v2a absolve.**

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

## `amb-06` — reprova sob v2a, pelo terceiro braço

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

# Verificações registradas

## Nomes curtos — buraco existe, mas é inerte aqui

`nomes_do_corpus()` descarta nomes com 10 caracteres ou menos, para evitar falso
positivo de substring. Isso torna quatro docentes **invisíveis ao detector**:
`AHYAS SISS`, `ANA ARAI`, `JOYCE SATO`, `SOFIA EDER`. O corte só pode **inflar** a
nota, nunca reduzi-la.

**Medido: zero ocorrências dos quatro nomes nas 150 respostas gravadas.** O
buraco não afeta esta repontuação, e afeta v1, v2a e v2b igualmente.

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

# O que fica em aberto para decisão humana

**A `amb-02` não é decidida aqui.** O agente disse que nenhum dos 35 docentes de
Ciências Sociais casou com o tema e ofereceu dez docentes de
`EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE`, corretamente rotulados.
Sob a v2a isso passa, porque todos os dez têm vínculo declarado e correto. Mas
"oferecer substituto de outro departamento" é competência diferente de "não
contaminar o subconjunto", e as opções serão apresentadas no relatório da Fase 5
em vez de resolvidas pelo checker.

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
