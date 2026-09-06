# Pré-registro da fase 4 — escrito antes de qualquer bateria nova

**Data: 5 set 2026.** Nenhuma bateria da fase 4 foi executada. Nenhum dado novo
foi olhado. Este documento existe para que os critérios não possam ser escolhidos
depois de ver o resultado — que é o erro que a fase 3 cometeu uma vez e gastou
seis fases de protocolo para não cometer de novo.

> **Regra que governa este arquivo:** o que estiver aqui vale como previsão. O
> que for medido e discordar entra como **divergência registrada**, nunca como
> correção do pré-registro. Previsão reescrita depois do resultado não é
> previsão.

## O que herda da fase 3 e não se discute de novo

| decisão | onde foi fixada |
|---|---|
| Repontuação offline sobre JSONL gravado, sem chamar o LLM | `interfaces/repontuar.py` |
| `checker_sha1` carimbado em toda saída | `interfaces/repontuar.py` |
| Gate de reprodução antes de comparar dois rótulos | `interfaces/repontuar.py` |
| AMBÍGUO como terceiro veredito; incerteza vira LARGURA de intervalo | `docs/criterios_avaliacao.md` |
| Todo script de medição imprime intermediários conferíveis | `docs/relatorio_fase5.md` §10 |
| Um critério que só pode aumentar a nota não é critério | `docs/criterios_avaliacao.md` |

---

# 1. Negação

## O problema

`amb-01#3` ficou AMBÍGUO porque o texto diz que `DIOGENES FERREIRA FILHO`
**não** pertence ao Departamento de Matemática. A regra de escopo enxerga um
departamento na peça do nome e não enxerga que a relação é de exclusão. Hoje isso
cai em AMBÍGUO, que é conservador e honesto — mas é 1 dos 2 ambíguos que impedem
a categoria de atribuição de concluir.

## O que se propõe medir

Marcadores de negação **dentro da peça do nome**, com precedência sobre a
atribuição de escopo:

```
NAO   (com fronteira de palavra, sem casar NAOZINHO/SERTANAO)
NENHUM / NENHUMA
FORA DE / EXCETO / SALVO
NAO PERTENCE / NAO E DO / NAO E DA / NAO ESTA LOTAD
```

Quando um marcador de negação estiver na mesma peça que o nome **e** o
departamento, o departamento negado **sai do escopo** desse nome em vez de entrar
nele.

## O critério, e ele pode reprovar

Esta mudança **não é relaxamento estrito**, e é essa a diferença em relação à
v2a. Um nome cuja única declaração é negativa passa de "declaração encontrada,
ambígua" para **sem declaração alguma** — e sem declaração o veredito é
**REPROVA**, não APROVA.

> **`passa→reprova` é possível aqui.** Registrado antes de medir. Se a bateria
> mostrar que a taxa de aprovação cai, isso é a regra funcionando, não defeito.

**Previsão pré-registrada:** sobre os 48 itens gravados de `624c82234acd`, a
regra de negação resolve **`amb-01#3` de AMBÍGUO para APROVA** (o DIOGENES sai
do escopo do Departamento de Matemática, e o texto declara o departamento real
dele em outra peça) e **não muda nenhum outro item**. Atribuição: 19/21 → 20/21,
com 1 AMBÍGUO restante.

Se algum item hoje APROVA virar REPROVA, a previsão está errada e a divergência
entra no relatório.

## Gold set exigido antes de implementar

Mínimo de **4 casos**, dos quais **≥ 2 REPROVA**:

| | estrutura | esperado |
|---|---|---|
| n1 | negação simples, sem outra declaração | REPROVA |
| n2 | negação de um departamento + declaração positiva de outro | APROVA |
| n3 | negação com o marcador longe do nome (peça diferente) | AMBIGUO |
| n4 | falso positivo léxico — "SERTANÃO", "NAOR" — não pode contar como negação | APROVA |

O `n4` existe porque a regra é de casamento de texto e o modo de falha dela é
casar demais.

---

# 2. Anáfora, agora obrigatória

## O problema

Na fase 3 o desempate anafórico rodou como **variante**, não como veredito
oficial, exatamente porque não estava pré-registrado. Ele resolve `amb-04#3` —
os três nomes (`VALERIA`, `ROSEMARY`, `CARMEN`) precedidos de "estes docentes",
onde a declaração posterior é a que vale.

Rodou uma vez, acertou o previsto, e **não é usado no veredito** por causa da
ordem em que as coisas aconteceram. Isso agora se resolve pela via correta:
pré-registrando antes.

## O que se propõe

`desempate_anaforico=True` passa a ser o **padrão** de `_conferir`, e a variante
sem ele vira o alternativo reportado para sensibilidade — inversão exata do que
vale hoje.

Marcadores (os mesmos da fase 3, sem acréscimo — acrescentar marcador depois de
ver o dado seria calibrar no dado):

```
ESTES DOCENTES · ESTAS DOCENTES · ESTES PROFESSORES
ESTAS PROFESSORAS · ELES · ELAS
```

A busca continua **localizada**: só na janela entre o marcador e a declaração
candidata, nunca em qualquer peça da resposta. Essa localização foi um conserto
feito na fase 3 (`_causa_do_ambiguo`) e entra aqui como requisito, não como
descoberta.

## O critério

**Previsão pré-registrada:** atribuição 20/21 com a negação (item 1) sobe para
**21/21**, zero AMBÍGUO, sobre os 48 gravados. Categoria de atribuição vira um
ponto, não um intervalo.

> ⚠️ **E isso é motivo de desconfiança, não de comemoração.** 21/21 sobre o mesmo
> conjunto de 48 que foi lido item a item durante toda a fase 3 é **nota em dado
> de treino**. Registrado aqui, antes de medir: **a fase 4 não pode fechar
> nenhuma afirmação de qualidade sobre os 48 de `624c82234acd`.** Esses itens
> foram inspecionados demais para servirem de validação de qualquer coisa.
>
> **A afirmação da fase 4 só vale sobre bateria nova**, com `prompt_sha1`
> diferente, sobre perguntas que ninguém leu ainda.

---

# 3. Calibração de ressalva

## O problema, medido na fase 3

`amb-02#1` traz a ressalva *"nenhum docente específico do Departamento de
Ciências Sociais apareceu"* e soa mais cuidadosa que a `amb-02#3`, que despeja o
departamento inteiro. Proporcionalmente é a **pior das duas**: 1 de 10 citações
com respaldo, contra 4 de 35.

**O hedge precede a afirmação menos sustentada.** Nenhum instrumento deste projeto
captura isso, e na fase 3 ele não contou nem a favor nem contra — decisão
deliberada e pré-registrada, para não avaliar uma coisa sem definir antes o que
ela é.

## O que se propõe medir

Uma métrica `calibracao_de_ressalva`, com **três classes**, no mesmo desenho de
`respaldo_de_citacao` (o que não se decide vira classe própria, não vira falha):

| classe | condição |
|---|---|
| `RESSALVA_CALIBRADA` | há ressalva **e** a fração com respaldo das citações que a seguem é baixa |
| `RESSALVA_DESCALIBRADA` | há ressalva **e** a fração com respaldo é alta — cautela desnecessária |
| `SEM_RESSALVA` | não há marcador de ressalva |
| `INCONCLUSIVO` | há ressalva, mas o respaldo das citações é indecidível (todas SEM RESPALDO por perfil vazio) |

**O corte entre "baixa" e "alta" tem de ser fixado ANTES de rodar.** Proposto:
`baixa` = fração com respaldo ≤ ⅓; `alta` = ≥ ⅔; entre os dois, `INCONCLUSIVO`.
Números redondos escolhidos por serem redondos, não por ajustarem a nenhum item.

## Previsão arriscada

> **`amb-02#1` sai `RESSALVA_DESCALIBRADA`?** Não. Ela tem 1 de 10 com respaldo —
> fração baixa. Ela sai **`RESSALVA_CALIBRADA`**, e isso expõe o limite da
> métrica: a ressalva é honesta sobre a ausência de dado, e ainda assim a resposta
> segue afirmando coisas sem respaldo depois dela. **Calibração de ressalva não é
> o mesmo que respaldo das afirmações**, e se as duas coincidirem em toda a
> amostra a métrica é redundante e deve ser descartada.

Critério de descarte, fixado antes: se `calibracao_de_ressalva` concordar com
`respaldo_de_citacao` em **100%** dos itens com ressalva, ela não mede nada novo
e sai do relatório.

## O que NÃO se propõe

Não se propõe premiar ressalva. Um instrumento que dá nota por hedgear ensina o
sistema a hedgear — e a `amb-02#1` é a prova de que hedge e ancoragem são
independentes.

---

# 4. Contexto persistido

## O problema

**Terceira cobrança da mesma lacuna.** `atribuicao_ok` e `nomes_sem_respaldo` —
a verificação de tolerância zero das interpretativas, que é o critério de
encerramento das perguntas interpretativas — **não são recomputáveis**. O JSONL
grava o contexto recuperado apenas como tamanho:

```json
"contexto": "<8254 caracteres>"
```

Sem o texto, não há como reconferir quais nomes tinham respaldo. O 100% da fase 3
é o que a bateria produziu, **não o que alguém conferiu**, e a seção 11 do
relatório da fase 5 marca isso na tabela de fechamento.

## O que se propõe

`interfaces/comparar.py` grava o **texto integral** do contexto recuperado em cada
linha do JSONL, junto com:

```
contexto            texto integral entregue ao LLM
contexto_chars      tamanho (o que existe hoje, mantido)
contexto_docs       lista de (nome_docente, siape, score) recuperados
contexto_truncado   bool — se algo foi cortado antes de entrar no prompt
```

`contexto_docs` importa tanto quanto o texto: é ele que permite distinguir
*"o documento não foi recuperado"* de *"foi recuperado e o LLM não usou"*, que
hoje são indistinguíveis e são coisas muito diferentes.

## Custo, estimado antes

92 citações em 21 itens; contextos de ~6.600 a ~8.300 chars. Para 150 células o
JSONL sai de ~1 MB para talvez ~2 MB. **É barato e devia ter sido feito na fase
3.** Fica registrado que a razão de não estar feito é omissão, não trade-off.

## Consequência para o encerramento da fase 3

O critério das interpretativas foi declarado atendido sobre um número não
auditável. **A fase 4 não pode repetir isso.** Nenhuma métrica cujo insumo não
esteja gravado entra em critério de encerramento da fase 4.

---

# 5. O par de cegueiras — precisão sem recall, recall sem precisão

## A estrutura, e ela é simétrica

| categoria | mede | é cega para |
|---|---|---|
| `precisao_de_atribuicao_departamental` | quem foi citado indevidamente | **quem faltou** |
| `cobertura_de_listagem` | quem faltou | **quem foi citado a mais** |

Cada uma é exatamente cega para o que a outra mede. As duas consequências, e as
duas foram observadas ou construídas:

1. **Precisão sem recall premia o despejo.** A estratégia ótima sob precisão pura
   é listar o departamento inteiro e mais ninguém: zero intrusos, nota perfeita.
   **A `amb-02#3` é essa estratégia executada** — 35 de 35, e no máximo 12 dos 35
   com respaldo próprio para o tema. *3 certos e 1 errado reprova; 35
   indiscriminados aprova.*
2. **Recall sem precisão premia o despejo maior ainda.** Uma resposta que
   listasse a universidade inteira passaria em `cobertura_de_listagem`. É o
   espelho exato do item 1, e está fixado em código no par `k1`/`k2` do gold set.

## Por que a fase 3 não resolveu isso

Tentou-se, e a tentativa morreu por análise, não por preguiça: a "v2b como
recall em atribuição" era **impossível** — a v2a é relaxamento estrito e nenhum
contrapeso pode existir dentro dela. E a "v2b como recall em listagem" era
**no-op**: o ramo de listagem sempre calculou `faltando` e sempre exigiu vazio.

> Registrado, porque a premissa errada foi minha: eu afirmei na Fase 0 que "a
> listagem nunca puniu omissão", generalizando a partir do ramo de atribuição sem
> ler o de listagem. Estava errado, e a afirmação era minha, não do orientando.

## O que se propõe medir na fase 4

Não uma métrica nova que some as duas — **não há denominador comum**, e a `amb-01`
mostra que a precisão pode nem produzir um número. Propõe-se, em vez disso,
**reportar o par sempre junto e nunca sozinho**:

```
atribuicao        APROVA / REPROVA / AMBIGUO
respaldo          [com ; com + inconclusivo] de N
cobertura         quem faltou, quando há elenco fechado
```

com uma regra de redação fixada antes:

> **Nenhuma afirmação de qualidade do agente pode citar uma dessas linhas sem as
> outras.** "O agente acerta 95,83% das objetivas" é verdadeiro e é o fechamento
> da fase 3; "o agente é bom em perguntas ambíguas" não é sustentado por nenhuma
> combinação delas.

## A pergunta que a fase 4 tem de responder, e a fase 3 não respondeu

> **Existe alguma métrica que separe a `amb-01` da `amb-02#3` na direção certa?**

A `amb-01` reconhece um falso positivo da recuperação, exclui em voz alta e nomeia
o departamento real do excluído — aplica o princípio 3 explicitamente. Hoje ela é
**não julgável**. A `amb-02#3` cita 23 pessoas sem respaldo e é **aprovada com
folga**.

Se a fase 4 terminar sem responder isso, a resposta honesta é registrar que a
competência central do projeto continua sem instrumento — não inventar um número
que a cubra.

---

# Ordem de execução, e o que bloqueia o quê

```
4  contexto persistido       →  PRIMEIRO. É insumo das outras e da auditoria
                                das interpretativas. Sem ele, 3 e 5 medem menos.
1  negação                   →  gold set (≥2 REPROVA) antes da implementação
2  anáfora obrigatória       →  depende de 1 estar fechado
3  calibração de ressalva    →  depende de 4 (precisa do respaldo real)
5  par de cegueiras          →  regra de redação; vale desde já
```

**Nada disso roda sobre `624c82234acd` como validação.** Aqueles 48 itens foram
lidos um a um durante toda a fase 3 e servem apenas para **regressão** — verificar
que uma mudança faz o que se previu que faria. Afirmação de qualidade exige
bateria nova, com perguntas não inspecionadas.

# Condições de execução

1. **Um item por vez.** Propor, receber o aval, então implementar.
2. **Gold set antes do código**, para 1 e 2. Sem gold set com poder de reprovar,
   não se implementa.
3. **Registrar toda iteração.** Um gold set que fecha na primeira tentativa e um
   que fecha na sétima não valem o mesmo, e o número de tentativas vai no
   relatório.
4. **Na falha do gold set, classificar antes de consertar:**
   - bug de implementação → conserta-se o código
   - a regra discorda do rótulo → **decisão do orientando**, não conserto
   - a fixture não implementa a condição do rótulo → **decisão do orientando**
     (terceira categoria, descoberta na fase 4 do protocolo anterior no caso `(d)`)

   Nunca ajustar regra e rótulo no mesmo passo.
5. **Nenhum critério é escolhido depois de ver o resultado.** O que este documento
   não previu entra como divergência registrada.
6. **A regra operacional de medição vale como condição de entrada**, na forma
   fixada em `docs/relatorio_fase5.md` §10:
   - todo script de medição imprime, **antes do resultado**, quantos registros
     carregou e **de onde**, quantos casaram, o tamanho do texto lido e se
     truncou;
   - todo comando de teste imprime **a origem do que executou** — o `.Id`
     (digest) da imagem, se houve build, o caminho do código carregado.
     **Não `created_at`:** o BuildKit o herda do cache e ele não distingue duas
     imagens diferentes (nona ocorrência);
   - **resultado sem intermediário conferível não entra em decisão**;
   - **marcador de origem só vale depois de demonstrado discriminante** — mostrar
     que ele muda quando a origem muda, antes de confiar nele.

   Esta condição existe porque nove erros meus nesta fase produziram, todos, saída
   no formato esperado e sem exceção. **A regra não é conselho de higiene: é o que
   separa auditar de projetar**, e auditoria depende de alguém voltar.

   O último item é regra sobre as outras regras, e existe porque a nona ocorrência
   aconteceu **dentro da redação da própria condição** — o marcador escolhido para
   provar a origem do teste não distinguia duas imagens. A regra falhou na
   primeira vez em que foi aplicada, e foi aplicada a si mesma.
