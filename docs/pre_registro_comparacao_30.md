# Pré-registro — a comparação das três abordagens nas 30 perguntas

**Escrito em 9 set 2026, ANTES de rodar e ANTES de apurar qualquer grupo novo.**
O commit deste arquivo antecede o do resultado, e a ordem está no histórico do
git.

---

## Por que este documento existe

O orientador leu o relato da fase 3 e respondeu:

> *"Uma questão importante para a apresentação seria fazer uma comparação entre
> diferentes abordagens. Pelo resultado que você relatou há apenas uma abordagem
> implementada."*

Ele está certo **sobre o relato**, não sobre o projeto. As três abordagens
existem desde 5 set e rodaram nas mesmas 30 perguntas
(`modulo2_inferencia/pipelines.py`):

| | o que é |
|---|---|
| `1-vetorial` | RAG clássico: busca semântica pura, sem tool calling |
| `2-estruturado` | consulta determinística ao SQLite, sem vetor |
| `3-agente` | agente com tool calling, decide entre os dois |

O que nunca foi feito é **apurar a comparação**. Este documento fixa como ela
será apurada, antes de os números existirem.

---

## O que já está medido, e não se move

Apurado em 9 set a partir de `docs/avaliacao_fase3.jsonl` (commitado em 5 set),
somando o campo `verdade.ok`, que já era gravado para as três pipelines:

| tipo | 1-vetorial | 2-estruturado | 3-agente |
|---|---|---|---|
| contagem | 0/6 | 4/6 | 18/18 |
| listagem | 0/2 | 2/2 | 6/6 |
| vínculo | 0/1 | 0/1 | 3/3 |
| subconjunto | 5/7 | 7/7 | 17/21 |
| **total** | **5/16 — 31,3%** | **13/16 — 81,3%** | **44/48 — 91,7%** |

O 91,7% do agente coincide com o valor já publicado em
`docs/avaliacao_fase3.md`. **Esta tabela é régua, não hipótese**, e nada abaixo
pode alterá-la.

---

## As 30 perguntas são QUATRO grupos, não três

Ao separar as 14 restantes, duas delas não eram o que o rótulo de rota sugeria.
A `sem-05` e a `sem-06` não perguntam "quem pesquisa X" — perguntam o que há no
perfil de **uma pessoa**. E os dois perfis são opostos:

    FILIPE BRAIDA DO CARMO ......... 1164 caracteres, formação acadêmica escrita
    MARCEL WILLIAM ROCHA DA SILVA ..  216 caracteres, só Lattes/sala/telefone/e-mail

A `sem-06` pergunta as **áreas de interesse do Marcel**, e o perfil dele não tem
nenhuma. **A resposta certa para a sem-06 é dizer que o SIGAA não traz isso** —
o mesmo critério das perguntas do tipo `nenhuma`, e não o de precisão.

| grupo | n | perguntas | critério |
|---|---|---|---|
| **A** | 16 | est-*, amb-* | resposta exata — **já medido** |
| **B** | 6 | nen-01..05, **sem-06** | recusou ou inventou? |
| **C** | 7 | sem-01, 02, 03, 04, 07, 08, 09 | precisão dos nomes citados |
| **D** | 1 | sem-05 | a resposta bate com o perfil? |

E, atravessando os quatro:

| grupo | n | critério |
|---|---|---|
| **E** | 30 | nenhum nome afirmado fora do contexto recuperado |

---

## ⚠️ A assimetria que tornaria o grupo E uma fraude, declarada antes

`modulo2_inferencia/pipelines.py:167` — o pipeline estruturado grava
**`contexto=resposta`**:

```python
return ResultadoPipeline(
    pipeline="2-estruturado",
    resposta=resposta,
    contexto=resposta,     # <-- o contexto E a propria resposta
)
```

Não é descuido: ele não tem LLM. A resposta é uma string montada a partir do
resultado do SQL, então "afirmou algo fora do contexto" **é impossível por
construção**. Ele tiraria **100% no grupo E sem que isso significasse nada.**

**Regra fixada agora:** o grupo E é reportado **apenas para `1-vetorial` e
`3-agente`**, e a célula do `2-estruturado` sai da tabela marcada como
*não aplicável — o contexto é a própria resposta*. Publicar 100% ali seria
exibir uma vitória fabricada pela instrumentação.

⚠️ Isto também significa que o grupo E **não é uma comparação das três
abordagens**. É uma comparação de duas. O critério só existe onde há um LLM
escrevendo prosa a partir de um contexto.

---

## As regras, com os casos difíceis decididos antes

### Grupo B — recusou ou inventou? (6 perguntas)

**CERTO** = a resposta declara não ter a informação **e não fornece um valor
para o que foi perguntado**.
**ERRADO** = fornece um valor, **ou** responde uma pergunta diferente da feita.

Casos difíceis, decididos agora:

1. **Recusa + encaminhamento** — *"Não encontrei no SIGAA. Verifique com a
   secretaria."* → **CERTO.** Encaminhar não é afirmar.
2. **Recusa + despejo de dado irrelevante** — recusa o que foi perguntado mas
   devolve outra coisa junto (ex.: pedem salário, ele lista os 44 docentes do
   departamento) → **ERRADO.** Responder outra pergunta não é recusar.
3. **Recusa genérica de escopo** — *"este pipeline só responde a perguntas
   objetivas sobre departamentos"* → **CERTO** pela regra, e **marcado**. Ver a
   ressalva abaixo.
4. **sem-06 (Marcel)** — listar áreas de interesse → **ERRADO**, porque o perfil
   não tem nenhuma. Dizer que o perfil não traz → **CERTO**.

⚠️ **Ressalva obrigatória no relatório.** O `2-estruturado` recusa por não saber
fazer outra coisa, não por avaliar que não tem o dado. É o limite já declarado
em `docs/criterios_avaliacao.md` — *"um agente que responda sempre 'não
encontrei' passa"*. O número dele no grupo B **tem de vir acompanhado da
contagem de recusas genéricas**, senão a tabela sugere discernimento onde há
incapacidade.

### Grupo C — precisão dos nomes citados (7 perguntas)

**Gabarito:** um docente pertence ao tema se o texto do perfil dele **contém a
frase do tema**, exatamente como
`modulo2_inferencia/medir_recuperacao.py::gabarito()` já define. Consultado no
ChromaDB, coleção `rag_sigaa`.

**Precisão** = (nomes citados que estão no gabarito) / (nomes citados).
**Cobertura** = (nomes do gabarito que foram citados) / (tamanho do gabarito).

Ambas reportadas. Precisão sozinha premia quem cita um nome só; cobertura
sozinha premia quem lista o departamento inteiro.

Casos difíceis, decididos agora:

1. **A ressalva salva?** Resposta que cita a pessoa **e admite na mesma frase
   que o perfil não menciona o tema** — o padrão que o item 11 do backlog
   encontrou no `amb-06` — **conta como ERRO de precisão**, e é contado à parte
   como `citou_com_ressalva`. Motivo: o nome citado é o que o leitor leva. A
   contagem separada existe porque ressalvar é melhor que não ressalvar, e a
   tabela não deve apagar essa diferença.
2. **Nome que não existe no corpus** → erro de precisão, e contado à parte como
   `nome_inexistente`. É falha mais grave que citar a pessoa errada.
3. **Zero nomes citados** (recusou) → precisão **não definida**, fica fora do
   denominador da precisão e conta como cobertura 0. Reportado ao lado, porque
   recusar tudo daria precisão perfeita.
4. **Sinônimos: NÃO haverá lista.** O gabarito é a frase exata do tema, e ponto.
   Ver a limitação declarada abaixo.

### Grupo D — a resposta bate com o perfil? (1 pergunta, sem-05)

**CERTO** = toda afirmação sobre a formação de Filipe Braida está no texto do
perfil dele. **ERRADO** = qualquer dado de formação que não esteja lá.

Uma pergunta só. **Não sustenta percentual** e será reportada como caso narrado,
não como taxa.

### Grupo E — sem afirmação fora do contexto (30 perguntas, 2 pipelines)

**CERTO** = todo nome de docente afirmado na resposta aparece no texto que o
pipeline entregou ao LLM.

Casos difíceis, decididos agora:

1. **Nome que veio na própria pergunta** (*"a formação de Filipe Braida"*) →
   **não conta** como afirmação sem respaldo. Repetir o que o usuário escreveu
   não é inventar.
2. **Comparação sem acento e sem caixa**, com a mesma normalização que
   `interfaces/comparar.py::_normalizar` já usa. Nome partido por quebra de
   linha no contexto conta como presente.
3. **Resposta sem nome nenhum** → CERTO, vacuamente, e contado à parte. Sem isso
   a recusa vira nota alta.

---

## Previsões

**23.** No grupo C, o `3-agente` **não** será claramente melhor que o
`1-vetorial` em precisão — os dois usam a mesma busca semântica, e a vantagem do
agente está no roteamento, não na metade interpretativa. Barra: diferença menor
que 15 pontos percentuais.

**24.** No grupo B, o `2-estruturado` terá nota **alta** (≥ 4 de 6) e **quase
toda ela vinda de recusa genérica**, não de discernimento. É a previsão que mais
importa: se ela se confirmar, a tabela sozinha enganaria o leitor.

**25.** Na `sem-06` (Marcel, perfil sem áreas de interesse), **pelo menos uma**
das três abordagens inventa áreas de interesse.

**26.** No grupo E, o `3-agente` fica **acima** do `1-vetorial` — as ferramentas
devolvem registros delimitados, e os chunks do vetorial misturam vários docentes
no mesmo texto, o que dá mais margem para o LLM atribuir errado.

### O que me derruba

> Se a **23** falhar e o agente for muito melhor em precisão interpretativa,
> minha leitura de que a vantagem dele é só roteamento está errada, e a tese da
> arquitetura é mais forte do que eu venho dizendo.
>
> Se a **24** falhar e o `2-estruturado` for mal no grupo B, então a recusa
> genérica dele não estava passando por discernimento, e a ressalva que estou
> exigindo é desnecessária.

---

## ⚠️ Fraquezas deste pré-registro, declaradas

1. **Eu já vi algumas respostas.** Ao investigar o pedido do orientador, li as
   respostas das três pipelines para a **sem-01** e para a **nen-04**. O
   pré-registro dessas duas é mais fraco que o das outras 28, e elas serão
   marcadas na apuração. Não vou fingir que não vi.
2. **O gabarito do grupo C é um substituto.** "Escreveu a palavra no perfil" não
   é "pesquisa o tema". Erra nos dois sentidos: exclui quem pesquisa e não
   escreveu, inclui quem escreveu de passagem. É a mesma limitação que os
   números de recall@10 já carregam, e a decisão de não criar lista de sinônimos
   é deliberada — escolher sinônimos depois de ver o placar seria escolher os
   que ajudam.
3. **Juiz único e nenhuma medida de concordância.** Não há segundo avaliador
   antes de 17 set.
4. **O grupo D tem uma pergunta.** Não vira taxa.

---

## O que esta apuração NÃO decide

- **Não diz que uma abordagem é melhor "no geral".** Diz em qual grupo de
  pergunta cada uma vence, que é uma afirmação diferente e mais fraca.
- **Não mede utilidade da resposta.** Uma resposta pode passar em C e E e ser
  inútil para quem perguntou.
- **Não vale para as abas novas.** Curso, componente e extensão mudam o conjunto
  de perguntas, e este pré-registro morre quando isso acontecer.

---

## Ordem de execução, e por que ela é essa

1. **Este arquivo, commitado.** ← e nenhum número novo antes disso
2. Corrigir `interfaces/comparar.py::_gravar` para persistir o texto do
   contexto — hoje ele grava `f"<{len(r.contexto)} caracteres>"`, que é o item 1
   do backlog e a razão de o grupo E não ser auditável nos dados de 5 set
3. Rodar as 30 nas três pipelines
4. Apurar A (já feito), B, C, D, E e montar a tabela única

O passo 2 antes do 3 é o que impede a rodada de nascer já sem o dado que o grupo
E precisa — que é exatamente o que aconteceu em 5 set.
