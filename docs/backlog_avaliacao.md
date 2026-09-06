# Backlog do instrumento de avaliação

Registrado, não implementado. Cada item existe porque uma limitação foi
encontrada em uso, não porque pareceu uma boa ideia.

## 1. Persistir o contexto recuperado (5 set 2026)

**O que falta.** `interfaces/comparar.py::_gravar` grava o contexto apenas como
tamanho:

```python
linha["contexto"] = f"<{len(r.contexto)} caracteres>"
```

**Por que dói.** Sem o texto, `atribuicao_ok` e `nomes_sem_respaldo` não são
recomputáveis depois do fato. Descoberto ao montar a repontuação determinística
(`interfaces/repontuar.py`): a condicional objetiva pôde ser reavaliada sobre a
bateria gravada, mas o critério de tolerância zero — *todo docente afirmado tem
de aparecer no contexto recuperado* — ficou congelado no valor que a bateria
produziu. **Nenhuma métrica de respaldo é auditável depois do fato.** Se um dia
ela estiver errada, não há como saber sem rodar tudo de novo, e rodar de novo
produz respostas diferentes porque o agente é estocástico.

**O que fazer.** Persistir o contexto, ou os ids dos chunks recuperados mais um
hash do texto de cada um. A segunda forma é mais barata e suficiente: os
documentos vivem no Chroma, e o hash detecta se mudaram entre a bateria e a
auditoria.

**Por que não agora.** A decisão foi tomada durante a correção do checker de
subconjunto sobre a bateria `624c82234acd`. Mudar o formato do registro no meio
disso misturaria duas coisas: a bateria gravada continuaria sem o campo, e a
comparação entre checkers deixaria de ser sobre o checker.

## 2. Cache do gabarito por pergunta (menor)

`Pergunta.verdade()` varre os 1302 registros do SQLite a cada chamada.
`repontuar.py` já mantém um cache local por pergunta; `comparar.py` não. Não é
bloqueio — é desperdício.

## 3. Departamento de nome temático atrai consultas sobre o tema (5 set 2026)

**Hipótese, a testar.** Um departamento cujo NOME contém um tema atrai consultas
sobre esse tema para **todos** os seus docentes, independentemente do que cada um
pesquisa — porque o nome do departamento está embutido no documento indexado de
cada pessoa.

**Evidência que originou a hipótese.** A `amb-02` pergunta por movimentos
sociais; o agente ofereceu dez docentes de
`DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE`. Medido
nos perfis do Chroma, removendo o nome do departamento antes de procurar o tema:
**nove dos dez não mencionam o assunto em lugar nenhum**. São perfis esparsos de
151 a 278 caracteres — nome, departamento, contato — os que o achado 09
resgatou. A recuperação casou com o nome do departamento, não com pesquisa de
ninguém.

**Por que é achado sobre o SISTEMA, não sobre a avaliação.** É o achado 03
(recuperação que não discrimina) sobrevivendo à calibração do limiar em 1.24. A
distância até um documento cujo nome de departamento casa com a consulta é
genuinamente curta — o limiar não tem como separar isso, porque a semelhança é
real. O que é falso é a inferência de que a pessoa pesquisa o tema.

**Teste barato, fora dos 21 itens.** Para 3 ou 4 departamentos de nome temático
— `EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE`,
`AGROTECNOLOGIAS E SUSTENTABILIDADE`, e outros a identificar — consultar o
ChromaDB com o tema contido no próprio nome e medir quantos dos TOP_K são
docentes daquele departamento **com perfil esparso e sem menção própria ao
tema**. É medição do recuperador, não da avaliação.

**Quando.** Depois da Fase 5. Registrado agora para não se perder.

## 4. A corrupção de nomes do gerador sobrevive em pergunta ambígua

Em `amb-04#1` o agente escreveu `LUIZ CARLOS ALVES DE MELO`. O ETL gravou
`LUIS CARLOS ALVES DE MELO`, no metadado e no texto do documento — **a corrupção
é do gerador**.

A regra de cópia literal do `SYSTEM_PROMPT` fechou o defeito em `listagem`
(3/6 → 6/6), mas ele reaparece em pergunta ambígua. E aqui é **invisível**: o
nome corrompido não casa com nenhum dos 1297, então some da detecção em vez de
virar erro. Um checker de atribuição nunca poderá flagrá-lo.

Nesta rodada não mudou veredito — o docente pertence ao departamento perguntado,
então nem seria intruso. Mas o mecanismo é o do achado 07: afirmar um nome que a
fonte não contém.

## 5. Perfis esparsos — linha de investigação própria (5 set 2026)

**Não é limitação da avaliação. É limite do SISTEMA de recuperação.**

Docentes com perfil mínimo — nome, departamento, contato, **mediana de 166
caracteres** — atravessam três achados distintos, sempre pelo mesmo mecanismo:

1. **Colisão com o nome do departamento.** O único texto substantivo no
   documento é o nome do departamento, então a similaridade da consulta é
   dominada por ele. Nove dos dez citados na `amb-02#1` são assim.
2. **Despejo premiado.** São eles que engordam a listagem completa do
   departamento sem custo de precisão: **23 dos 35** da `amb-02#3` são SEM
   RESPALDO — perfil sem campo descritivo nenhum, não há onde procurar o tema.
   Outros 8 são INCONCLUSIVO e 4 COM RESPALDO. *(Corrigido em 5 set 2026: dizia
   "25 dos 35", número da medição frouxa. Ver `relatorio_fase5.md` §10.)*
3. **Evidência ausente.** Não há como o agente afirmar nada sobre a pesquisa
   deles sem violar o princípio 3 — e o sistema afirma mesmo assim, porque o
   documento foi recuperado e o LLM lê o nome do departamento como conteúdo.

A recuperação para docentes com perfil mínimo é um problema **estrutural**: o
embedding de um documento de 166 caracteres cujo único conteúdo é institucional
não pode discriminar tema de pesquisa, porque não há tema de pesquisa no
documento. Nenhum ajuste de limiar resolve isso — a semelhança medida é real, o
que é falso é a inferência.

Direções possíveis, não avaliadas: excluir do índice semântico perfis sem seção
descritiva (mantendo-os no SQLite, onde respondem contagem e listagem
corretamente); ou marcar o documento com um campo de densidade que a tool
devolva junto, para o agente saber que não há base para afirmar tema.

Relacionado aos itens 3 e 4 acima, e ao achado 09, que foi quem resgatou esses
perfis de serem descartados — corretamente, porque nome e departamento são o
que contagem e listagem precisam. O problema é usá-los na rota semântica.
