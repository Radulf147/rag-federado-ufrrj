# Pré-registro — quatro formas de montar a consulta semântica

> Escrito e commitado **antes de rodar**. Nenhuma previsão daqui será reescrita
> depois do resultado; o que errar fica errado, com o motivo do erro anotado
> embaixo.
>
> Data: 14 set 2026. Commit de código no momento da escrita: `adfcc07`.

---

## 1. O achado que motiva

A instrumentação de `adfcc07` gravou, pela primeira vez, **o argumento** que o
agente passa à ferramenta de busca — e não só o nome dela. A bateria de 14 set
(`docs/bateria_consultas.jsonl`) mostrou o seguinte nas 7 perguntas semânticas:

| pergunta do usuário | o que o agente manda ao recuperador |
|---|---|
| *Que professores atuam na área de formação de professores?* | `formação de professores` |
| *Algum professor atua com didática?* | `didática` |
| *Quais docentes pesquisam agroecologia?* | `agroecologia` |

Ele reduz a pergunta ao termo nu, de forma estável nas 3 repetições em 6 das 7.

**A causa raiz é nossa, não do modelo.** O parâmetro se chama
`pergunta_semantica` e a descrição, em `modulo2_inferencia/tools.py`, pede
textualmente *"A pergunta otimizada para buscar no banco de dados vetorial."*
O modelo está obedecendo a uma instrução escrita por nós.

**Por que isso piora a busca neste corpus.** Medido nos 70 documentos
recuperados (7 perguntas × top-10, uma repetição), com o cabeçalho que a
ferramenta cola descontado para comparar só o perfil:

| | RAG puro (pergunta inteira) | agente (termo nu) |
|---|---|---|
| tamanho médio do perfil recuperado | 953 chars | 710 chars |
| perfis esparsos (< 500 chars) | 15/70 — 21% | 33/70 — **47%** |

Um perfil esparso é *nome + departamento + telefone*. Sendo curto, o nome do
departamento domina o vetor dele. A consulta `formação de professores` casa com
`DEPARTAMENTO DE FORMAÇÃO DOCENTE`, que está em **todos** os perfis daquele
departamento — inclusive nos vazios. A pergunta inteira, que carrega *"professores
atuam na área"*, puxa para perfis que descrevem atuação.

O efeito é concentrado onde o nome do departamento carrega o tema:

| pergunta | tema | esparsos: RAG puro → agente |
|---|---|---|
| sem-02 | movimentos sociais | 4/10 → **9/10** |
| sem-03 | formação de professores | 4/10 → **8/10** |
| sem-09 | didática | 3/10 → 6/10 |
| sem-01 | agroecologia | 0/10 → **0/10** |

`sem-01` é o controle natural: **nenhum departamento se chama agroecologia**, e
ali não há efeito nenhum — o agente até recupera perfis maiores (1262 contra
1027 chars). É o item 3 do backlog aparecendo por dentro do item 8.

---

## 2. O que vai ser comparado

Quatro formas de decidir qual texto é embutido pelo recuperador, todas com o
mesmo LLM, mesmo bge-m3, mesmo `TOP_K=10`, mesmo `LIMIAR_DISTANCIA=1.24`.

| | nome | o que embute |
|---|---|---|
| **V0** | controle | o argumento do LLM — exatamente o que roda hoje |
| **V1** | pergunta original | a pergunta do usuário, ignorando o argumento do LLM |
| **V2** | união | as duas, documentos unidos sem repetir, ordenados por distância, cortados em `TOP_K` |
| **V3** | só o schema | o argumento do LLM, mas com a descrição do parâmetro pedindo a pergunta na íntegra |

**V0 roda de novo, junto.** Não se compara o resultado de hoje com o número de
10 set: o Ollama caiu no meio da bateria de 14 set, e comparar execuções de dias
diferentes com infraestrutura instável é como o instrumento engana.

---

## 3. Como vai ser apurado

Reusando a régua que já existe, **sem inventar métrica nova**:

- **Gabarito:** `modulo2_inferencia/medir_recuperacao.py::gabarito()` — pertence
  ao tema quem escreveu a frase **sobre si**, em `texto_descritivo` (Perfil,
  Formação, Áreas de interesse), com o nome do departamento removido. Chave é
  `identidade(meta)`, não o nome, por causa dos homônimos.
- **Nomes afirmados:** `interfaces/comparar.py::nomes_afirmados`.
- **Precisão** = nomes corretos ÷ nomes citados.
- **Cobertura** = nomes corretos ÷ 158 × repetições.

Gabarito por tema, como já apurado em `docs/comparacao_abordagens.md`:

| tema | sem- | docentes |
|---|---|---|
| agroecologia | 01 | 12 |
| movimentos sociais | 02 | 13 |
| formação de professores | 03 | 33 |
| segurança alimentar | 04 | 9 |
| ecologia | 07 | 44 |
| literatura | 08 | 33 |
| didática | 09 | 14 |
| | **total** | **158** |

Duas medidas a mais, **independentes do LLM**, porque são as que a mudança
ataca diretamente e não flutuam com a geração de texto:

- **recall@10 do gabarito** — quantos dos docentes do tema entram no top-10.
- **fração de perfis esparsos** (< 500 chars) no top-10.

7 perguntas × 4 variantes × 3 repetições = **84 execuções do agente**.

### A barra

O alvo declarado é *"o agente melhor que todos"*. Contra os números de 10 set,
isso significa **precisão acima de 72,4% sem cair abaixo de 13,3% de cobertura**.

As duas juntas, e não a precisão sozinha: citar um nome certo e mais nenhum dá
100% de precisão e não é resposta. Qualquer variante que suba a precisão
derrubando a cobertura **não conta como vitória**.

---

## 4. Previsões

Escritas antes de rodar. Cada uma com o que a derruba.

**P1 — V1 recupera exatamente o mesmo conjunto que o `1-vetorial`, nas 7.**
Isso é construção, não aposta: se o texto embutido é o mesmo, o top-10 tem de
ser o mesmo. *Me derruba:* qualquer diferença — e aí é bug, não achado.

**P2 — V1 sobe a precisão acima de 36,0% e mesmo assim NÃO alcança os 72,4% do
RAG puro.** A razão: mesmo com a recuperação idêntica, o agente recebe um
cabeçalho `[NOME — DEPARTAMENTO]` colado em cada bloco, que o RAG puro não
recebe, e na `sem-03` a resposta dele copia esse formato literalmente. Se a
recuperação fosse a história inteira, V1 empataria. *Me derruba:* V1 ≥ 72,4%.

**P3 — V2 terá cobertura maior que V1 e precisão menor que V1.** Mais
documentos distintos, mais nomes citados, mais chance de errar. *Me derruba:*
V2 com precisão ≥ V1, ou cobertura ≤ V1.

**P4 — V3 não vai obedecer sempre.** Em pelo menos 1 das 7 perguntas, em pelo
menos 1 repetição, o argumento continuará diferente da pergunta do usuário.
*Me derruba:* 21/21 com argumento igual à pergunta.

**P5 — o ganho será concentrado em `sem-02`, `sem-03` e `sem-09`, e quase nulo
em `sem-01`.** São os três temas com departamento homônimo; agroecologia não
tem. *Me derruba:* ganho em `sem-01` comparável ao das outras três.

**P6 — nenhuma variante inventará pessoa.** Zero nomes fora do corpus nas 84
execuções, replicando as 150 de 10 set. *Me derruba:* um nome que não exista.

**P7 — nenhuma das quatro variantes atinge a barra do §3** (precisão > 72,4%
com cobertura ≥ 13,3%). A perda tem duas causas empilhadas — a consulta e o
dado contaminado do item 3 — e mexer só na consulta ataca uma. *Me derruba:*
qualquer variante que passe nas duas ao mesmo tempo.

---

## 5. O que este experimento NÃO decide

- **Não mexe no item 3.** O nome do departamento continua dentro do texto
  indexado de cada pessoa. Nenhuma das quatro variantes toca no índice.
- **Não mede as 23 perguntas restantes.** Só as 7 do grupo C. Uma variante que
  ajude aqui pode atrapalhar nas objetivas, e isso **não estará medido** —
  antes de adotar qualquer uma, a bateria das 30 tem de rodar inteira.
- **Não separa a contribuição do cabeçalho** `[NOME — DEPARTAMENTO]`. Se P2 se
  confirmar, esse vira o experimento seguinte.
