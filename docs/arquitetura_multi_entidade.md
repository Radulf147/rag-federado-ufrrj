# Arquitetura multi-entidade — decisões e o que cada uma exige medir

**7 set 2026.** Documento de decisão, escrito **antes** de qualquer código.
Registra o que foi decidido, por quê, qual a fonte, e — em cada caso — **o que
precisa ser medido para a decisão continuar valendo**.

---

## O problema

O projeto inteiro foi construído sobre uma suposição que nunca foi escrita:
**existe um tipo de entidade, o docente.** A suposição não está num comentário;
está espalhada no código, e é isso que a torna cara de mudar.

Onde ela vazou:

| lugar | como aparece |
|---|---|
| `tools.py::busca_vetorial_sigaa` | monta o cabeçalho de cada documento a partir de `nome_docente` |
| `parte5_carga.py` | `content_type` é sempre `docente_perfil` |
| `medir_recuperacao.py` | `recall()` deduplica por `nome_docente` |
| `medir_hibrido.py::fundir` | a chave do RRF é `nome_docente` |
| `reindexar_descritivo.py` | recorta os campos do **perfil de pessoa** |

Um documento de curso entrando hoje na coleção sairia da tool como
`(nome ausente no metadado)`. **É o achado 02 voltando por outra porta** — o
texto chegando ao LLM sem dizer de quem ou de que é.

A partir de agora a premissa é outra: **a pergunta pode precisar de dado de
qualquer aba**, e o armazenamento tem de tornar a busca viável nas duas
direções — dentro de um tipo, e atravessando tipos.

---

## D1 — Esquema de metadados tipado: identidade **e** rótulo

> **CORRIGIDO EM 7 SET 2026, ANTES DE IMPLEMENTAR.** A primeira redação pedia
> só `rotulo`. Estava certa na direção e **incompleta no campo que a literatura
> lista primeiro**: o identificador. Pior: o próprio parágrafo "por quê" dizia
> *"id, fonte, data, tipo"* — e a lista de campos logo acima **não tinha id**.
> A justificativa estava certa e o esquema não a seguia.
>
> A correção veio de uma pergunta do orientando: *"trocar `nome_docente` por
> `rotulo` foi a melhor decisão depois de analisar a literatura?"*

**Decisão.** Todo documento carrega quatro campos obrigatórios, e os campos
específicos do tipo ao lado:

| campo | serve para | exemplo |
|---|---|---|
| **`id_entidade`** | **identidade estável** — dedupe, chave do RRF, junções do D4 | `docente:1800852`, `departamento:7686` |
| `rotulo` | exibição ao LLM, e só isso | `TN745 — APRENDIZADO DE MÁQUINA` |
| `tipo` | o filtro do D2 | `docente`, `curso`, `componente` |
| `source_url` · `scraped_at` | rastreabilidade e D6 | |

`nome_docente` continua existindo como campo do tipo. Deixa de ser o que a tool
assume, **e deixa de ser identidade**.

### Por que `rotulo` sozinho não bastava

`nome_docente` faz hoje **dois trabalhos diferentes**, e só um deles é
exibição:

| trabalho | onde | precisa de |
|---|---|---|
| dizer ao LLM de quem é o texto | `tools.py:238` | rótulo legível |
| deduplicar e chavear o RRF | `medir_recuperacao.py:225`, `medir_hibrido.py:141` | identidade única |

`rotulo` conserta o primeiro e deixa o segundo como está. E o segundo **já está
errado**:

```python
posicao = {d.meta.get("nome_docente"): i for i, d in enumerate(ranking, 1)}
```

Dicionário chaveado por nome: dois documentos com o mesmo nome, e o segundo
**apaga** o primeiro. `FERNANDA SILVA FERREIRA CHAER` colide hoje (item 10
deste backlog: dois SIAPEs, dois departamentos, a duplicação é da fonte), então
a posição medida dela é, em silêncio, a pior das duas.

**Hoje o efeito numérico é 1 em 1302 — desprezível, e não é o argumento.** O
argumento é que os tipos novos pioram o mecanismo: na listagem de cursos,
`CIÊNCIAS BIOLÓGICAS` aparece **duas vezes**, mesmo campus, uma Bacharelado
(id 1990463) e outra Licenciatura (id 450595). Chaveados por rótulo, viram um.

### Por que o `id` do próprio Haystack não serve

`Document.id` é hash do conteúdo. O mesmo docente tem **três ids diferentes**
nas três coleções que existem hoje:

```
FILIPE BRAIDA DO CARMO
  rag_sigaa             df97b2dd...
  rag_sigaa_descritivo  b72a1e6d...
  rag_sigaa_filtrado    0ecc19f4...
```

Reindexar troca a identidade — e reindexamos duas vezes em 7 set. Uma
identidade que não sobrevive a uma decisão de indexação não é identidade.

O `id_entidade` sai do SIAPE (docente) ou do `id` do SIGAA (departamento,
curso, componente). Já são coletados, e **não mudam quando reindexamos**.

### O que exige medir

Nada de recuperação: é refatoração de forma, e a bateria de regressão cobre o
comportamento atual. Mas exige **um teste novo**: dois documentos de mesmo
rótulo e ids distintos têm de sobreviver aos dois como entradas separadas em
`recall()` e em `fundir()`. É o defeito acima, e teste que não distingue o que
mediu não é teste.

### O limite da correção

`rotulo` continua sendo escolha de implementação, não achado de literatura.
Nenhuma fonte diz "acrescente um campo rótulo". O que é respaldado é o esquema
tipado com campos obrigatórios — e daí sai o `id_entidade`, não o `rotulo`.
Fica registrado para não virar autoridade emprestada.

---

## D2 — Uma coleção, com filtro por tipo no metadado

**Decisão.** Tudo na mesma coleção do Chroma. O tipo é metadado, e vira filtro
quando a pergunta implica um tipo.

**Por quê.** Coleções separadas são o padrão para **modalidades** diferentes
(grafo, relacional, texto), não para tipos de entidade do mesmo tipo de dado.
Separar por tipo exigiria que o agente soubesse a coleção antes de buscar — ou
seja, resolver o roteamento antes de ter a informação que decide o roteamento.

**⚠️ O QUE EXIGE MEDIR, E NÃO É OPCIONAL.** A literatura registra que
**pré-filtro estreito degrada o recall**: o índice foi construído sobre o
corpus inteiro, e um filtro seletivo pode deixar candidatos de menos nas
regiões que a busca aproximada explora.

Aqui isso é concreto: 76 cursos num índice de ~2500 documentos é um filtro de
3%. **Antes de confiar no filtro por tipo, medir o recall com e sem ele sobre o
mesmo conjunto de perguntas.** Se degradar, a saída é buscar sem filtro e
filtrar depois — mais caro, e correto.

---

## D3 — Prefixo discriminante antes de vetorizar

**Decisão.** Cada documento recebe, antes do embedding, uma frase curta que diz
o que ele é. Não uma etiqueta genérica: uma frase que **só aquele documento
teria**.

**Por quê, e por que isso não contradiz a reindexação.** O padrão publicado
(*Contextual Retrieval*, Anthropic) mede 35% menos falha de recuperação com
contexto prefixado, 49% combinado com BM25. Nossa reindexação fez o oposto —
**tirou** texto e melhorou de 14% para 27%.

Não são opostos. O eixo não é *quantidade* de contexto, é **discriminância**:

| texto | efeito | por quê |
|---|---|---|
| `DEPARTAMENTO DE FORMAÇÃO DOCENTE`, repetido em 40 perfis | **atrapalha** | não distingue ninguém dos 40 |
| `Curso de graduação em Agronomia, Instituto de Agronomia, Seropédica` | **ajuda** | só um documento tem |

O item 7 deste backlog e o padrão publicado dizem a mesma coisa por ângulos
opostos. O critério operacional que sai daí: **contexto que aparece em muitos
documentos sai; contexto que aparece em um fica.**

**O que exige medir.** O recall nos 6 temas, antes e depois do prefixo, sobre o
corpus de docentes — onde já existe régua. Se o prefixo piorar ali, ele não
entra para os tipos novos.

---

## D4 — Chaves explícitas no SQLite; o salto entre entidades é do agente

**Decisão.** Cada entidade guarda a chave da entidade a que se liga
(`curso -> departamento`, `componente -> unidade`, `acao_extensao -> unidade`,
`docente -> departamento`). O encadeamento entre tipos é feito pelo tool
calling iterativo que já existe.

**A chave é o `id_entidade` do D1, não o nome.** Ligar por nome funcionaria em
66 dos 67 departamentos hoje — e é exatamente o tipo de acerto que esconde o
caso que falha. `DEPARTAMENTO DE COMPUTAÇÃO` e
`DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM` são unidades distintas com nomes
que o LLM já confundiu em produção (posts 9 e 17 da rede simulada); o nome é
bom para exibir e ruim para juntar.

**Por quê.** A pergunta que este projeto passa a poder receber é multi-salto:

> *"Quais docentes do DCC/IM pesquisam IA e também dão disciplina da área?"*

A resposta está espalhada entre perfil e componente curricular, e **busca por
similaridade pura é documentadamente insuficiente** para esse formato. As duas
saídas na literatura são grafo de conhecimento ou recuperação agêntica em
vários passos — e a segunda já está implementada aqui (`MAX_RODADAS_TOOL`,
tool calling iterativo, corrigido em set/2026 justamente para o agente poder
usar o resultado de uma tool para decidir a próxima).

**O que exige medir.** Perguntas multi-salto não têm régua neste projeto. Antes
de afirmar que o agente as resolve, é preciso um conjunto pré-registrado delas
— e ele não existe. **Até lá, isto é uma decisão de arquitetura, não uma
capacidade demonstrada.**

---

## D5 — NÃO construir grafo de conhecimento

**Decisão.** Não haverá Neo4j nem construção de grafo.

**Por quê.** É a peça mais cara da lista, e substituiria um mecanismo que já
existe e já foi corrigido (D4). Adotá-la agora seria trocar código que funciona
por código a escrever, com a justificativa de um ganho não medido neste corpus.

**Quando reabrir.** Se o conjunto multi-salto de D4 for construído e o agente
iterativo falhar nele de forma sistemática. Aí a comparação passa a ter um
número dos dois lados.

---

## D6 — Dado que vence carrega janela de validade

**Decisão.** Documento cujo conteúdo expira — inscrições abertas de extensão,
turmas do período — carrega a data no metadado, e a resposta a exibe.

**Por quê.** Sem isso o bot afirma como atual uma inscrição encerrada há três
meses, e não tem como saber que está errado. É o mesmo defeito dos "dois
zeros" já corrigido em `db_manager` — ausência de dado e dado vencido não são
a mesma coisa que dado válido.

**O que exige medir.** Nada a medir; é uma guarda. Mas precisa de teste de
regressão: pergunta sobre inscrição vencida deve produzir a data, não a
omissão.

---

## D7 — Self-query fica ADIADO, e o motivo é uma medição nossa

**Decisão.** O padrão *self-query retriever* — o LLM extrair da pergunta os
filtros de metadado — **não entra agora.**

**Por quê.** É o padrão recomendado para exatamente o problema de D2. Mas o
teste de 7 set 2026 (item 8 do backlog) mediu que **o LLM, neste projeto, é um
mau planejador de consulta**: com a pergunta do usuário inteira o
`FILIPE BRAIDA` vem em **posição 1**; com a reescrita que o próprio agente
emitiu, ele **sai do TOP_10**.

Adotar self-query agora é dar **mais** responsabilidade de recuperação a um
componente que acabamos de medir errando. A ordem correta é: consertar o item
8, remedir, e só então avaliar self-query — com previsão escrita antes.

**Registro de honestidade:** este é o único ponto em que a literatura
recomenda algo e este documento recusa. A recusa está apoiada em medição
própria, não em preferência, e cai no dia em que o item 8 for corrigido e a
medição refeita.

---

## O mesmo vale para híbrido, e não é contradição

O item 7 mediu RRF **piorando** (49 contra 50 na Parte A, 23 contra 24 na
Parte B). A literatura recomenda híbrido com força.

Não se contradizem: o número publicado é *contextual embeddings* + *contextual
BM25* — os dois lados enriquecidos por D3 —, e o que medimos foi BM25 cru com
denso cru. **São experimentos diferentes.** Depois de D3 implementado, o
híbrido merece ser remedido; antes disso, o resultado atual permanece o que
vale.

---

## Ordem de implementação

1. **D1** — refatoração de forma, sem risco, destrava todo o resto.
2. **D6** — guarda barata, entra junto com o primeiro tipo que vence.
3. **D3** — prefixo, medido contra a régua de docentes antes de valer para os tipos novos.
4. **D2** — coleção única com filtro, **com a medição de degradação do filtro**.
5. **D4** — chaves explícitas, e o conjunto multi-salto pré-registrado.
6. **D7** — reavaliar depois do item 8.

---

## O que este documento NÃO decide

- **Quais abas coletar.** Isso está no planejamento de coleta, e depende de
  duas sondagens ainda não feitas (componentes e extensão aceitam busca ampla,
  ou exigem escolher unidade?).
- **Se os tipos novos ajudam.** Nenhuma medida foi feita com eles. As réguas de
  hoje — recall nos 6 temas e as 36 respostas — são o piso: **se qualquer uma
  piorar, a adição não entra.**
- **Como o LLM deve apresentar tipos diferentes** na mesma resposta.

---

## Fontes

- [Contextual Retrieval in AI Systems — Anthropic](https://www.anthropic.com/engineering/contextual-retrieval)
- [ER-RAG: Enhance RAG with ER-Based Unified Modeling of Heterogeneous Data Sources](https://arxiv.org/pdf/2504.06271)
- [Develop a RAG Solution on Azure — Information-Retrieval Phase (Microsoft Learn)](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-information-retrieval)
- [RAG Metadata Filtering: Four Strategies for Production](https://mudassirkhan.me/blog/rag-metadata-filtering-strategies)
- [Enhancing RAG Performance with Metadata: Self-Query Retrievers](https://medium.com/@lorevanoudenhove/enhancing-rag-performance-with-metadata-the-power-of-self-query-retrievers-e29d4eecdb73)
- [PolyUQuest: Verifiable Structure-Aware Web RAG over Heterogeneous Graphs](https://arxiv.org/pdf/2607.08269)
- [How to improve multi-hop reasoning with knowledge graphs and LLMs — Neo4j](https://neo4j.com/blog/genai/knowledge-graph-llm-multi-hop-reasoning/)
- [Agent-Orchestrated Adaptive RAG: A Comparative Study on Structured and Multi-Hop Retrieval](https://arxiv.org/html/2606.05658v1)
