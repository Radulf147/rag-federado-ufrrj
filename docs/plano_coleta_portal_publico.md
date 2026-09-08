# Plano de coleta — abas do portal público do SIGAA

**7 set 2026.** Escopo definido pelo orientando; padrões de acesso **medidos**,
não estimados. Cada número abaixo veio de uma requisição real feita neste dia.

Arquitetura em [`arquitetura_multi_entidade.md`](arquitetura_multi_entidade.md).
Este documento é o *o quê* e *quanto custa*; aquele é o *onde mora e como se
busca*.

---

## O que já é coletado hoje, e por que quase não conta

`parte1_scraping_sigaa.py::scrape_sigaa()` **já raspa a tela inicial** — mas
extrai os `<h3>` dos cards e o texto do link de cada um. Ou seja, guarda as
frases do próprio menu:

```
"Docentes: Acesse as páginas públicas dos docentes da UFRRJ"
"Pesquisadores: Consulte quem pesquisa o que na UFRRJ"
```

Tecnicamente é scraping da home. Na prática guarda a **descrição do link**, não
o conteúdo atrás dele. Nenhum dos links é seguido.

---

## Escopo aprovado

| menu | alvo |
|---|---|
| Acadêmico | Departamentos |
| Graduação | Cursos · Estruturas curriculares · Componentes curriculares |
| Extensão | Ações · Programas · Projetos · Cursos · Eventos · Visualizar cursos ou eventos |

---

## Custo medido, por alvo

### Departamentos — ~~2~~ **32 requisições** ✅ IMPLEMENTADO (7 set 2026)

`/sigaa/public/departamento/lista.jsf?aba=p-academico` ·
`modulo1_etl/coletar_departamentos.py`

GET devolve só o formulário: um `<select name="form:programas">` com 16 opções
(a primeira é `value="0"`, `-- TODOS --`) e o botão `form:buscar`.

**POST com `-- TODOS --` devolve os 72 departamentos de uma vez.** Testado — e
foi por isso que este plano dizia 2 requisições.

> ⚠️ **CORREÇÃO — 2 requisições dão os NOMES, e só.**
>
> A listagem de `-- TODOS --` traz **uma única** linha de centro
> (`td.subListagem`, "INSTITUTO DE AGRONOMIA") para os 72 departamentos. O
> parser óbvio — "o centro é o último `subListagem` visto" — atribuiria os 72
> ao Instituto de Agronomia. **Plausível, silencioso, e o total continuaria
> dando 72.**
>
> O vínculo departamento→centro, que a decisão D4 pede, sai de um POST por
> centro: 15 centros, cada um com um GET de formulário antes (estado JSF).
> **32 requisições**, e o POST de `-- TODOS --` vira **controle**.

**O que a coleta encontrou:**

| | |
|---|---|
| departamentos | **72** |
| com centro | 68 |
| **sem centro nenhum** | **4** |

Os quatro sem centro são `PROGRAMAS E PROJETOS DE EXTENSÃO`,
`RELAÇÕES COMUNITÁRIAS E INTERINSTITUCIONAIS`, `ARTE E CULTURA` e
`ESPORTE E LAZER` — unidades administrativas que pendem de pró-reitoria, não de
instituto. Não é defeito: é a estrutura real, e **entrou no banco sem centro em
vez de ser descartada para o esquema ficar limpo**. Foi o controle que a
revelou.

**Uma verificação que desmentiu uma sondagem minha.** Numa sondagem anterior eu
li 22 departamentos no Instituto Multidisciplinar; são **11**. A tabela tem 22
âncoras `<a>`, metade **vazias**, e a contagem crua dobrava tudo. Confirmado por
um terceiro caminho: o corpus de docentes tem exatamente 11 departamentos com
sufixo `/IM`, e **os 11 casam**.

### Cursos (graduação) — 1 requisição

`/sigaa/public/curso/lista.jsf?nivel=G&aba=p-graduacao`

**O GET já traz a listagem**, sem POST: 76 cursos, cada linha com nome, campus,
modalidade, grau acadêmico e link para `portal.jsf?id=NNN&lc=pt_BR&nivel=G`.

### Estruturas curriculares — 76 requisições

⚠️ **No menu, "Estruturas Curriculares" aponta para a mesma URL que "Cursos".**
Não existe página própria: a estrutura fica dentro do portal de cada curso, em
`curriculo.jsf?lc=pt_BR&id=<id_do_curso>`.

O `id` é o mesmo da listagem, então **não é preciso passar pelo portal do
curso** — vai-se direto da listagem para o currículo. 1 GET por curso.

**Opcional, +76:** o portal do curso (`portal.jsf?id=`) traz apresentação,
coordenador, título profissional, área de conhecimento CNPq e modalidade. É
conteúdo real, mas é decisão à parte.

> **Correção de um número que eu dei antes.** Na conversa eu disse "156
> requisições certas". Aquilo somava o portal do curso como se fosse
> obrigatório. Não é — o `id` da listagem já serve. **São 80 certas**, ou 156
> se a apresentação do curso entrar junto.

### Visualizar cursos ou eventos — 1 requisição

`/sigaa/public/extensao/paginaListaPeriodosInscricoesAtividadesPublico.jsf`

**O GET já traz 78 linhas**: "Inscrições Abertas (21)", com título, tipo, prazo
de inscrição e vagas.

⚠️ **Este é dado que vence.** Cai direto na decisão D6 da arquitetura: janela de
validade no metadado, e data visível na resposta. Sem isso o bot afirma como
aberta uma inscrição encerrada.

### Componentes curriculares — volume DESCONHECIDO

`/sigaa/public/componentes/busca_componentes.jsf?nivel=G&aba=p-graduacao`

Formulário: `form:nivel` (10 opções), `form:tipo` (5), **`form:unidades` (173)**,
botão `form:btnBuscarComponentes`.

**Pergunta aberta:** aceita busca ampla (sem escolher unidade), ou exige uma?
Se aceitar, é 1 POST. Se exigir, são 173.

### Extensão (ações, programas, projetos, cursos, eventos) — volume DESCONHECIDO

`/sigaa/public/extensao/consulta_extensao.jsf?acao=N&aba=p-extensao`

O `acao` da URL só pré-seleciona o tipo; é um formulário só:

| `acao` | tipo |
|---|---|
| *(ausente)* | Ações de Extensão (todos) |
| 1 | Programas |
| 2 | Projetos |
| 3 | Cursos |
| 4 | Eventos |
| 6 | Produtos — *fora do escopo aprovado* |

Formulário: `formBuscaAtividade:buscaTipoAcao` (7 opções),
**`formBuscaAtividade:buscaUnidade` (412)**, botão `btBuscar`.

**Mesma pergunta aberta.** Se exigir unidade, 412 × 5 tipos é inviável e o
plano muda de forma.

---

## Resumo

| alvo | requisições | acesso |
|---|---|---|
| Departamentos | **32** ✅ feito | POST por centro, mais o controle |
| Cursos | **1** | GET |
| Estruturas curriculares | **76** | GET |
| Visualizar cursos/eventos | **1** | GET |
| **subtotal certo** | **110** | |
| Componentes curriculares | **?** | POST, falta sondar |
| Extensão (5 tipos) | **?** | POST, falta sondar |
| *(opcional)* apresentação dos cursos | *+76* | GET |

---

## O que ficou de fora, e por qual critério

**Site externo** (critério do orientando — redireciona para fora do SIGAA):
Calendário Acadêmico, Regulamento dos Cursos de Graduação, Outros Processos
Seletivos, SISU, Biblioteca Mobile, BDTD.

**Exige autenticação:** Autenticação de Documentos, Consultar e Renovar
Empréstimos, Área de Inscritos em Cursos, Emitir Certificados.

**Relevância:** Diplomas e Diplomas Digitais (**dado pessoal de ex-aluno** —
fora do escopo e melhor não tocar), acervo e aquisições da biblioteca (volume
alto, e o bot é sobre a universidade e suas pessoas, não sobre livros).

**Existe, é relevante, mas fora deste escopo por decisão de 7 set 2026:**
a aba Pesquisa inteira (Pesquisadores, Bases de Pesquisa, Iniciação Científica,
Laboratórios), Turmas, Centros/Unidades, Programas de Pós-Graduação, Cursos
Abertos, Chefes e Coordenações. Registrado aqui para não se perder.

---

## Ordem de implementação

1. **Departamentos.** 2 requisições. Menor peça útil, e serve de piloto: prova
   o caminho inteiro — coleta, SQLite, embedding, tool — para um tipo de
   entidade que **não é docente**. Se D1 da arquitetura estiver certo, isto
   funciona sem tocar em mais nada.
2. **Cursos + estruturas curriculares.** 77 requisições, só GET, sem
   formulário. Risco baixo, conteúdo denso.
3. **Visualizar cursos/eventos.** 1 requisição, e obriga D6 a existir.
4. **Sondar componentes e extensão**, e só então planejar o volume.

A ordem põe o certo e barato na frente e adia o que ainda é pergunta.

---

## Armadilhas conhecidas, que este plano não pode repetir

**Corrida de sessão (achado 08).** O SIGAA guarda estado no servidor atrelado
ao cookie. Foi isso que fez o scraper de docentes trazer 6 de 15 perfis com a
pessoa errada. Estas páginas são JSF com POST e o mesmo risco: **cliente
isolado por requisição**, e sequências de POST em série.

**Acúmulo entre execuções (achado 10).** Cada execução é um retrato completo,
não um incremento. Apagar antes de escrever, nas duas pontas.

**Codificação.** O SIGAA serve `iso-8859-1`, não UTF-8. `parte1` já trata
(`r.content.decode("iso-8859-1")`); qualquer coletor novo precisa do mesmo, ou
os acentos viram lixo em silêncio.

**`verify=False` em `parte1`.** A função `acessar_pagina` desliga a verificação
de TLS. As sondagens de hoje acessaram as mesmas páginas **sem** isso e
devolveram HTTP 200 — ou seja, é desnecessário. Não copiar para o coletor
novo, e vale remover de lá.

---

## Como saber que não estragou

As réguas existentes são o piso: **recall nos 6 temas** e as **36 respostas**
do teste de coleção. Se qualquer uma piorar, a adição não entra.

E uma pergunta que só o dado novo responde — *"quais disciplinas o curso de
Ciência da Computação tem?"* — que hoje o bot não tem como acertar. Ela precisa
ser escrita **antes** da coleta, junto da previsão.
