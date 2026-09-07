# Pré-registro — trocar a coleção de produção para `rag_sigaa_descritivo`?

**Escrito em 7 set 2026, ANTES de rodar.** O commit deste arquivo antecede o do
resultado; a ordem fica provada no git, não na minha palavra.

Mede-se **a resposta que o usuário lê**, não o recall. São coisas diferentes, e
a segunda já está medida (`docs/backlog_avaliacao.md` item 7): mediana de
recall@10 subiu de **14% para 27%** ao indexar só o texto descritivo.

---

## Por que não basta o 27%

A reindexação tira do índice vetorial quem não escreveu nada sobre si:
**1302 → 746 documentos, 556 docentes fora.**

Esses 556 **nunca estiveram em gabarito nenhum**. O gabarito deste projeto é
casamento literal da frase no perfil; quem não escreveu perfil não tem frase
para casar, e portanto nunca foi contado como acerto perdido.

> Removê-los **só pode subir a métrica** e **só pode piorar a resposta** para
> quem perguntar sobre eles.

Não é uma métrica errada. É uma métrica **cega exatamente no eixo em que a
mudança pode fazer estrago** — o par de cegueiras que o pré-registro da fase 4
antecipou, aparecendo sozinho e num lugar onde eu não o tinha procurado.

Uma medição que só pode confirmar quem a fez não é medição. Por isso o grupo C
existe, e por isso ele é o único grupo que decide.

---

## O que eu já sabia antes de escrever as previsões

Declarado para que não se confunda com resultado. São fatos do corpus,
levantados no dia 7 set antes de qualquer execução do agente:

1. `rag_sigaa` = 1302 docs · `rag_sigaa_descritivo` = 746 docs.
2. Dos **30** docentes dos dois departamentos de computação, **13 (43%)**
   ficaram fora do índice novo.
3. **O perfil do LEANDRO GUIMARAES MARQUES ALVIM está vazio.** O documento
   dele, inteiro, é:

   ```
   Docente: LEANDRO GUIMARAES MARQUES ALVIM.
   Departamento: DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM.
   Telefone: 21981734381  E-mail: alvim.lgm@gmail.com
   ```

   Sem `Perfil`, sem `Formação`, sem `Áreas de interesse`. **Não há nada na
   base dizendo que ele pesquisa IA.** Ele está fora do índice novo.
4. MARCEL WILLIAM ROCHA DA SILVA está na mesma situação.

O ponto 3 corrige a premissa da investigação inteira do item 7. Ela começou com
"a ferramenta não achou o Alvim, que pesquisa IA". A ferramenta não errou a
recuperação: **não havia o que recuperar.** É falha de cobertura de dado, e a
diferença importa — nenhuma mudança na busca conserta um campo em branco.

O 14% → 27% continua valendo. O caso que o motivou, não.

---

## O LLM não é determinístico, e isso já se viu aqui

Os posts 10 e 19 da rede simulada respondem **a mesma pergunta** com listas
diferentes: o 10 traz ADRIA, FILIPE e RONALDO; o 19 traz só FILIPE e RONALDO.
Mesma coleção, mesmo modelo.

Uma execução por pergunta mediria o sorteio, não a coleção. **3 execuções por
pergunta por coleção**, e o que conta é o padrão nas três, não uma resposta
bonita. Divergência entre as três é resultado e vai no relatório.

---

## As perguntas

Escritas antes de rodar. Os grupos A e B saem da rede simulada **literalmente**
— são os textos que já foram publicados lá, não reescritas minhas.

| id | grupo | pergunta | serve para |
|----|-------|----------|------------|
| A1 | controle | quantos professores tem o Departamento de Computação hoje? | caminho estruturado; **não toca no índice vetorial**. Se mudar, algo mais mudou junto e o teste todo é suspeito. |
| A2 | real | quais docentes do Departamento de Ciência da Computação trabalham com inteligência artificial? | post 4 da rede — o caso principal |
| A3 | controle | qual a nota de corte da monitoria de Cálculo 1 esse ano? | post 6 — fora do escopo do SIGAA. A resposta certa é "não sei". Mede se o índice novo aumenta a invenção. |
| B1 | real | Quais docentes de computacao do IM sao de IA? | post 16 — a variante que o orientando digitou |
| C1 | **decide** | o que o professor Leandro Alvim pesquisa? | perfil vazio, **fora do índice novo** |
| C2 | **decide** | quais são as áreas de interesse do professor Marcel William Rocha da Silva? | idem |

6 perguntas × 2 coleções × 3 execuções = **36 respostas**.

### Como cada resposta é classificada

Antes de ver qualquer uma:

- **CERTA** — diz o que a base sustenta, e só isso.
- **ABSTÉM** — diz que não há informação. Em C isto é a resposta **correta**,
  não uma falha.
- **INVENTA** — atribui a alguém uma área que o documento daquela pessoa não
  contém. Em C, é o desastre: o agente vê 10 perfis de outros professores de
  computação e nenhum do perguntado.
- **DESVIA** — não responde e não abstém (pede esclarecimento, lista o
  departamento inteiro).

`INVENTA` em C conta como piora mesmo que a resposta pareça útil.

---

## Previsões (continuam a numeração; as 1–12 estão no item 7 do backlog)

**13. C PIORA na coleção nova.** Perguntar do Alvim lá devolve 10 perfis de
outras pessoas e nenhum dele. Prevejo **pelo menos uma** das 6 execuções de C
na coleção nova classificada como `INVENTA`, contra **zero** na coleção atual.

**14. A2 e B1 melhoram ou empatam** na coleção nova — mais nomes corretos de
IA. É exatamente o que o 27% mede, e seria estranho não aparecer.

**15. O limiar de 1.24 não salva C.** Os perfis de outros docentes de
computação ficam **abaixo** de 1.24 para a consulta sobre o Alvim, porque a
consulta carrega "professor", "pesquisa" e o nome de alguém da computação. O
filtro que existe para cortar o irrelevante não corta o *plausível*.

**16. A1 e A3 não mudam** entre as coleções.

### O que me derruba

> Se C **não** piorar em nenhuma das 6 execuções — nenhum `INVENTA`, nenhuma
> troca de `ABSTÉM` por `DESVIA` —, minha objeção à troca cai, e a coleção
> nova deve ser adotada direto pelos 27%.

Escrito assim de propósito. A previsão 13 é a minha posição, e é ela que está
exposta.

---

## O que este teste NÃO decide

- Não mede qualidade de escrita da resposta, só o que ela afirma.
- Não cobre os outros 543 docentes fora do índice; C são dois casos escolhidos
  por serem de computação e conhecidos do orientando, não uma amostra.
- Não diz nada sobre perguntas que não foram feitas na rede simulada. A medida
  que faltaria — com que frequência alguém pergunta sobre um docente de perfil
  vazio — só sai de uso real.
