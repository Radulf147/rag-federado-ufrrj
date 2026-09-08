# Pré-registro — o roteamento com CINCO ferramentas

**Escrito em 8 set 2026, ANTES de rodar.** O commit deste arquivo e do
`interfaces/bateria_roteamento.py` antecede o do resultado.

---

## Por que esta bateria precisa existir

O tipo `departamento` entrou no registro em `273ac61`. O agente passou de **três**
ferramentas para **cinco**, e duas delas aceitam a mesma coisa e devolvem
coisas diferentes:

| ferramenta | recebe | devolve |
|---|---|---|
| `buscar_docentes_por_departamento` | nome de departamento | quantos docentes, e quem |
| `buscar_departamento_por_nome` | nome de departamento | a que instituto pertence |

É a **primeira colisão semântica entre ferramentas** deste projeto.

## ⚠️ A métrica existente não consegue ver isso

`interfaces/comparar.py` deduz a rota da **fonte de dado tocada**, não da
ferramenta chamada:

```python
ROTA_POR_FONTES = {
    frozenset({"sqlite"}):   "estruturada",
    frozenset({"chromadb"}): "semantica",
    frozenset({"sqlite", "chromadb"}): "ambigua",
    frozenset(): "nenhuma",
}
```

As duas ferramentas de departamento leem o **mesmo SQLite**. Chamar a errada
produz `frozenset({"sqlite"})` e a métrica anota **"estruturada" = correto**.

A acurácia de 97,8% continuaria alta com o agente respondendo a coisa errada.
Não é uma métrica ruim: ela mede a tese da arquitetura — objetivo vs.
interpretativo — e faz isso certo. **É cega no eixo exato em que a mudança de
ontem pode estragar**, que é o par de cegueiras do pré-registro da fase 4
aparecendo pela terceira vez neste projeto.

**Por isso a métrica nova é acurácia por FERRAMENTA**, medida ao lado da
acurácia por rota. Sem ela a bateria não teria como me contrariar.

---

## Duas partes, e só uma compara com o passado

| | perguntas | serve para |
|---|---|---|
| **A** | as **mesmas 30** do `conjunto_avaliacao.py` | comparável ao 97,8% — mede se PIOROU |
| **B** | **10 novas** sobre departamento e instituto | mede se o tipo novo é alcançado |

Só a parte A entra na comparação com o número antigo. Juntar as duas produziria
uma linha de base que nunca existiu, e o número novo pareceria igualmente
válido.

---

## ⚠️ A fraqueza deste pré-registro, declarada

Os rótulos de **rota** das 30 perguntas foram escritos e commitados **antes de
qualquer execução** — é isso que dá validade ao 97,8%. Os rótulos de
**ferramenta** estão sendo escritos agora, depois de meses vendo o sistema
rodar. **É pré-registro mais fraco, e não adianta fingir que não é.**

O que limita o estrago: o rótulo de ferramenta é **derivado mecanicamente** do
rótulo de rota que já existe, não escolhido de novo.

```
estruturada + contagem/listagem de docentes  ->  buscar_docentes_por_departamento
estruturada + vínculo de UMA pessoa          ->  buscar_docente_por_nome
semantica                                    ->  busca_vetorial_sigaa
ambigua                                      ->  as duas acima, juntas
nenhuma                                      ->  nenhuma
```

Aplicando isso às 30: **oito** caem em contagem/listagem, **uma** (est-07) em
vínculo de pessoa, nove em semântica, sete em ambígua, cinco em nenhuma.
**Zero exigiram julgamento meu.** Se alguma tivesse exigido, entraria marcada e
somaria num total à parte.

---

## As perguntas da parte B

Escritas antes de rodar, e cada uma existe por um motivo declarado.

| id | pergunta | ferramenta esperada | por que existe |
|----|----------|--------------------|-----------------|
| dep-01 | Quais departamentos fazem parte do Instituto Multidisciplinar? | `..._por_centro` | o caso central do tipo novo |
| dep-02 | Que departamentos tem o Instituto de Ciências Exatas? | `..._por_centro` | idem, outro instituto |
| dep-03 | A que instituto pertence o Departamento de Ciência da Computação? | `..._por_nome` | o vínculo, direção inversa |
| dep-04 | O Departamento de Matemática fica em qual instituto? | `..._por_nome` | mesma coisa, redação diferente |
| dep-05 | Quantos departamentos tem o Instituto de Veterinária? | `..._por_centro` | contagem, não listagem |
| dep-06 | O Departamento de Arte e Cultura pertence a qual instituto? | `..._por_nome` | **um dos 4 SEM centro** — a resposta certa é dizer que não há |
| dep-07 | Quantos docentes tem o Departamento de Ciência da Computação? | `buscar_docentes_por_departamento` | **O TESTE DA COLISÃO.** Cita um departamento e pergunta de docentes |
| dep-08 | A que instituto pertence o departamento onde trabalha o Filipe Braida? | `buscar_docente_por_nome` **+** `..._por_nome` | **multi-salto entre tipos** (D4), nunca medido |
| dep-09 | Liste os departamentos do Instituto de Educação. | `..._por_centro` | listagem explícita |
| dep-10 | Quantos institutos tem a UFRRJ? | **nenhuma** | controle: pergunta legítima sem ferramenta que a responda |

O **dep-07** é o que decide a previsão 20. O **dep-08** é o primeiro multi-salto
medido neste projeto.

---

## Previsões

**19.** A acurácia por **rota** na parte A fica **dentro do ruído** do 97,8% —
justamente porque a métrica não distingue as duas ferramentas estruturadas.
Barra: não cai abaixo de 93%.

**20.** A acurácia por **ferramenta** revela **pelo menos uma** confusão entre
`buscar_departamento_por_nome` e `buscar_docentes_por_departamento`, nas 90
execuções da parte A ou no dep-07.

**21.** Na parte B, `buscar_departamentos_por_centro` é escolhida corretamente
com **mais frequência** que `buscar_departamento_por_nome` — o nome de um
instituto é sinal mais forte que o de um departamento, que agora é ambíguo
entre duas ferramentas.

**22.** O **dep-08** (multi-salto) **falha** na maioria das execuções: exige
encadear duas ferramentas estruturadas, e o `SYSTEM_PROMPT` só descreve o
encadeamento estruturado → semântico.

### O que me derruba

> Se a previsão 20 **não** se confirmar — zero confusão em 90 execuções da
> parte A mais 3 do dep-07 —, minha objeção sobre colisão semântica cai, as
> descrições escritas ontem bastam, e o registro não precisa de mais nada.

---

## O que esta bateria NÃO decide

- **Não mede qualidade de resposta**, só qual ferramenta foi chamada. Acurácia
  condicional continua sendo a métrica da fase 3, e não é remedida aqui.
- **Não vale para os tipos que ainda não existem.** Curso, componente e
  extensão acrescentarão mais ferramentas, e a previsão 19 provavelmente deixa
  de valer perto de dez. Remedir antes de passar disso.
- **N=3 por pergunta.** Suficiente para ver instabilidade, insuficiente para
  distinguir 97,8% de 96,5%. Diferenças de um ou dois pontos aqui são ruído, e
  serão relatadas como tal.
