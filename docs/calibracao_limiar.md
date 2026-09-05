# Calibração do limiar de distância — achado 03

Gerado por `modulo2_inferencia/calibrar_limiar.py` em 2026-09-05 02:19.

- Corpus: **1302** documentos · embedding `BAAI/bge-m3`
- **TOP_K = 10** — a medida é feita sobre a janela que o agente de fato vê
- Termos de dentro do domínio: **12**, derivados dos campos *Áreas de interesse* mais frequentes (mín. 5 docentes)
- Consultas de fora do domínio: **6**

> O `score` é **distância**: menor é mais parecido. O filtro é `score <= limiar`.

## A pergunta que decide tudo

O limiar precisa ficar **abaixo** do melhor resultado de toda consulta fora do domínio — senão ela recebe contexto que não existe — e **acima** do primeiro resultado de toda consulta de dentro — senão ela emudece.

- Pior 1º resultado entre as consultas de dentro: **1.192**
- Melhor 1º resultado entre as consultas de fora: **1.170**

**As duas faixas se sobrepõem** (1.192 ≥ 1.170): não existe limiar que cale todo o ruído sem emudecer alguma consulta legítima.

## Dentro do domínio

| Termo | Docentes no corpus | 1º | 10º | Relevantes no topo |
|---|---|---|---|---|
| formacao de professores | 8 | 0.786 | 0.820 | 1 de 10 |
| educacao | 17 | 0.830 | 0.935 | 7 de 10 |
| ecologia | 6 | 0.887 | 1.006 | 5 de 10 |
| agroecologia | 6 | 0.932 | 1.003 | 3 de 10 |
| hospitalidade | 10 | 0.956 | 0.996 | 4 de 10 |
| didatica | 6 | 0.970 | 1.066 | 0 de 10 |
| movimentos sociais | 7 | 0.989 | 1.074 | 10 de 10 |
| politicas publicas | 12 | 1.031 | 1.122 | 3 de 10 |
| seguranca alimentar | 6 | 1.032 | 1.142 | 1 de 10 |
| literatura | 9 | 1.034 | 1.092 | 7 de 10 |
| historia | 10 | 1.049 | 1.099 | 10 de 10 |
| politica | 7 | 1.192 | 1.236 | 5 de 10 |

## Fora do domínio

| Consulta | 1º resultado |
|---|---|
| letra da música que toca no rádio | 1.170 |
| preço do bitcoin hoje | 1.196 |
| receita de pão de queijo mineiro | 1.245 |
| como trocar o óleo de uma motocicleta | 1.246 |
| culinária japonesa medieval | 1.347 |
| resultado do campeonato escocês de futebol | 1.417 |

## Curva de decisão

| Limiar | Consultas de dentro com resposta | Relevantes mantidos | Relevantes perdidos | Consultas de fora caladas |
|---|---|---|---|---|
| 0.90 | 3 de 12 | 4 | 52 | 6 de 6 |
| 0.92 | 3 de 12 | 9 | 47 | 6 de 6 |
| 0.94 | 4 de 12 | 11 | 45 | 6 de 6 |
| 0.96 | 5 de 12 | 12 | 44 | 6 de 6 |
| 0.98 | 6 de 12 | 17 | 39 | 6 de 6 |
| 1.00 | 7 de 12 | 20 | 36 | 6 de 6 |
| 1.02 | 7 de 12 | 22 | 34 | 6 de 6 |
| 1.04 | 10 de 12 | 25 | 31 | 6 de 6 |
| 1.06 | 11 de 12 | 33 | 23 | 6 de 6 |
| 1.08 | 11 de 12 | 42 | 14 | 6 de 6 |
| 1.10 | 11 de 12 | 50 | 6 | 6 de 6 |
| 1.12 | 11 de 12 | 50 | 6 | 6 de 6 |
| 1.14 | 11 de 12 | 51 | 5 | 6 de 6 |
| 1.16 | 11 de 12 | 51 | 5 | 6 de 6 |
| 1.18 | 11 de 12 | 51 | 5 | 5 de 6 |
| 1.20 | 12 de 12 | 52 | 4 | 4 de 6 |
| 1.22 | 12 de 12 | 54 | 2 | 4 de 6 |
| 1.24 | 12 de 12 | 56 | 0 | 4 de 6 ← **recomendado** |
| 1.26 | 12 de 12 | 56 | 0 | 2 de 6 |
| 1.28 | 12 de 12 | 56 | 0 | 2 de 6 |
| 1.30 | 12 de 12 | 56 | 0 | 2 de 6 |
| 1.32 | 12 de 12 | 56 | 0 | 2 de 6 |
| 1.34 | 12 de 12 | 56 | 0 | 2 de 6 |
| 1.36 | 12 de 12 | 56 | 0 | 1 de 6 |
| 1.38 | 12 de 12 | 56 | 0 | 1 de 6 |
| 1.40 | 12 de 12 | 56 | 0 | 1 de 6 |
| 1.42 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.44 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.46 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.48 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.50 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.52 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.54 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.56 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.58 | 12 de 12 | 56 | 0 | 0 de 6 |
| 1.60 | 12 de 12 | 56 | 0 | 0 de 6 |

## Recomendação

**`LIMIAR_DISTANCIA=1.24`** — mantém resposta em **todas** as 12 consultas de dentro do domínio, sem perder **nenhum** dos 56 documentos relevantes do topo, e ainda cala **4 das 6** consultas de puro ruído.

É o maior ganho que sai de graça. A escolha é assimétrica de propósito: emudecer uma consulta legítima produz "não encontrei" para uma pergunta que **tem** resposta — erro invisível, com cara de cautela. Deixar passar ruído entrega documentos irrelevantes a um agente que recebe nome e departamento de cada um (achado 02) e está autorizado a dizer que não sabe (achado 07). Distância absoluta não distingue *pergunta sem resposta* de *pergunta legítima sobre tema pouco representado*; o modelo, lendo o que veio, distingue.

**A alternativa agressiva não compensa.** Com `LIMIAR_DISTANCIA=1.14` as 6 consultas de ruído ficariam caladas, mas ao preço de emudecer 1 consulta(s) legítima(s) e descartar 5 documentos relevantes.

### O que estes números NÃO são

"Relevantes no topo" é um **piso**: a verdade-base é lexical e conta como relevante só o documento que contém o termo literalmente. Vizinho semântico legítimo — quem escreveu "aprendizado de máquina" para a consulta "inteligência artificial" — entra como não relevante. As colunas de distância, que são as que decidem o limiar, não sofrem desse viés.

Uma versão anterior deste script calculava precisão sobre os 1302 documentos do corpus filtrados por distância, ignorando o TOP_K. Aquele número descrevia um sistema que não existe e foi descartado.
