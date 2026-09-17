# Avaliação da fase 3 — acurácia de roteamento

Gerado por `interfaces/comparar.py` em 2026-09-14 17:35. Execução `20260914T173208`.

- Conjunto pré-registrado: **0** perguntas (`interfaces/conjunto_avaliacao.py`, commitado antes desta execução)
- Repetições do agente: **3** · Registro bruto: `docs/bateria_v3.jsonl`
- Execuções do agente: **0** válidas · **0** descartadas por falha de infraestrutura (timeout de rede não é decisão de roteamento)
- Modelo `qwen2.5:32b-instruct-q4_K_M` · embedding `BAAI/bge-m3` · TOP_K 10 · limiar 1.24

> ⚠️ **BATERIA INTERROMPIDA** — o Ollama ficou inacessível. O que está
> abaixo é parcial e não vale como resultado.

## As três métricas

| Métrica | Valor | Critério | |
|---|---|---|---|
| Acurácia de roteamento | **0.0%** | ≥ 95% | ❌ |
| Estabilidade | **0.0%** | ≥ 90% | ❌ |
| Acurácia condicional (objetivas) | **—** | ≥ 95% | — |
| Interpretativas sem afirmação sem respaldo | **—** | 100% | ❌ |

A última linha é o critério de tolerância zero do CLAUDE.md, verificado automaticamente: todo docente que a resposta afirma tem de aparecer no contexto que as ferramentas devolveram.

> ⚠️ **Estes são os valores AUTOMÁTICOS.** Item que o instrumento não consegue julgar com segurança entra aqui como reprovado, porque é o único palpite conservador que um cálculo sabe dar. A política de denominador e o **intervalo de robustez** — resolver todos os ambíguos como reprovados, depois todos como aprovados, e só declarar veredito quando as duas pontas caem do mesmo lado — estão em `docs/criterios_avaliacao.md`.
>
> **Um ❌ nesta tabela não significa critério reprovado**, e um ✅ não significa auditado. O veredito de uma fase sai da apuração contra o pré-registro dela, não daqui.

### Condicional por tipo de checagem

| Tipo | Corretas |
|---|---|

## Matriz de roteamento

Linhas = rota pré-registrada · colunas = rota escolhida pelo agente.

| esperada \ escolhida | estruturada | semantica | ambigua | nenhuma | outra |
|---|---|---|---|---|---|
| **estruturada** | 0 | 0 | 0 | 0 | 0 |
| **semantica** | 0 | 0 | 0 | 0 | 0 |
| **ambigua** | 0 | 0 | 0 | 0 | 0 |
| **nenhuma** | 0 | 0 | 0 | 0 | 0 |

## Por pergunta
