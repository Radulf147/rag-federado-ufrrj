# Auditoria de amostra dos perfis de docente — achado 04

Gerado por `modulo1_etl/auditoria_perfis.py` em 2026-09-04 16:46.

- Amostra: **40** docentes sorteados de 40 pedidos (seed `42`, reprodutível)
- Falhas de requisição: 0
- Registro bruto: `docs/auditoria_perfis.jsonl`

A pergunta desta auditoria é a do princípio 2 do CLAUDE.md: *se o docente tivesse preenchido o dado, nosso sistema o encontraria?* Ela separa o que falta por culpa nossa do que falta na fonte.

## Como ler as colunas

| Coluna | Significado |
|---|---|
| **Conteúdo** | o campo existe na página e tem texto real — é dado aproveitável |
| **Placeholder** | o campo existe, mas o SIGAA gravou "não informada" — não é dado, e desde a correção do achado 03 não entra no texto indexado |
| **Ausente** | o campo não aparece na página — lacuna da fonte, não é falha nossa |
| **Parser captura** | quantos o código **corrigido** extrai hoje |
| **Está no store** | quantos estão de fato no ChromaDB agora — se bater com a coluna anterior, não há perda nossa |

## Campos que já coletamos

| Campo | Conteúdo | Placeholder | Ausente | Parser captura | Está no store |
|---|---|---|---|---|---|
| Perfil | 15 | 17 | 8 | 15 | 15 |
| Formação | 23 | 9 | 8 | 23 | 23 |
| Áreas de interesse | 21 | 11 | 8 | 21 | 21 |
| Currículo Lattes | 32 | 0 | 8 | 32 | 32 |
| Endereço | 17 | 23 | 0 | 17 | 17 |
| Sala | 11 | 29 | 0 | 11 | 11 |
| Telefone | 36 | 4 | 0 | 36 | 36 |
| E-mail | 37 | 3 | 0 | 37 | 37 |

## O achado 01, quantificado

- Perfis em que a chave antiga (com espaço) casava: **0 de 40**
- Perfis com áreas de interesse de conteúdo real na página: **21 de 40**
- Perfis com o campo no store hoje: **21 de 40**

## Campos da aba de docentes ainda não coletados

**Nenhum.** Todos os campos que a página do docente oferece entraram em `CAMPOS_DO_PERFIL` em 4 set 2026. O que falta agora são as outras abas do SIGAA, que é o achado 05 e exige requisições novas.

## Veredito

- Campos com conteúdo real na página: **192** de 320 possíveis
- Desses, **não** estão no store hoje: **0** (**0%**) — é a falha nossa
- Campos que são só placeholder "não informada": **96** (30% do total) — corretamente **fora** do texto indexado desde o achado 03
