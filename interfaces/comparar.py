"""
Runner da bateria de avaliação (CLAUDE.md §3) — mede roteamento, não texto.

Roda o conjunto PRÉ-REGISTRADO de `interfaces/conjunto_avaliacao.py` pelos três
pipelines e calcula as três métricas da fase 3.

    docker compose --profile agente run --rm agente python -m interfaces.comparar

O QUE MUDOU EM 4 SET 2026, E POR QUÊ
------------------------------------
A versão anterior rodava 7 perguntas sem rótulo, uma vez cada, e deixava
"Qualidade" e "Alucinou?" em branco para preenchimento manual. Isso não mede
roteamento — mede impressão de leitor.

Três mudanças:

1. **N repetições, só para o agente.** O LLM é estocástico: medir uma vez
   esconde um roteador que acerta 60% das vezes. Os pipelines 1 e 2 rodam uma
   vez — o 2 é determinístico e o 1 não roteia nada, então repetir não informa.

2. **Rota escolhida é deduzida das ferramentas chamadas**, que é fato
   registrado, não interpretação.

3. **Julgamento automático onde ele é possível**, e silêncio onde não é. O
   runner continua sem dar nota de qualidade — isso seria fabricar o resultado.
   Mas três coisas são computáveis e passaram a ser computadas:

   - a rota escolhida bate com a pré-registrada?
   - a resposta contém os valores que a verdade-base calcula do SQLite?
   - todo docente que a resposta AFIRMA aparece no contexto recuperado?

   A terceira é a que substitui a leitura no olho. A lição do achado 08 é que
   inspeção não pega o que importa: um sistema comparado consigo mesmo parece
   sempre coerente. O nome afirmado sem respaldo no contexto é o modo de falha
   do achado 07, e é detectável sem julgamento semântico.
"""

import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

import config
from interfaces.conjunto_avaliacao import CHECAGEM, CONJUNTO, _docentes, validar
from modulo2_inferencia.agent import SYSTEM_PROMPT
from modulo2_inferencia.llm_setup import NUM_CTX, REASONING_EFFORT, montar_componentes
from modulo2_inferencia.pipelines import PIPELINES, ResultadoPipeline

SAIDA = Path("docs/avaliacao_fase3.md")
REGISTRO = Path("docs/avaliacao_fase3.jsonl")

ESPERA_OLLAMA = int(os.getenv("ESPERA_OLLAMA", 180))

# Repetições do agente. Só ele precisa: é o único estocástico que roteia.
REPETICOES = int(os.getenv("REPETICOES", 3))

# Fonte de dado -> rota. É a tradução de "que ferramenta foi chamada" para "que
# caminho o agente escolheu", e é o coração da métrica de roteamento.
ROTA_POR_FONTES = {
    frozenset(): "nenhuma",
    frozenset({"sqlite"}): "estruturada",
    frozenset({"chromadb"}): "semantica",
    frozenset({"sqlite", "chromadb"}): "ambigua",
}


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.upper().split())


def _carimbo() -> dict:
    """Configuração sob a qual um registro foi produzido."""
    return {
        "modelo": config.MODELO_LLM,
        "embedding": config.MODELO_EMBEDDING,
        "top_k": config.TOP_K,
        "num_ctx": NUM_CTX,
        "reasoning_effort": REASONING_EFFORT,
        "limiar_distancia": os.getenv("LIMIAR_DISTANCIA", "") or None,
        "repeticoes": REPETICOES,
        "prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
    }


def _ollama_no_ar() -> bool:
    try:
        return httpx.get(f"{config.OLLAMA_HOST}/api/version", timeout=5).status_code == 200
    except Exception:
        return False


def _aguardar_ollama() -> float:
    """Espera o Ollama voltar. Devolve segundos esperados, ou -1 se desistiu."""
    if _ollama_no_ar():
        return 0.0
    print(f"  [ESPERA] Ollama não responde. Aguardando até {ESPERA_OLLAMA}s...")
    inicio = time.monotonic()
    while time.monotonic() - inicio < ESPERA_OLLAMA:
        time.sleep(5)
        if _ollama_no_ar():
            esperou = time.monotonic() - inicio
            print(f"  [ESPERA] Ollama voltou após {esperou:.0f}s.")
            return esperou
    return -1.0


# ------------------------------------------------------------------ avaliação

_NOMES_CORPUS: list[str] | None = None


def nomes_do_corpus() -> list[str]:
    """
    Todos os nomes de docente, normalizados, do mais longo para o mais curto.

    Do mais longo primeiro para "MARIA DA SILVA SANTOS" ser detectada antes de
    uma eventual "MARIA DA SILVA" contida nela.
    """
    global _NOMES_CORPUS
    if _NOMES_CORPUS is None:
        nomes = {_normalizar(r.get("nome", "")) for r in _docentes()}
        _NOMES_CORPUS = sorted((n for n in nomes if len(n) > 10), key=len, reverse=True)
    return _NOMES_CORPUS


def nomes_afirmados(resposta: str) -> list[str]:
    """Docentes do corpus que a resposta menciona."""
    alvo = _normalizar(resposta)
    encontrados = []
    for nome in nomes_do_corpus():
        if nome in alvo:
            encontrados.append(nome)
            alvo = alvo.replace(nome, " ")  # evita contar sobreposição duas vezes
    return encontrados


def _conferir(checagem, verdade, resposta, afirmados) -> dict:
    """
    Confere a resposta conforme o TIPO da pergunta.

    A primeira versão exigia o total do departamento em toda resposta com
    verdade-base, e reprovava "quem da Matemática pesquisa estatística?" por não
    conter o número 44 — que a resposta certa não tem por que conter. Media a
    coisa errada e devolvia 66,7% de acurácia condicional.
    """
    numeros = {int(n) for n in re.findall(r"\d+", resposta)}
    resposta_norm = _normalizar(resposta)

    if checagem == "contagem":
        esperados = list(verdade["contagens"].values())
        faltando = [v for v in esperados if v not in numeros]
        # Somar departamentos homônimos é o defeito do achado 06: a soma
        # aparecer no lugar das parcelas é justamente o erro a flagrar.
        soma_indevida = bool(
            verdade["ambiguo"] and sum(esperados) in numeros and faltando
        )
        return {
            "tipo": checagem,
            "esperado": verdade["contagens"],
            "faltando": faltando,
            "soma_indevida": soma_indevida,
            "ok": not faltando and not soma_indevida,
        }

    if checagem == "listagem":
        esperados = [n for nomes in verdade["departamentos"].values() for n in nomes]
        faltando = [n for n in esperados if _normalizar(n) not in resposta_norm]
        return {
            "tipo": checagem,
            "esperados": len(esperados),
            "faltando": faltando,
            "ok": not faltando,
        }

    if checagem == "vinculo":
        faltando = [
            d for d in verdade["departamentos"] if _normalizar(d) not in resposta_norm
        ]
        return {
            "tipo": checagem,
            "esperado": verdade["departamentos"],
            "faltando": faltando,
            "ok": not faltando,
        }

    if checagem == "subconjunto":
        # Não dá para saber quem DEVERIA estar na resposta — isso exigiria um
        # gabarito semântico. Dá para saber quem não poderia: todo docente
        # citado tem de pertencer ao departamento pedido. É o que pega a
        # resposta sobre o departamento errado.
        #
        # LIMITE HONESTO: um agente que responda sempre "não encontrei" passa
        # nesta checagem. Por isso `citados` vai no registro — sem ele, zero
        # intrusos seria indistinguível de zero esforço.
        elenco = {
            _normalizar(n) for nomes in verdade["departamentos"].values() for n in nomes
        }
        intrusos = [n for n in afirmados if n not in elenco]
        return {
            "tipo": checagem,
            "elenco": len(elenco),
            "citados": len(afirmados),
            "intrusos": intrusos,
            "ok": not intrusos,
        }

    return {"tipo": checagem, "ok": None}


def avaliar(pergunta, resultado: ResultadoPipeline) -> dict:
    """
    Verificações COMPUTÁVEIS sobre uma resposta. Nada de nota de qualidade.

    FALHA DE EXECUÇÃO NÃO É DECISÃO DE ROTEAMENTO. Um ReadTimeout devolve
    fontes=[], e a primeira versão pontuava isso como "o agente escolheu não
    usar ferramenta nenhuma". Quatro timeouts na bateria de 5 set derrubaram o
    roteamento de 93,0% para 88,9% e a estabilidade de 100% para 86,7% — queda
    de rede contada como escolha do modelo, que é exatamente o tipo de
    contaminação que esta bateria existe para evitar.
    """
    falha = resultado.detalhe == "exceção durante a execução"
    rota_escolhida = ROTA_POR_FONTES.get(frozenset(resultado.fontes), "outra")

    contexto = _normalizar(resultado.contexto)
    afirmados = nomes_afirmados(resultado.resposta)
    sem_respaldo = [n for n in afirmados if n not in contexto]

    avaliacao = {
        "falha_execucao": falha,
        "rota_esperada": pergunta.rota,
        "rota_escolhida": None if falha else rota_escolhida,
        "rota_correta": (not falha) and rota_escolhida == pergunta.rota,
        "nomes_afirmados": len(afirmados),
        "nomes_sem_respaldo": sem_respaldo,
        "atribuicao_ok": not sem_respaldo,
    }

    checagem = CHECAGEM.get(pergunta.id, "")
    if checagem and pergunta.verdade and not falha:
        avaliacao["verdade"] = _conferir(
            checagem, pergunta.verdade(), resultado.resposta, afirmados
        )

    return avaliacao


def _gravar(execucao: str, pergunta, repeticao: int, r: ResultadoPipeline, aval: dict) -> None:
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    linha = {
        "execucao": execucao,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "pergunta_id": pergunta.id,
        "repeticao": repeticao,
        **asdict(r),
        "avaliacao": aval,
        **_carimbo(),
    }
    # O contexto pode ter dezenas de KB; o registro guarda só o tamanho, porque
    # o que a checagem precisa dele já virou `nomes_sem_respaldo`.
    linha["contexto"] = f"<{len(r.contexto)} caracteres>"
    with REGISTRO.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(linha, ensure_ascii=False) + "\n")


def _rodar(componentes, pergunta, nome_pipeline: str) -> ResultadoPipeline:
    try:
        return PIPELINES[nome_pipeline](componentes, pergunta.texto)
    except Exception as erro:
        return ResultadoPipeline(
            pipeline=nome_pipeline,
            pergunta=pergunta.texto,
            resposta=f"**FALHOU:** `{type(erro).__name__}: {erro}`",
            fontes=[],
            detalhe="exceção durante a execução",
        )


# ------------------------------------------------------------------- métricas


def calcular_metricas(registros: list[dict]) -> dict:
    """
    As três métricas do CLAUDE.md §3, sobre as execuções do agente.

    Execuções que FALHARAM por infraestrutura ficam de fora de tudo e são
    contadas à parte. Um ReadTimeout não é uma decisão de roteamento, e
    tratá-lo como tal contamina a medida com um problema que não é do modelo.
    """
    do_agente = [r for r in registros if r["pipeline"] == "3-agente"]
    falhas = [r for r in do_agente if r["avaliacao"].get("falha_execucao")]
    validas = [r for r in do_agente if not r["avaliacao"].get("falha_execucao")]

    # 1. Acurácia de roteamento — por execução, não por pergunta.
    acertos = sum(1 for r in validas if r["avaliacao"]["rota_correta"])
    roteamento = acertos / len(validas) if validas else 0.0

    # 2. Estabilidade — a pergunta roteou igual em TODAS as repetições válidas?
    por_pergunta = defaultdict(list)
    for r in validas:
        por_pergunta[r["pergunta_id"]].append(r["avaliacao"]["rota_escolhida"])
    estaveis = sum(1 for rotas in por_pergunta.values() if len(set(rotas)) == 1)
    estabilidade = estaveis / len(por_pergunta) if por_pergunta else 0.0

    # 3. Acurácia condicional — dado roteamento certo, a resposta se sustenta?
    #    Separa erro de roteamento de erro de resposta, que é o ponto da métrica.
    corretas = [r for r in validas if r["avaliacao"]["rota_correta"]]
    objetivas = [r for r in corretas if r["avaliacao"].get("verdade")]
    ok_objetivas = sum(1 for r in objetivas if r["avaliacao"]["verdade"]["ok"])
    interpretativas = [r for r in corretas if not r["avaliacao"].get("verdade")]
    ok_interpretativas = sum(1 for r in interpretativas if r["avaliacao"]["atribuicao_ok"])

    # Detalhe por tipo de checagem: um agregado esconde qual classe quebrou.
    por_tipo = defaultdict(lambda: [0, 0])
    for r in objetivas:
        tipo = r["avaliacao"]["verdade"]["tipo"]
        por_tipo[tipo][1] += 1
        if r["avaliacao"]["verdade"]["ok"]:
            por_tipo[tipo][0] += 1

    return {
        "execucoes_do_agente": len(do_agente),
        "falhas_de_infraestrutura": len(falhas),
        "execucoes_validas": len(validas),
        "roteamento": roteamento,
        "estabilidade": estabilidade,
        "perguntas": len(por_pergunta),
        "objetivas_avaliadas": len(objetivas),
        "objetivas_corretas": ok_objetivas,
        "condicional_objetivas": ok_objetivas / len(objetivas) if objetivas else None,
        "por_tipo": dict(por_tipo),
        "interpretativas_avaliadas": len(interpretativas),
        "interpretativas_sem_afirmacao_solta": ok_interpretativas,
        "condicional_interpretativas": (
            ok_interpretativas / len(interpretativas) if interpretativas else None
        ),
        "matriz": Counter(
            (r["avaliacao"]["rota_esperada"], r["avaliacao"]["rota_escolhida"])
            for r in validas
        ),
    }


def _pct(valor) -> str:
    return "—" if valor is None else f"{valor * 100:.1f}%"


def _renderizar(registros, metricas, execucao, esperas, abortou) -> str:
    rotas = ["estruturada", "semantica", "ambigua", "nenhuma", "outra"]
    linhas = [
        "# Avaliação da fase 3 — acurácia de roteamento",
        "",
        f"Gerado por `interfaces/comparar.py` em {datetime.now():%Y-%m-%d %H:%M}. "
        f"Execução `{execucao}`.",
        "",
        f"- Conjunto pré-registrado: **{metricas['perguntas']}** perguntas "
        "(`interfaces/conjunto_avaliacao.py`, commitado antes desta execução)",
        f"- Repetições do agente: **{REPETICOES}** · Registro bruto: `{REGISTRO}`",
        f"- Execuções do agente: **{metricas['execucoes_validas']}** válidas · "
        f"**{metricas['falhas_de_infraestrutura']}** descartadas por falha de "
        "infraestrutura (timeout de rede não é decisão de roteamento)",
        f"- Modelo `{config.MODELO_LLM}` · embedding `{config.MODELO_EMBEDDING}` · "
        f"TOP_K {config.TOP_K} · limiar {os.getenv('LIMIAR_DISTANCIA') or 'desligado'}",
        "",
    ]
    if abortou:
        linhas += [
            "> ⚠️ **BATERIA INTERROMPIDA** — o Ollama ficou inacessível. O que está",
            "> abaixo é parcial e não vale como resultado.",
            "",
        ]
    if esperas:
        linhas += [
            f"> ⚠️ O Ollama ficou fora do ar {len(esperas)} vez(es), somando "
            f"{sum(esperas):.0f}s de espera. Resultados válidos, infraestrutura instável.",
            "",
        ]

    criterio = [
        ("Acurácia de roteamento", metricas["roteamento"], 0.95),
        ("Estabilidade", metricas["estabilidade"], 0.90),
        ("Acurácia condicional (objetivas)", metricas["condicional_objetivas"], 0.95),
    ]
    linhas += ["## As três métricas", "", "| Métrica | Valor | Critério | |", "|---|---|---|---|"]
    for nome, valor, meta in criterio:
        situacao = "—" if valor is None else ("✅" if valor >= meta else "❌")
        linhas.append(f"| {nome} | **{_pct(valor)}** | ≥ {meta * 100:.0f}% | {situacao} |")
    linhas += [
        f"| Interpretativas sem afirmação sem respaldo | **{_pct(metricas['condicional_interpretativas'])}** "
        f"| 100% | {'✅' if metricas['condicional_interpretativas'] == 1 else '❌'} |",
        "",
        "A última linha é o critério de tolerância zero do CLAUDE.md, verificado "
        "automaticamente: todo docente que a resposta afirma tem de aparecer no "
        "contexto que as ferramentas devolveram.",
        "",
        "### Condicional por tipo de checagem",
        "",
        "| Tipo | Corretas |",
        "|---|---|",
        *[f"| {t} | {v[0]} de {v[1]} |" for t, v in sorted(metricas["por_tipo"].items())],
        "",
        "## Matriz de roteamento",
        "",
        "Linhas = rota pré-registrada · colunas = rota escolhida pelo agente.",
        "",
        "| esperada \\ escolhida | " + " | ".join(rotas) + " |",
        "|---" * (len(rotas) + 1) + "|",
    ]
    for esperada in rotas[:-1]:
        celulas = [str(metricas["matriz"].get((esperada, e), 0)) for e in rotas]
        linhas.append(f"| **{esperada}** | " + " | ".join(celulas) + " |")

    linhas += ["", "## Por pergunta", ""]
    por_id = defaultdict(list)
    for r in registros:
        por_id[r["pergunta_id"]].append(r)

    for pergunta in CONJUNTO:
        registros_p = por_id.get(pergunta.id, [])
        if not registros_p:
            continue
        linhas += [
            f"### `{pergunta.id}` — {pergunta.texto}",
            "",
            f"- **Rota pré-registrada:** `{pergunta.rota}` — {pergunta.porque}",
        ]
        do_agente = [r for r in registros_p if r["pipeline"] == "3-agente"]
        if do_agente:
            escolhidas = [r["avaliacao"]["rota_escolhida"] for r in do_agente]
            linhas.append(f"- **Rotas escolhidas ({len(escolhidas)} execuções):** "
                          + ", ".join(f"`{r}`" for r in escolhidas))
            soltos = {n for r in do_agente for n in r["avaliacao"]["nomes_sem_respaldo"]}
            if soltos:
                linhas.append(f"- ⚠️ **Afirmados sem respaldo no contexto:** "
                              + ", ".join(sorted(soltos)))
            verdade = next((r["avaliacao"].get("verdade") for r in do_agente if r["avaliacao"].get("verdade")), None)
            if verdade and "contagens_esperadas" in verdade:
                linhas.append(f"- **Verdade-base (SQLite):** "
                              + ", ".join(f"{d} = {n}" for d, n in verdade["contagens_esperadas"].items()))
        linhas.append("")
        for r in registros_p:
            linhas += [
                f"**{r['pipeline']}** — fonte: {', '.join(r['fontes']) or 'nenhuma'}"
                + (f" · repetição {r['repeticao']}" if r["pipeline"] == "3-agente" else ""),
                "",
                "> " + (r["resposta"] or "").replace("\n", "\n> ")[:1200],
                "",
            ]
        linhas.append("---")
        linhas.append("")

    return "\n".join(linhas)


def executar_comparacao() -> None:
    validar()
    execucao = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Execução {execucao} — {_carimbo()}")
    print(f"{len(CONJUNTO)} perguntas pré-registradas · {REPETICOES} repetições do agente")
    componentes = montar_componentes()

    registros: list[dict] = []
    esperas: list[float] = []
    abortou = False

    for i, pergunta in enumerate(CONJUNTO, 1):
        # Portão de saúde ANTES de cada pergunta: sem ele um Ollama fora do ar
        # faz cada célula virar "FALHOU", a bateria conclui com código 0 e
        # escreve um relatório que parece completo e não vale nada.
        esperou = _aguardar_ollama()
        if esperou < 0:
            print(f"\n[ABORTADO] Ollama não voltou. Parou antes de {pergunta.id}.")
            abortou = True
            break
        if esperou > 0:
            esperas.append(esperou)

        print(f"[{i}/{len(CONJUNTO)}] {pergunta.id} ({pergunta.rota}) {pergunta.texto[:58]}")

        # Pipelines 1 e 2 uma vez: o 2 é determinístico, o 1 não roteia.
        for nome in ("1-vetorial", "2-estruturado"):
            resultado = _rodar(componentes, pergunta, nome)
            avaliacao = avaliar(pergunta, resultado)
            _gravar(execucao, pergunta, 1, resultado, avaliacao)
            registros.append({"pergunta_id": pergunta.id, "repeticao": 1,
                              **asdict(resultado), "avaliacao": avaliacao})

        for repeticao in range(1, REPETICOES + 1):
            resultado = _rodar(componentes, pergunta, "3-agente")
            avaliacao = avaliar(pergunta, resultado)
            _gravar(execucao, pergunta, repeticao, resultado, avaliacao)
            registros.append({"pergunta_id": pergunta.id, "repeticao": repeticao,
                              **asdict(resultado), "avaliacao": avaliacao})
            marca = "ok" if avaliacao["rota_correta"] else "ERRO"
            print(f"      rep {repeticao}: {avaliacao['rota_escolhida']:12} [{marca}]")

    metricas = calcular_metricas(registros)
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(
        _renderizar(registros, metricas, execucao, esperas, abortou), encoding="utf-8"
    )

    print()
    print(f"  roteamento ................ {_pct(metricas['roteamento'])}  (critério 95%)")
    print(f"  estabilidade .............. {_pct(metricas['estabilidade'])}  (critério 90%)")
    print(f"  condicional (objetivas) ... {_pct(metricas['condicional_objetivas'])}  (critério 95%)")
    print(f"  interpretativas limpas .... {_pct(metricas['condicional_interpretativas'])}  (critério 100%)")
    print(f"\nRelatório em {SAIDA} · registro bruto em {REGISTRO}")

    if abortou:
        raise SystemExit(1)


if __name__ == "__main__":
    executar_comparacao()
