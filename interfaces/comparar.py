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
from interfaces.conjunto_avaliacao import CONJUNTO, _docentes, validar
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


def avaliar(pergunta, resultado: ResultadoPipeline) -> dict:
    """
    Verificações COMPUTÁVEIS sobre uma resposta. Nada de nota de qualidade.

    Tudo aqui é fato verificável por terceiro a partir do registro bruto — que é
    o padrão que este projeto passou a exigir depois de descobrir que inspeção
    no olho não pega defeito estrutural.
    """
    rota_escolhida = ROTA_POR_FONTES.get(frozenset(resultado.fontes), "outra")

    contexto = _normalizar(resultado.contexto)
    afirmados = nomes_afirmados(resultado.resposta)
    sem_respaldo = [n for n in afirmados if n not in contexto]

    avaliacao = {
        "rota_esperada": pergunta.rota,
        "rota_escolhida": rota_escolhida,
        "rota_correta": rota_escolhida == pergunta.rota,
        "nomes_afirmados": len(afirmados),
        "nomes_sem_respaldo": sem_respaldo,
        "atribuicao_ok": not sem_respaldo,
    }

    # Verdade-base: os valores que o SQLite diz, conferidos contra o texto.
    if pergunta.verdade:
        verdade = pergunta.verdade()
        numeros = {int(n) for n in re.findall(r"\d+", resultado.resposta)}
        if verdade["tipo"] == "departamentos":
            esperados = list(verdade["contagens"].values())
            avaliacao["verdade"] = {
                "ambiguo": verdade["ambiguo"],
                "contagens_esperadas": verdade["contagens"],
                "presentes": [v for v in esperados if v in numeros],
                "faltando": [v for v in esperados if v not in numeros],
            }
            if verdade["ambiguo"]:
                # Somar departamentos homônimos é o defeito do achado 06. Se a
                # soma aparece e as parcelas não, o agente somou o que não devia.
                soma = sum(esperados)
                avaliacao["verdade"]["soma_indevida"] = (
                    soma in numeros and not all(v in numeros for v in esperados)
                )
        else:
            esperados = [_normalizar(d) for d in verdade["departamentos"]]
            resposta_norm = _normalizar(resultado.resposta)
            avaliacao["verdade"] = {
                "departamentos_esperados": verdade["departamentos"],
                "presentes": [d for d in esperados if d in resposta_norm],
            }

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
    """As três métricas do CLAUDE.md §3, sobre as execuções do agente."""
    do_agente = [r for r in registros if r["pipeline"] == "3-agente"]

    # 1. Acurácia de roteamento — por execução, não por pergunta.
    acertos = sum(1 for r in do_agente if r["avaliacao"]["rota_correta"])
    roteamento = acertos / len(do_agente) if do_agente else 0.0

    # 2. Estabilidade — a pergunta roteou igual em TODAS as repetições?
    por_pergunta = defaultdict(list)
    for r in do_agente:
        por_pergunta[r["pergunta_id"]].append(r["avaliacao"]["rota_escolhida"])
    estaveis = sum(1 for rotas in por_pergunta.values() if len(set(rotas)) == 1)
    estabilidade = estaveis / len(por_pergunta) if por_pergunta else 0.0

    # 3. Acurácia condicional — dado roteamento certo, a resposta se sustenta?
    #    Separa erro de roteamento de erro de resposta, que é o ponto da métrica.
    corretas = [r for r in do_agente if r["avaliacao"]["rota_correta"]]
    objetivas = [r for r in corretas if "verdade" in r["avaliacao"]]
    ok_objetivas = sum(
        1 for r in objetivas
        if not r["avaliacao"]["verdade"].get("faltando")
        and not r["avaliacao"]["verdade"].get("soma_indevida")
    )
    interpretativas = [r for r in corretas if "verdade" not in r["avaliacao"]]
    ok_interpretativas = sum(1 for r in interpretativas if r["avaliacao"]["atribuicao_ok"])

    return {
        "execucoes_do_agente": len(do_agente),
        "roteamento": roteamento,
        "estabilidade": estabilidade,
        "perguntas": len(por_pergunta),
        "objetivas_avaliadas": len(objetivas),
        "objetivas_corretas": ok_objetivas,
        "condicional_objetivas": ok_objetivas / len(objetivas) if objetivas else None,
        "interpretativas_avaliadas": len(interpretativas),
        "interpretativas_sem_afirmacao_solta": ok_interpretativas,
        "condicional_interpretativas": (
            ok_interpretativas / len(interpretativas) if interpretativas else None
        ),
        "matriz": Counter(
            (r["avaliacao"]["rota_esperada"], r["avaliacao"]["rota_escolhida"])
            for r in do_agente
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
