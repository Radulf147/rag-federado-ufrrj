"""
Runner da comparação de pipelines (CLAUDE.md §3) — produz o documento de testes.

Roda o MESMO conjunto de perguntas pelos três pipelines e escreve
docs/testes_pipelines.md com os resultados lado a lado.

Arquivo irmão de cli.py: mesma camada (interface), mesma dependência de
llm_setup.montar_componentes(). A diferença é que este não é interativo —
roda em lote e escreve um arquivo.

    docker compose --profile agente run --rm agente python -m interfaces.comparar

O que o runner NÃO faz: julgar a qualidade das respostas. Ele preenche
objetivamente a fonte de dado usada por cada pipeline (é verificável) e deixa
os campos de julgamento em branco para preenchimento manual — inventar uma
nota automática de qualidade seria fabricar o resultado do experimento.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx

import config
from modulo2_inferencia.agent import SYSTEM_PROMPT
from modulo2_inferencia.llm_setup import NUM_CTX, REASONING_EFFORT, montar_componentes
from modulo2_inferencia.pipelines import PIPELINES, ResultadoPipeline

SAIDA = Path("docs/testes_pipelines.md")

# Registro bruto, uma linha por (pergunta × pipeline × repetição).
#
# NÃO é backup do markdown — é o dado primário, e o markdown é uma projeção
# dele. Duas razões: a métrica de estabilidade da fase de validação precisa
# saber qual rota foi escolhida em CADA repetição, e o markdown (uma resposta
# por pergunta por pipeline) não tem como representar isso; e o registro é
# gravado à medida que cada célula termina, então uma bateria interrompida
# deixa dado aproveitável em vez de nada.
REGISTRO = Path("docs/testes_pipelines.jsonl")

# Quanto esperar o Ollama voltar antes de abortar a bateria.
ESPERA_OLLAMA = int(os.getenv("ESPERA_OLLAMA", 180))

# Conjunto de perguntas. Misturado de propósito: as objetivas devem favorecer o
# pipeline 2, as interpretativas devem quebrá-lo, e a graça é ver o pipeline 3
# decidir. Mexer aqui é o jeito de estender a bateria de testes.
PERGUNTAS = [
    # -- objetivas (resposta exata, verificável no SQLite) --
    "Quantos professores tem o Departamento de Ciência da Computação?",
    "Liste os docentes do Departamento de Matemática.",
    "Quantos docentes tem o Departamento de Física?",
    # -- interpretativas (exigem o texto dos perfis, não o dado estruturado) --
    "Quais docentes pesquisam sobre Inteligência Artificial?",
    "Algum professor trabalha com banco de dados ou ciência de dados?",
    "Que docente tem formação em engenharia?",
    # -- de controle (nenhum pipeline deveria inventar uma resposta) --
    "Qual o telefone pessoal do reitor da UFRRJ?",
]


def _carimbo() -> dict:
    """
    Configuração sob a qual um registro foi produzido.

    Sem isto, dois registros do mesmo arquivo são indistinguíveis mesmo tendo
    saído de modelos ou prompts diferentes — e uma comparação entre eles seria
    inválida sem que nada no dado denunciasse.
    """
    return {
        "modelo": config.MODELO_LLM,
        "embedding": config.MODELO_EMBEDDING,
        "top_k": config.TOP_K,
        "num_ctx": NUM_CTX,
        "reasoning_effort": REASONING_EFFORT,
        "prompt_sha1": hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12],
    }


def _ollama_no_ar() -> bool:
    try:
        r = httpx.get(f"{config.OLLAMA_HOST}/api/version", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _aguardar_ollama() -> float:
    """
    Espera o Ollama voltar. Devolve os segundos esperados, ou -1 se desistiu.

    A espera existe porque o túnel SSH cai sozinho e volta — uma oscilação de
    30s não deveria envenenar uma célula. Mas a espera é contabilizada e
    reportada: retry que esconde instabilidade contamina a medição de
    qualidade com um problema de infraestrutura invisível.
    """
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


def _gravar(execucao: str, indice: int, repeticao: int, r: ResultadoPipeline) -> None:
    """Acrescenta uma célula ao registro bruto, imediatamente."""
    REGISTRO.parent.mkdir(parents=True, exist_ok=True)
    linha = {
        "execucao": execucao,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "indice": indice,
        "repeticao": repeticao,
        **asdict(r),
        **_carimbo(),
    }
    with REGISTRO.open("a", encoding="utf-8") as f:
        f.write(json.dumps(linha, ensure_ascii=False) + "\n")


def _executar(componentes, pergunta: str) -> list[ResultadoPipeline]:
    """Roda os três pipelines para uma pergunta, isolando falhas por célula."""
    resultados = []

    for nome, funcao in PIPELINES.items():
        try:
            resultados.append(funcao(componentes, pergunta))
        except Exception as erro:
            # Um pipeline quebrado não pode derrubar a bateria inteira — a falha
            # em si é um resultado do experimento e precisa aparecer no relatório.
            resultados.append(
                ResultadoPipeline(
                    pipeline=nome,
                    pergunta=pergunta,
                    resposta=f"**FALHOU:** `{type(erro).__name__}: {erro}`",
                    fontes=[],
                    detalhe="exceção durante a execução",
                )
            )
    return resultados


def _renderizar(
    todos: list[list[ResultadoPipeline]],
    execucao: str = "",
    esperas: list[float] | None = None,
    abortou: bool = False,
) -> str:
    agora = datetime.now().strftime("%Y-%m-%d %H:%M")
    esperas = esperas or []
    linhas = [
        "# Comparação dos três pipelines — dados de docentes",
        "",
        f"Gerado por `interfaces/comparar.py` em {agora}. Execução `{execucao}`.",
        "",
    ]

    if abortou:
        linhas += [
            "> ⚠️ **BATERIA INTERROMPIDA.** O Ollama ficou inacessível e a execução",
            f"> foi abortada após {len(todos)} de {len(PERGUNTAS)} perguntas. O que está",
            "> abaixo é parcial — não use como resultado final.",
            "",
        ]

    if esperas:
        linhas += [
            f"> ⚠️ **Instabilidade de infraestrutura:** o Ollama ficou fora do ar"
            f" {len(esperas)} vez(es) durante esta bateria, somando"
            f" {sum(esperas):.0f}s de espera. Os resultados são válidos, mas a"
            " infraestrutura não estava estável.",
            "",
        ]

    linhas += [
        f"- Modelo LLM: `{config.MODELO_LLM}`",
        f"- Modelo de embedding: `{config.MODELO_EMBEDDING}` (dim {config.EMBEDDING_DIM})",
        f"- TOP_K: {config.TOP_K}",
        f"- Registro bruto: `{REGISTRO}`",
        "",
        "Os campos **Qualidade** e **Alucinou?** são para preenchimento manual —",
        "o runner não julga resposta. **Fonte** é preenchida automaticamente e é",
        "verificável: diz de onde o dado saiu de fato.",
        "",
        "**Como julgar as perguntas objetivas:** o pipeline 2 consulta o SQLite",
        "por template, sem LLM — ele é incapaz de alucinar, então a resposta dele",
        "é a verdade-base contra a qual os pipelines 1 e 3 devem ser comparados.",
        "Nas interpretativas não há verdade-base automática; aí o julgamento é todo",
        "manual, conferindo contra os chunks citados no campo **Como**.",
        "",
    ]

    for resultados in todos:
        linhas += [f"## {resultados[0].pergunta}", ""]
        for r in resultados:
            fontes = ", ".join(r.fontes) if r.fontes else "nenhuma"
            linhas += [
                f"### Pipeline {r.pipeline}",
                "",
                f"- **Fonte:** {fontes}",
                f"- **Como:** {r.detalhe}",
                "- **Qualidade:** _(preencher)_",
                "- **Alucinou?:** _(preencher)_",
                "",
                "> " + r.resposta.replace("\n", "\n> "),
                "",
            ]
        linhas.append("---")
        linhas.append("")

    return "\n".join(linhas)


def executar_comparacao() -> None:
    execucao = datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"Execução {execucao} — configuração: {_carimbo()}")
    print("Montando componentes (pode demorar no primeiro run — baixa o embedder)...")
    componentes = montar_componentes()

    todos: list[list[ResultadoPipeline]] = []
    esperas: list[float] = []
    abortou = False

    for i, pergunta in enumerate(PERGUNTAS, 1):
        # Portão de saúde ANTES de cada pergunta. Sem ele, um Ollama fora do ar
        # não interrompe nada: cada célula vira "FALHOU" pelo try/except, a
        # bateria conclui com código 0 e escreve um relatório que parece
        # completo e não vale nada. Falha barulhenta é melhor que dado errado.
        esperou = _aguardar_ollama()
        if esperou < 0:
            print(f"\n[ABORTADO] Ollama não voltou em {ESPERA_OLLAMA}s.")
            print(f"           Interrompido antes da pergunta {i} de {len(PERGUNTAS)}.")
            print(f"           O que já rodou está em {REGISTRO}.")
            abortou = True
            break
        if esperou > 0:
            esperas.append(esperou)

        print(f"[{i}/{len(PERGUNTAS)}] {pergunta}")
        resultados = _executar(componentes, pergunta)
        for r in resultados:
            _gravar(execucao, i, 1, r)
        todos.append(resultados)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(_renderizar(todos, execucao, esperas, abortou), encoding="utf-8")
    print(f"\nRegistro bruto em {REGISTRO}")
    print(f"Relatório escrito em {SAIDA}")

    if abortou:
        raise SystemExit(1)


if __name__ == "__main__":
    executar_comparacao()
