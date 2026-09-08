"""
Bateria de roteamento com CINCO ferramentas.

    docker compose run --rm agente python -m interfaces.bateria_roteamento

Pré-registro em `docs/pre_registro_roteamento_5_tools.md`, commitado antes.

POR QUE NÃO REUSEI `interfaces/comparar.py`
--------------------------------------------
Ele mede a tese da arquitetura — objetivo vs. interpretativo — e faz isso
deduzindo a rota da FONTE DE DADO tocada. As duas ferramentas de departamento
leem o mesmo SQLite, então chamar a errada produz `{"sqlite"}` e ele anota
"estruturada = correto". A acurácia de 97,8% continuaria alta com o agente
respondendo a coisa errada.

Não é defeito dele: é cegueira no eixo que a mudança de ontem abriu. Aqui a
unidade é a FERRAMENTA, e a rota é calculada junto só para os dois números
ficarem lado a lado.

O QUE ESTA BATERIA NÃO FAZ
--------------------------
Não julga a resposta. Acurácia condicional é métrica da fase 3 e não é
remedida aqui — misturar as duas coisas foi o que produziu o par de cegueiras
que este projeto já registrou duas vezes.
"""

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import config
from interfaces.conjunto_avaliacao import CONJUNTO
from interfaces.tipos import TIPOS

TOOL_SEMANTICA = "busca_vetorial_sigaa"

# Toda tool que sai de uma Busca do registro lê o SQLite; a semântica lê o
# Chroma. Derivado do registro em vez de escrito à mão: uma lista literal
# ficaria desatualizada no primeiro tipo novo, em silêncio.
TOOLS_ESTRUTURADAS = {b.nome_tool for t in TIPOS.values() for b in t.buscas}

# A ÚNICA pergunta das 30 cujo rótulo estruturado é vínculo de pessoa, e não
# contagem/listagem. Explícita para que a derivação abaixo seja auditável.
VINCULO_DE_PESSOA = {"est-07"}


def ferramenta_esperada(rota: str, id_pergunta: str) -> set[str]:
    """
    Derivação MECÂNICA do rótulo de ferramenta a partir do rótulo de rota, que
    já estava pré-registrado antes de qualquer execução.

    Escrever rótulo de ferramenta hoje é pré-registro mais fraco que o de rota
    — eu já vi o sistema rodar. Derivar em vez de escolher é o que limita o
    quanto eu poderia, sem querer, rotular na direção do que o agente faz.
    """
    if rota == "nenhuma":
        return set()
    if rota == "semantica":
        return {TOOL_SEMANTICA}
    if rota == "ambigua":
        return {"buscar_docentes_por_departamento", TOOL_SEMANTICA}
    if id_pergunta in VINCULO_DE_PESSOA:
        return {"buscar_docente_por_nome"}
    return {"buscar_docentes_por_departamento"}


@dataclass(frozen=True)
class Caso:
    id: str
    texto: str
    esperadas: frozenset


# --------------------------------------------------------------------- B ---
# Escritas ANTES de rodar. Cada uma existe por um motivo, e os dois que mais
# importam estão marcados no pré-registro: dep-07 é o teste da colisão, dep-08
# é o primeiro multi-salto entre tipos medido neste projeto.
PARTE_B = [
    Caso("dep-01", "Quais departamentos fazem parte do Instituto Multidisciplinar?",
         frozenset({"buscar_departamentos_por_centro"})),
    Caso("dep-02", "Que departamentos tem o Instituto de Ciências Exatas?",
         frozenset({"buscar_departamentos_por_centro"})),
    Caso("dep-03", "A que instituto pertence o Departamento de Ciência da Computação?",
         frozenset({"buscar_departamento_por_nome"})),
    Caso("dep-04", "O Departamento de Matemática fica em qual instituto?",
         frozenset({"buscar_departamento_por_nome"})),
    Caso("dep-05", "Quantos departamentos tem o Instituto de Veterinária?",
         frozenset({"buscar_departamentos_por_centro"})),
    Caso("dep-06", "O Departamento de Arte e Cultura pertence a qual instituto?",
         frozenset({"buscar_departamento_por_nome"})),
    Caso("dep-07", "Quantos docentes tem o Departamento de Ciência da Computação?",
         frozenset({"buscar_docentes_por_departamento"})),
    Caso("dep-08", "A que instituto pertence o departamento onde trabalha o Filipe Braida?",
         frozenset({"buscar_docente_por_nome", "buscar_departamento_por_nome"})),
    Caso("dep-09", "Liste os departamentos do Instituto de Educação.",
         frozenset({"buscar_departamentos_por_centro"})),
    Caso("dep-10", "Quantos institutos tem a UFRRJ?", frozenset()),
]


def _tools_chamadas(historico) -> list[str]:
    nomes = []
    for msg in historico:
        for chamada in (getattr(msg, "tool_calls", None) or []):
            nomes.append(
                getattr(chamada, "tool_name", None) or getattr(chamada, "name", "?")
            )
    return nomes


def _rota(tools: set[str]) -> str:
    """A rota como `comparar.py` a calcula, para os dois números ficarem lado a lado."""
    tem_sql = bool(tools & TOOLS_ESTRUTURADAS)
    tem_vec = TOOL_SEMANTICA in tools
    if tem_sql and tem_vec:
        return "ambigua"
    if tem_sql:
        return "estruturada"
    if tem_vec:
        return "semantica"
    return "nenhuma"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--saida", type=Path,
                        default=Path("docs/roteamento_5_tools.json"))
    parser.add_argument("--parte", choices=("A", "B", "AB"), default="AB")
    args = parser.parse_args()

    parte_a = [
        Caso(p.id, p.texto, frozenset(ferramenta_esperada(p.rota, p.id)))
        for p in CONJUNTO
    ]
    lotes = []
    if "A" in args.parte:
        lotes.append(("A", parte_a))
    if "B" in args.parte:
        lotes.append(("B", PARTE_B))

    print("=" * 78)
    print("INTERMEDIARIOS")
    print("=" * 78)
    from modulo2_inferencia.llm_setup import montar_componentes
    from modulo2_inferencia.agent import (
        montar_historico_inicial,
        processar_pergunta,
    )
    from modulo2_inferencia.tools import TOOLS_SCHEMA

    anunciadas = [s["function"]["name"] for s in TOOLS_SCHEMA]
    print(f"  tools anunciadas ... {len(anunciadas)}")
    for n in anunciadas:
        print(f"      {n}")
    print(f"  estruturadas ....... {sorted(TOOLS_ESTRUTURADAS)}")
    print(f"  llm ................ {config.MODELO_LLM}")
    total = sum(len(c) for _, c in lotes) * args.repeticoes
    print(f"  execucoes .......... {total}")

    comp = montar_componentes()
    registros = []

    for nome_parte, casos in lotes:
        print()
        print("=" * 78)
        print(f"PARTE {nome_parte} -- {len(casos)} perguntas x {args.repeticoes}")
        print("=" * 78)
        for caso in casos:
            for n in range(1, args.repeticoes + 1):
                try:
                    texto, historico = processar_pergunta(
                        comp.chat_generator, comp.embedder, comp.retriever,
                        montar_historico_inicial(), caso.texto,
                    )
                    erro = None
                except Exception as exc:            # noqa: BLE001
                    # Falha de execucao NAO e decisao de roteamento. Quatro
                    # timeouts na bateria de 5 set derrubaram o roteamento de
                    # 93,0% para 88,9% por serem contados como "escolheu nao
                    # usar ferramenta". Aqui a execucao fica marcada e sai do
                    # denominador.
                    texto, historico, erro = "", [], f"{type(exc).__name__}: {exc}"

                chamadas = _tools_chamadas(historico)
                usadas = frozenset(chamadas)
                acertou = None if erro else (usadas == caso.esperadas)
                # A CONFUSAO QUE A PREVISAO 20 MEDE: chamou uma ferramenta
                # estruturada que nao era a esperada.
                confundiu = None if erro else bool(
                    (usadas & TOOLS_ESTRUTURADAS) - caso.esperadas
                )
                marca = "!!" if erro else ("ok" if acertou else "XX")
                print(f"  {marca} {caso.id:8} ex{n}  {sorted(usadas) or '(nenhuma)'}")
                registros.append({
                    "parte": nome_parte, "id": caso.id, "execucao": n,
                    "pergunta": caso.texto,
                    "esperadas": sorted(caso.esperadas),
                    "chamadas": chamadas,
                    "usadas": sorted(usadas),
                    "rota": None if erro else _rota(set(usadas)),
                    "ferramenta_correta": acertou,
                    "confusao_estruturada": confundiu,
                    "resposta": texto,
                    "erro": erro,
                })
                args.saida.write_text(json.dumps({
                    "gerado_em": datetime.now().isoformat(timespec="seconds"),
                    "pre_registro": "docs/pre_registro_roteamento_5_tools.md",
                    "modelo_llm": config.MODELO_LLM,
                    "tools_anunciadas": anunciadas,
                    "repeticoes": args.repeticoes,
                    "registros": registros,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nEscrito em {args.saida} ({len(registros)} execucoes)")
    print("A apuracao contra o pre-registro vai no relatorio, nao aqui.")


if __name__ == "__main__":
    main()
