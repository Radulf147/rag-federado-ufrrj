"""
Repontua uma bateria JÁ GRAVADA, sem chamar o LLM.

    docker compose run --rm --no-deps agente \
        python -m interfaces.repontuar docs/avaliacao_fase3.jsonl --rotulo v1

POR QUE EXISTE
--------------
Até 5 set 2026 rodar o checker significava rodar a bateria inteira — 150
células, ~1h30 de máquina, e um resultado diferente a cada vez, porque o agente
é estocástico. Isso torna impossível responder à única pergunta que importa ao
mudar uma checagem: *este item mudou de veredito por causa da mudança, ou por
causa do sorteio?*

Aqui as respostas são dado fixo, lido do disco. Duas execuções sobre o mesmo
JSONL com o mesmo checker devolvem exatamente o mesmo veredito, e qualquer
diferença entre dois rótulos é atribuível ao checker e a mais nada.

ESCOPO — só a condicional objetiva, 48 itens
--------------------------------------------
`atribuicao_ok` e `nomes_sem_respaldo` NÃO são recomputáveis: o JSONL grava o
contexto recuperado apenas como tamanho (`"<8254 caracteres>"`), não o texto.
Os valores dessas chaves ficam congelados como a bateria os produziu, e o
arquivo de saída declara isso em `nao_recomputavel`.

O QUE ESTE MÓDULO NÃO FAZ
-------------------------
Não toca em `executar_comparacao()`. A bateria continua sendo a bateria; isto é
um leitor. Também não decide qual checker está certo — só carimba qual rodou,
pelo sha1 do código-fonte, para que o rótulo seja verificável e não uma
afirmação de quem escreveu o arquivo.

O SQLITE AINDA É DEPENDÊNCIA. Offline aqui quer dizer "sem LLM", não "sem
nada": o gabarito de cada pergunta é calculado por consulta ao corpus na hora.
Se o corpus mudar, o gabarito muda e duas repontuações deixam de ser
comparáveis — por isso a saída carimba a contagem de docentes e um sha256 dos
pares (nome, departamento). Contagem sozinha não detectaria uma renomeação.
"""

import argparse
import hashlib
import inspect
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import config
from interfaces.comparar import _conferir, _normalizar, nomes_afirmados
from interfaces.conjunto_avaliacao import CHECAGEM, CONJUNTO, _docentes

POR_ID = {p.id: p for p in CONJUNTO}


def _impressao_do_corpus() -> dict:
    """Contagem e sha256 dos pares (nome, departamento), normalizados."""
    pares = sorted(
        f"{_normalizar(r.get('nome', ''))}|{_normalizar(r.get('departamento', ''))}"
        for r in _docentes()
    )
    digest = hashlib.sha256("\n".join(pares).encode("utf-8")).hexdigest()
    return {"docentes": len(pares), "sha256": digest}


def _impressao_do_checker() -> str:
    """
    sha1 do código-fonte do checker.

    O rótulo (`v1`, `v2`) é escolha de quem chama e poderia mentir. Isto não:
    se dois arquivos trazem o mesmo `checker_sha1`, rodaram o mesmo código.
    """
    fonte = "".join(
        inspect.getsource(f) for f in (_conferir, nomes_afirmados, _normalizar)
    )
    return hashlib.sha1(fonte.encode("utf-8")).hexdigest()[:12]


def _objetivos(caminho: Path) -> list[dict]:
    """
    Os itens que a condicional objetiva mede.

    Mesmo filtro de `calcular_metricas`: execuções do agente, sem falha de
    infraestrutura, com roteamento correto e com verdade-base. Repetido aqui de
    propósito — se o filtro divergir, o gate de reprodução acusa.
    """
    itens = []
    for linha in caminho.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        registro = json.loads(linha)
        avaliacao = registro.get("avaliacao") or {}
        if registro.get("pipeline") != "3-agente":
            continue
        if avaliacao.get("falha_execucao") or not avaliacao.get("rota_correta"):
            continue
        if not avaliacao.get("verdade"):
            continue
        itens.append(registro)
    return itens


def repontuar(caminho: Path, rotulo: str) -> dict:
    registros = _objetivos(caminho)

    cache_verdade: dict[str, dict] = {}
    itens = []
    por_tipo = defaultdict(lambda: [0, 0])

    for registro in registros:
        pergunta = POR_ID[registro["pergunta_id"]]
        if pergunta.id not in cache_verdade:
            cache_verdade[pergunta.id] = pergunta.verdade()

        tipo = CHECAGEM[pergunta.id]
        afirmados = nomes_afirmados(registro["resposta"])
        veredito = _conferir(
            tipo, cache_verdade[pergunta.id], registro["resposta"], afirmados
        )

        por_tipo[tipo][1] += 1
        por_tipo[tipo][0] += bool(veredito["ok"])
        itens.append(
            {
                "pergunta_id": pergunta.id,
                "repeticao": registro["repeticao"],
                "pergunta": pergunta.texto,
                "tipo": tipo,
                "ok": bool(veredito["ok"]),
                "veredito_gravado": bool((registro["avaliacao"]["verdade"] or {}).get("ok")),
                "detalhe": veredito,
                "nomes_afirmados": afirmados,
            }
        )

    ok = sum(1 for i in itens if i["ok"])
    carimbos = {r.get("prompt_sha1") for r in registros}
    execucoes = {r.get("execucao") for r in registros}

    return {
        "rotulo": rotulo,
        "gerado_em": datetime.now().isoformat(timespec="seconds"),
        "jsonl": str(caminho),
        "execucao": sorted(e for e in execucoes if e),
        "prompt_sha1": sorted(c for c in carimbos if c),
        "checker_sha1": _impressao_do_checker(),
        "corpus": _impressao_do_corpus(),
        "db_path": config.DB_PATH,
        "escopo": (
            "condicional objetiva — execucoes do agente, sem falha de "
            "infraestrutura, com roteamento correto e verdade-base"
        ),
        "nao_recomputavel": {
            "chaves": ["atribuicao_ok", "nomes_sem_respaldo"],
            "motivo": (
                "o JSONL grava o contexto recuperado apenas como tamanho "
                "('<N caracteres>'), nao o texto; sem ele nao ha como reconferir "
                "quais nomes tinham respaldo"
            ),
            "situacao": "congelados no valor produzido pela bateria",
        },
        "resumo": {
            "ok": ok,
            "total": len(itens),
            "por_tipo": {t: {"ok": v[0], "total": v[1]} for t, v in sorted(por_tipo.items())},
        },
        "itens": itens,
    }


def _gate(resultado: dict, esperado_ok: int, esperado_total: int, sub: str) -> None:
    """
    Gate de reprodução — o checker atual tem de reproduzir o que a bateria
    gravou. Sem isso, comparar dois rótulos é comparar duas coisas
    desconhecidas.
    """
    resumo = resultado["resumo"]
    subconjunto = resumo["por_tipo"].get("subconjunto", {"ok": -1, "total": -1})
    real_sub = f"{subconjunto['ok']}/{subconjunto['total']}"
    problemas = []
    if (resumo["ok"], resumo["total"]) != (esperado_ok, esperado_total):
        problemas.append(
            f"total: esperado {esperado_ok}/{esperado_total}, obtido "
            f"{resumo['ok']}/{resumo['total']}"
        )
    if real_sub != sub:
        problemas.append(f"subconjunto: esperado {sub}, obtido {real_sub}")

    divergentes = [
        i for i in resultado["itens"] if i["ok"] != i["veredito_gravado"]
    ]
    if divergentes:
        problemas.append(
            "itens que divergem do veredito gravado: "
            + ", ".join(f"{i['pergunta_id']}#{i['repeticao']}" for i in divergentes)
        )

    if problemas:
        raise SystemExit(
            "GATE DE REPRODUCAO FALHOU — a comparacao entre rotulos nao vale.\n  "
            + "\n  ".join(problemas)
        )
    print(f"[GATE] reproduziu a bateria: {esperado_ok}/{esperado_total}, subconjunto {sub}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--rotulo", required=True, help="v1, v2, ...")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument(
        "--gate",
        metavar="OK/TOTAL,SUBCONJUNTO",
        default=None,
        help="exige reproducao exata, ex.: 44/48,17/21",
    )
    args = parser.parse_args()

    resultado = repontuar(args.jsonl, args.rotulo)

    if args.gate:
        total, sub = args.gate.split(",")
        ok_esperado, n_esperado = (int(x) for x in total.split("/"))
        _gate(resultado, ok_esperado, n_esperado, sub)

    carimbo = (resultado["prompt_sha1"] or ["sem_carimbo"])[0]
    saida = args.saida or Path("resultados") / f"{args.rotulo}_{carimbo}.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    resumo = resultado["resumo"]
    print(f"checker_sha1 {resultado['checker_sha1']} · corpus {resultado['corpus']['docentes']} docentes")
    print(f"condicional objetiva: {resumo['ok']} de {resumo['total']}")
    for tipo, v in resumo["por_tipo"].items():
        print(f"    {tipo:14} {v['ok']} de {v['total']}")
    print(f"\nEscrito em {saida}")


if __name__ == "__main__":
    main()
