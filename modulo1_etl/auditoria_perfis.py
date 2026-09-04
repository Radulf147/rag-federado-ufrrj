"""
Auditoria de amostra dos perfis de docente — ACHADO 04.

PERGUNTA QUE ESTE SCRIPT RESPONDE: do corpus vazio que temos hoje, quanto é
falha nossa (o dado está na página e não capturamos) e quanto é ausência real
(o docente não preencheu o campo no SIGAA)?

Essa distinção é o princípio 2 do CLAUDE.md — "se o docente tivesse preenchido
o dado, nosso sistema o encontraria?". Sem medi-la, não dá para saber se vale
investir em extração ou se estamos no limite da fonte. Por isso a auditoria não
recarrega nada nem toca no store: ela só compara três coisas, por docente, para
cada campo do perfil:

    1. o que a página do SIGAA mostra AGORA
    2. o que o nosso parser extrai dela AGORA (código corrigido)
    3. o que está guardado no ChromaDB (carga anterior, código antigo)

A diferença (1) vs (2) é falha nossa atual. A diferença (2) vs (3) é o ganho que
a recarga vai trazer. O que falta em (1) é lacuna do SIGAA e não é problema
nosso.

TERCEIRA CATEGORIA, QUE NÃO É ÓBVIA: o SIGAA não deixa o campo vazio — ele
grava a string "não informada". Um perfil sem nada preenchido vira
"Perfil: não informada Formação: não informada Endereço: não informado", texto
que nós vetorizamos como se fosse conteúdo. Isso não é ausência nem captura: é
ruído idêntico em centenas de perfis, e é candidato direto a causa do achado 03
(recuperação que não discrimina). Por isso "placeholder" é contado à parte.

    docker compose run --rm etl python -m modulo1_etl.auditoria_perfis [--n 40] [--seed 42]

Tem de ser com `-m`, a partir da raiz — ver "Convenção de imports mista" no
CLAUDE.md.
"""

import argparse
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

import config

# parte2_scraping_docentes.py usa imports planos entre os arquivos do módulo 1
# (`from db_manager import ...`). Rodando este script com `-m` a partir da raiz,
# a pasta do módulo não está no sys.path e aquele import falha. Acrescentá-la
# aqui é o que permite reusar o parser DE VERDADE em vez de reimplementá-lo —
# e reimplementá-lo seria justamente o erro que esta auditoria existe para
# evitar: mediria a minha cópia, não o código que roda no ETL.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from parte2_scraping_docentes import (  # noqa: E402
    BASE_URL,
    CAMPOS_DO_PERFIL,
    _e_placeholder,
    _valor_por_prefixo,
)

# Campos que o perfil oferece e que NÃO coletamos hoje. Confirmados presentes
# no <dl> em 4 set 2026. São os "campos grátis" do achado 05: entram na mesma
# recarga sem custo de requisição nenhum, porque já vêm na página que já
# buscamos. A auditoria mede quantos de fato têm conteúdo antes de decidirmos.
# Vazio desde 4 set 2026: Currículo Lattes, Sala e E-mail passaram a ser
# coletados e vivem agora em CAMPOS_DO_PERFIL. A lista continua existindo
# porque as outras abas do SIGAA (achado 05) vão repovoá-la.
CAMPOS_NAO_COLETADOS: list[tuple[str, str]] = []

SAIDA_MD = Path("docs/auditoria_perfis.md")
SAIDA_JSONL = Path("docs/auditoria_perfis.jsonl")

# Delay entre requisições. O mesmo DELAY_NIVEL3 do scraping de perfis — não há
# motivo para esta auditoria ser mais agressiva com o servidor da UFRRJ que o
# ETL. 40 perfis × 1s é cerca de um minuto, e tempo não é critério aqui.
DELAY = 1.0


# _e_placeholder vem de parte2_scraping_docentes de propósito: é a MESMA regra
# que decide o que entra no texto indexado. Duas cópias divergiriam, e a
# auditoria passaria a medir um critério diferente do que o ETL aplica —
# exatamente o tipo de discrepância silenciosa que ela existe para detectar.

def classificar(valor: str) -> str:
    """ausente | placeholder | conteudo"""
    if not (valor or "").strip():
        return "ausente"
    if _e_placeholder(valor):
        return "placeholder"
    return "conteudo"


def campos_da_pagina(soup: BeautifulSoup) -> dict[str, str]:
    """
    Reproduz exatamente a construção do dict de campos em
    _montar_conteudo_docente — inclusive o get_text(strip=True), que é onde o
    achado 01 nascia. Reproduzir e não aproximar é o ponto.
    """
    return {
        dt.get_text(strip=True).rstrip(":").lower(): dd.get_text(separator=" ", strip=True)
        for dl in soup.find_all("dl")
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd"))
    }


# A chave literal que o código usava antes da correção do achado 01. Mantida
# aqui só para quantificar o ganho: quantos perfis passam a ter áreas de
# interesse por causa da correção.
CHAVE_ANTIGA_AREAS = "áreas de interesse (áreas de interesse de ensino e pesquisa)"


def amostrar_docentes(n: int, seed: int) -> list[dict]:
    """Sorteia n docentes distintos do que já está carregado no ChromaDB."""
    from haystack_integrations.document_stores.chroma import ChromaDocumentStore

    if config.CHROMA_REMOTE:
        store = ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            host=config.CHROMA_HOST,
            port=config.CHROMA_PORT,
            embedding_function="default",
        )
    else:
        store = ChromaDocumentStore(
            collection_name=config.CHROMA_COLECAO,
            persist_path=config.CHROMA_PERSIST_DIR,
            embedding_function="default",
        )

    por_siape: dict[str, dict] = {}
    for doc in store.filter_documents():
        if doc.meta.get("content_type") != "docente_perfil":
            continue
        siape = str(doc.meta.get("siape") or "")
        if not siape:
            continue
        registro = por_siape.setdefault(
            siape,
            {
                "siape": siape,
                "nome": doc.meta.get("nome_docente", ""),
                "departamento": doc.meta.get("departamento", ""),
                "conteudo_armazenado": "",
            },
        )
        # Um docente pode ter virado mais de um chunk; a comparação precisa do
        # texto inteiro que está no store, não do primeiro pedaço.
        registro["conteudo_armazenado"] += doc.content or ""

    universo = sorted(por_siape.values(), key=lambda d: d["siape"])
    print(f"[AUDITORIA] {len(universo)} docentes distintos no store.")
    random.Random(seed).shuffle(universo)
    return universo[:n]


def auditar_um(cliente: httpx.Client, docente: dict) -> dict:
    url = f"{BASE_URL}/sigaa/public/docente/portal.jsf?siape={docente['siape']}"
    resposta = cliente.get(url)
    resposta.raise_for_status()
    html = resposta.text
    soup = BeautifulSoup(html, "lxml")
    campos = campos_da_pagina(soup)

    armazenado = docente["conteudo_armazenado"]
    linha = {
        "siape": docente["siape"],
        "nome": docente["nome"],
        "departamento": docente["departamento"],
        "url": url,
        "pagina_valida": f"siape={docente['siape']}" in html,
        "campos": {},
        "nao_coletados": {},
    }

    for prefixo, rotulo in CAMPOS_DO_PERFIL:
        valor = _valor_por_prefixo(campos, prefixo)
        linha["campos"][rotulo] = {
            "na_pagina": classificar(valor),
            # Mesmo critério que _montar_conteudo_docente aplica hoje.
            "parser_novo_capturou": bool(valor.strip() and not _e_placeholder(valor)),
            # O que está no store veio da carga anterior, com o código antigo.
            "no_store": f"{rotulo}:" in armazenado,
            "tamanho": len(valor or ""),
        }

    # Ganho específico do achado 01: a chave antiga casava?
    linha["areas_chave_antiga_casava"] = CHAVE_ANTIGA_AREAS in campos

    for prefixo, rotulo in CAMPOS_NAO_COLETADOS:
        valor = _valor_por_prefixo(campos, prefixo)
        linha["nao_coletados"][rotulo] = classificar(valor)

    return linha


def escrever_relatorio(linhas: list[dict], n: int, seed: int, falhas: list[str]) -> None:
    SAIDA_MD.parent.mkdir(parents=True, exist_ok=True)

    with SAIDA_JSONL.open("w", encoding="utf-8") as f:
        for linha in linhas:
            f.write(json.dumps(linha, ensure_ascii=False) + "\n")

    total = len(linhas)
    partes = [
        "# Auditoria de amostra dos perfis de docente — achado 04",
        "",
        f"Gerado por `modulo1_etl/auditoria_perfis.py` em "
        f"{datetime.now(timezone.utc).astimezone():%Y-%m-%d %H:%M}.",
        "",
        f"- Amostra: **{total}** docentes sorteados de {n} pedidos (seed `{seed}`, reprodutível)",
        f"- Falhas de requisição: {len(falhas)}",
        "- Registro bruto: `docs/auditoria_perfis.jsonl`",
        "",
        "A pergunta desta auditoria é a do princípio 2 do CLAUDE.md: *se o docente "
        "tivesse preenchido o dado, nosso sistema o encontraria?* Ela separa o que "
        "falta por culpa nossa do que falta na fonte.",
        "",
        "## Como ler as colunas",
        "",
        "| Coluna | Significado |",
        "|---|---|",
        "| **Conteúdo** | o campo existe na página e tem texto real — é dado aproveitável |",
        "| **Placeholder** | o campo existe, mas o SIGAA gravou \"não informada\" — não é dado, e desde a correção do achado 03 não entra no texto indexado |",
        "| **Ausente** | o campo não aparece na página — lacuna da fonte, não é falha nossa |",
        "| **Parser captura** | quantos o código **corrigido** extrai hoje |",
        "| **Está no store** | quantos estão de fato no ChromaDB agora — se bater com a coluna anterior, não há perda nossa |",
        "",
        "## Campos que já coletamos",
        "",
        "| Campo | Conteúdo | Placeholder | Ausente | Parser captura | Está no store |",
        "|---|---|---|---|---|---|",
    ]

    for _, rotulo in CAMPOS_DO_PERFIL:
        c = Counter(l["campos"][rotulo]["na_pagina"] for l in linhas)
        captura = sum(1 for l in linhas if l["campos"][rotulo]["parser_novo_capturou"])
        store = sum(1 for l in linhas if l["campos"][rotulo]["no_store"])
        partes.append(
            f"| {rotulo} | {c['conteudo']} | {c['placeholder']} | {c['ausente']} "
            f"| {captura} | {store} |"
        )

    com_conteudo_areas = sum(
        1 for l in linhas if l["campos"]["Áreas de interesse"]["na_pagina"] == "conteudo"
    )
    chave_antiga_ok = sum(1 for l in linhas if l["areas_chave_antiga_casava"])

    # Só rende seção se ainda houver campo por coletar. Com a lista vazia, uma
    # tabela sem linhas nenhuma sugere "medimos e deu zero", que é diferente de
    # "não há mais o que coletar nesta aba".
    partes += [
        "",
        "## O achado 01, quantificado",
        "",
        f"- Perfis em que a chave antiga (com espaço) casava: **{chave_antiga_ok} de {total}**",
        f"- Perfis com áreas de interesse de conteúdo real na página: **{com_conteudo_areas} de {total}**",
        f"- Perfis com o campo no store hoje: "
        f"**{sum(1 for l in linhas if l['campos']['Áreas de interesse']['no_store'])} de {total}**",
    ]

    # A seção só existe se ainda houver campo por coletar. Com a lista vazia, uma
    # tabela sem linhas sugere "medimos e deu zero", que é bem diferente de "não
    # há mais o que coletar nesta aba".
    if CAMPOS_NAO_COLETADOS:
        partes += [
            "",
            "## Campos que a página oferece e nós ainda não coletamos (achado 05)",
            "",
            "Já vêm na mesma requisição que já fazemos — incluí-los não custa acesso "
            "nenhum ao servidor da UFRRJ.",
            "",
            "| Campo | Conteúdo | Placeholder | Ausente |",
            "|---|---|---|---|",
        ]
        for _, rotulo in CAMPOS_NAO_COLETADOS:
            c = Counter(l["nao_coletados"][rotulo] for l in linhas)
            partes.append(f"| {rotulo} | {c['conteudo']} | {c['placeholder']} | {c['ausente']} |")
    else:
        partes += [
            "",
            "## Campos da aba de docentes ainda não coletados",
            "",
            "**Nenhum.** Todos os campos que a página do docente oferece entraram em "
            "`CAMPOS_DO_PERFIL` em 4 set 2026. O que falta agora são as outras abas "
            "do SIGAA, que é o achado 05 e exige requisições novas.",
        ]

    # O veredito que interessa: do que está faltando, quanto é nosso?
    total_campos = total * len(CAMPOS_DO_PERFIL)
    tem_conteudo = sum(
        1
        for l in linhas
        for _, r in CAMPOS_DO_PERFIL
        if l["campos"][r]["na_pagina"] == "conteudo"
    )
    perdidos = sum(
        1
        for l in linhas
        for _, r in CAMPOS_DO_PERFIL
        if l["campos"][r]["na_pagina"] == "conteudo" and not l["campos"][r]["no_store"]
    )
    placeholders = sum(
        1
        for l in linhas
        for _, r in CAMPOS_DO_PERFIL
        if l["campos"][r]["na_pagina"] == "placeholder"
    )

    pct = (perdidos / tem_conteudo * 100) if tem_conteudo else 0.0
    partes += [
        "",
        "## Veredito",
        "",
        f"- Campos com conteúdo real na página: **{tem_conteudo}** de {total_campos} possíveis",
        f"- Desses, **não** estão no store hoje: **{perdidos}** (**{pct:.0f}%**) — é a falha nossa",
        f"- Campos que são só placeholder \"não informada\": **{placeholders}** "
        f"({placeholders / total_campos * 100:.0f}% do total) — corretamente **fora** do texto indexado desde o achado 03",
        "",
    ]

    if falhas:
        partes += ["## Requisições que falharam", ""] + [f"- {f}" for f in falhas] + [""]

    SAIDA_MD.write_text("\n".join(partes), encoding="utf-8")
    print(f"[AUDITORIA] relatório em {SAIDA_MD}")
    print(f"[AUDITORIA] registro bruto em {SAIDA_JSONL}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Auditoria de amostra dos perfis (achado 04).")
    parser.add_argument("--n", type=int, default=40, help="tamanho da amostra")
    parser.add_argument("--seed", type=int, default=42, help="semente do sorteio")
    parser.add_argument(
        "--somente-relatorio",
        action="store_true",
        help="regenera o markdown a partir do jsonl já gravado, sem tocar no SIGAA",
    )
    args = parser.parse_args()

    # O jsonl é o dado primário; o markdown é uma projeção dele. Quando só a
    # redação do relatório muda, refazer as 40 requisições ao servidor da UFRRJ
    # seria desperdício — e ainda por cima produziria uma amostra diferente da
    # que já foi discutida.
    if args.somente_relatorio:
        if not SAIDA_JSONL.exists():
            print(f"[AUDITORIA] {SAIDA_JSONL} não existe — rode a auditoria antes.", file=sys.stderr)
            raise SystemExit(1)
        registros = [
            json.loads(linha)
            for linha in SAIDA_JSONL.read_text(encoding="utf-8").splitlines()
            if linha.strip()
        ]
        escrever_relatorio(registros, args.n, args.seed, [])
        return

    docentes = amostrar_docentes(args.n, args.seed)
    print(f"[AUDITORIA] auditando {len(docentes)} perfis (seed {args.seed})...")

    linhas: list[dict] = []
    falhas: list[str] = []

    with httpx.Client(
        headers=config.HTTP_HEADERS, timeout=30.0, follow_redirects=True
    ) as cliente:
        for i, docente in enumerate(docentes, 1):
            try:
                linha = auditar_um(cliente, docente)
                linhas.append(linha)
                marca = "ok " if linha["pagina_valida"] else "SUSPEITA"
                areas = linha["campos"]["Áreas de interesse"]["na_pagina"]
                print(f"  [{i:>3}/{len(docentes)}] {marca} {docente['siape']:>8} "
                      f"areas={areas:<11} {docente['nome'][:40]}")
            except Exception as erro:
                falhas.append(f"siape {docente['siape']} ({docente['nome']}): {erro}")
                print(f"  [{i:>3}/{len(docentes)}] FALHA {docente['siape']}: {erro}")
            time.sleep(DELAY)

    if not linhas:
        # Relatório vazio que conclui com sucesso é o modo de falha que este
        # projeto trata como inaceitável — melhor sair com erro.
        print("[AUDITORIA] nenhuma página auditada com sucesso.", file=sys.stderr)
        raise SystemExit(1)

    escrever_relatorio(linhas, args.n, args.seed, falhas)


if __name__ == "__main__":
    main()
