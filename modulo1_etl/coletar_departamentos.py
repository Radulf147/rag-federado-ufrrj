"""
Etapa 1 do plano de coleta: os departamentos do portal público.

    docker compose run --rm agente python -m modulo1_etl.coletar_departamentos

Plano em `docs/plano_coleta_portal_publico.md`; arquitetura em
`docs/arquitetura_multi_entidade.md`.

É a MENOR PEÇA ÚTIL, e é de propósito: prova o caminho de coleta e persistência
para um tipo de entidade que **não é docente** — a suposição sobre a qual o
projeto inteiro foi construído sem nunca escrevê-la.

NÃO INDEXA NO CHROMA. Isto grava só no SQLite. Indexar exige a decisão D1
(`id_entidade` para identidade e `rotulo` para exibição, no lugar de
`nome_docente` fazendo os dois), que mexe em `tools.py` — código que as
medições atuais usam — e por isso é passo próprio, com aval próprio. Os dois
campos já saem gravados daqui.

⚠️ POR QUE NÃO SÃO 2 REQUISIÇÕES, COMO O PLANO DIZIA
-----------------------------------------------------
O plano previa `POST` com `-- TODOS --` e pronto: 72 departamentos numa
requisição. Isso funciona **para os nomes**, e só.

A listagem de TODOS traz **uma única** linha de centro (`td.subListagem`,
"INSTITUTO DE AGRONOMIA") para os 72 departamentos. Quem escrevesse o parser da
forma óbvia — "o centro é o último `subListagem` visto" — atribuiria os 72 ao
Instituto de Agronomia. **Plausível, silencioso e inteiramente errado**, e o
número final (72) continuaria certo.

O vínculo departamento→centro sai de um `POST` por centro. São 15 centros, e
cada um exige um `GET` do formulário antes (estado JSF). Custa 32 requisições
em vez de 2, e é o preço de ter a chave explícita que a decisão D4 pede.

O `POST` de `-- TODOS --` continua sendo feito, agora como **CONTROLE**: a
união dos 15 centros tem de dar exatamente o mesmo conjunto. Divergência é
erro, e é relatada.

⚠️ CLIENTE ISOLADO POR CONSULTA (achado 08)
--------------------------------------------
O SIGAA guarda estado no servidor atrelado ao cookie. Foi isso que fez o
scraper de docentes trazer 6 de 15 perfis com a pessoa errada. Aqui cada
consulta abre o seu próprio cliente, mesmo sendo tudo em série: é barato, e a
disciplina não depende de lembrar dela na próxima vez.
"""

import argparse
import re
import time
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

import config
from modulo1_etl.db_manager import salvar_entidades

URL = "https://sigaa.ufrrj.br/sigaa/public/departamento/lista.jsf?aba=p-academico"
BASE = "https://sigaa.ufrrj.br"
DELAY = 1.0  # segundos entre consultas

# O SIGAA serve iso-8859-1. Decodificar como UTF-8 não dá erro: dá acento
# corrompido, em silêncio, no meio dos nomes.
CODIFICACAO = "iso-8859-1"

_ID = re.compile(r"[?&]id=(\d+)")


def _texto(html: bytes) -> BeautifulSoup:
    return BeautifulSoup(html.decode(CODIFICACAO, "replace"), "lxml")


def _formulario(sopa: BeautifulSoup) -> tuple[str, dict, list]:
    """Devolve (url_do_post, campos_ocultos, opcoes_do_select)."""
    form = sopa.find("form")
    campos = {
        i.get("name"): i.get("value", "")
        for i in form.find_all("input")
        if i.get("name") and i.get("type") != "submit"
    }
    opcoes = [
        (o.get("value"), o.get_text(strip=True))
        for o in form.find("select").find_all("option")
    ]
    return BASE + form.get("action"), campos, opcoes


def consultar(valor_centro: str) -> list[tuple[str, int]]:
    """
    Um centro (ou '0' para todos) -> [(nome_do_departamento, id_sigaa)].

    Cliente próprio, aberto e fechado aqui: ver o cabeçalho sobre o achado 08.
    """
    with httpx.Client(
        headers=config.HTTP_HEADERS, timeout=45, follow_redirects=True
    ) as cliente:
        url_post, campos, _ = _formulario(_texto(cliente.get(URL).content))
        time.sleep(DELAY)
        campos["form:programas"] = valor_centro
        campos["form:buscar"] = "Buscar"
        sopa = _texto(cliente.post(url_post, data=campos).content)

    tabelas = [t for t in sopa.find_all("table") if "listagem" in (t.get("class") or [])]
    if not tabelas:
        return []

    achados = []
    for a in tabelas[0].find_all("a", href=True):
        nome = a.get_text(" ", strip=True)
        casamento = _ID.search(a["href"])
        # Âncoras vazias existem na listagem e não são departamento nenhum.
        if nome and casamento:
            achados.append((nome, int(casamento.group(1))))
    return achados


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gravar", action="store_true",
                        help="grava no SQLite; sem isto so imprime (ensaio)")
    args = parser.parse_args()

    print("=" * 76)
    print("INTERMEDIARIOS")
    print("=" * 76)
    with httpx.Client(
        headers=config.HTTP_HEADERS, timeout=45, follow_redirects=True
    ) as c:
        _, _, opcoes = _formulario(_texto(c.get(URL).content))
    centros = [(v, t) for v, t in opcoes if v and v != "0"]
    print(f"  centros no formulario ... {len(centros)}")
    print(f"  requisicoes previstas ... {2 * (len(centros) + 1)}"
          f"  (GET+POST por consulta, {len(centros)} centros + o controle)")
    print(f"  db ..................... {config.DB_PATH}")

    print()
    print("=" * 76)
    print("CONTROLE -- a listagem de TODOS")
    print("=" * 76)
    todos = consultar("0")
    print(f"  departamentos em '-- TODOS --': {len(todos)}")

    print()
    print("=" * 76)
    print("POR CENTRO -- e daqui sai o vinculo")
    print("=" * 76)
    por_centro: dict[int, tuple[str, str, int]] = {}
    for valor, nome_centro in centros:
        time.sleep(DELAY)
        deps = consultar(valor)
        print(f"  {len(deps):3}  {nome_centro[:58]}")
        for nome_dep, id_dep in deps:
            if id_dep in por_centro:
                # Um departamento em dois centros é ambiguidade a RELATAR, não
                # a resolver escolhendo o primeiro (mesma disciplina do achado 06).
                anterior = por_centro[id_dep][1]
                print(f"       ⚠️ {nome_dep} tambem em {anterior}")
            por_centro[id_dep] = (nome_dep, nome_centro, int(valor))

    print()
    print("=" * 76)
    print("CONFERENCIA")
    print("=" * 76)
    ids_todos = {i for _, i in todos}
    ids_centros = set(por_centro)
    print(f"  '-- TODOS --' .......... {len(ids_todos)}")
    print(f"  uniao dos centros ...... {len(ids_centros)}")
    faltando = ids_todos - ids_centros
    sobrando = ids_centros - ids_todos
    if faltando or sobrando:
        # Não é para abortar: o dado dos que casam continua bom, e esconder a
        # diferença seria pior que relatá-la.
        print(f"  ⚠️ so em TODOS: {len(faltando)}  |  so nos centros: {len(sobrando)}")
        nomes = dict(todos)
        for i in list(faltando)[:10]:
            print(f"     sem centro: {[n for n, x in todos if x == i]}")
    else:
        print("  ✅ os dois caminhos concordam")

    agora = datetime.now(timezone.utc).isoformat()
    entidades = []
    for id_dep, (nome_dep, nome_centro, id_centro) in sorted(
        por_centro.items(), key=lambda kv: kv[1][0]
    ):
        entidades.append({
            # D1 pede DOIS campos genericos, e eles fazem trabalhos diferentes.
            #
            # `id_entidade` e IDENTIDADE: dedupe, chave do RRF, e a juncao do
            # D4. Sai do id do SIGAA, que nao muda quando reindexamos -- ao
            # contrario do Document.id do Haystack, que e hash do conteudo e
            # deu tres valores diferentes para o mesmo docente nas tres
            # colecoes que existem hoje.
            #
            # `rotulo` e EXIBICAO, e so. Aqui coincide com o nome; em
            # componente curricular sera "codigo — nome". Nao serve de chave:
            # na listagem de cursos, CIENCIAS BIOLOGICAS aparece duas vezes
            # (Bacharelado e Licenciatura), mesmo campus, ids distintos.
            "id_entidade": f"departamento:{id_dep}",
            "rotulo": nome_dep,
            "nome": nome_dep,
            "id_sigaa": id_dep,
            "centro": nome_centro,
            # Chave do centro no SIGAA. Vira `centro:<id>` quando (e se) centro
            # for um tipo coletado; por ora e o id cru, que ja e a juncao.
            "centro_id": id_centro,
            "source_url": f"{BASE}/sigaa/public/departamento/portal.jsf?lc=pt_BR&id={id_dep}",
            "scraped_at": agora,
        })
    # Os que TODOS anuncia e nenhum centro reivindica entram mesmo assim, sem
    # centro. Perder um departamento real para manter o esquema limpo seria
    # trocar dado por arrumação.
    for nome_dep, id_dep in todos:
        if id_dep not in por_centro:
            entidades.append({
                "id_entidade": f"departamento:{id_dep}",
                "rotulo": nome_dep, "nome": nome_dep, "id_sigaa": id_dep,
                "centro": None, "centro_id": None,
                "source_url": f"{BASE}/sigaa/public/departamento/portal.jsf?lc=pt_BR&id={id_dep}",
                "scraped_at": agora,
            })

    print()
    print("=" * 76)
    print(f"RESULTADO -- {len(entidades)} departamentos")
    print("=" * 76)
    sem_centro = sum(1 for e in entidades if not e["centro"])
    print(f"  com centro ............. {len(entidades) - sem_centro}")
    print(f"  sem centro ............. {sem_centro}")
    print()
    for e in entidades[:6]:
        print(f"  {e['id_sigaa']:>6}  {e['nome'][:44]:46} {(e['centro'] or '-')[:28]}")
    print("  ...")

    if not args.gravar:
        print()
        print("ENSAIO -- nada gravado. Use --gravar para persistir.")
        return

    # substituir=True: cada execucao e um retrato COMPLETO deste tipo, nao um
    # incremento (achado 10). Nao toca em tipo_entidade='docente'.
    salvar_entidades("departamento", entidades, substituir=True)
    from modulo1_etl.db_manager import total_de_entidades
    print(f"\n  no SQLite agora: departamento={total_de_entidades('departamento')}"
          f"  docente={total_de_entidades('docente')}")


if __name__ == "__main__":
    main()
