"""
CRITÉRIO DE ACEITE DO D0 — a saída das tools não pode mudar.

`testes/instantaneos/tools_antes_do_registro.json` foi gravado em 8 set 2026,
com o código anterior à geração pelo registro (git `704a5bb`), rodando contra o
corpus real de 1302 docentes. Onze casos, escolhidos para cobrir os caminhos
que produzem texto diferente:

    ambiguidade de departamento ..... 'Computação', 'Física'
    departamento único com listagem . 'Ciência da Computação'
    nada encontrado ................. 'Departamento de Alquimia'
    argumento vazio ................. ''
    nome que casa um ................ 'Filipe Braida', 'Marcel...'
    nome que casa nenhum ............ 'Leandro Alvim' (item 9 do backlog)
    nome que casa muitos ............ 'Silva'

POR QUE ISTO EXISTE
-------------------
A reestruturação do D0 troca `TOOLS_SCHEMA` literal e `criar_dispatcher` escrito
à mão por versões geradas a partir de `interfaces/tipos.py`. O roteamento de
97,8% da fase 3 e as 36 respostas do teste de coleção são propriedades do texto
que essas tools devolvem e das descrições que anunciam. Se qualquer um mudar, a
fase 3 deixa de descrever o sistema que está rodando — **e nada acusaria**,
porque a saída nova seria igualmente plausível.

Este teste é a única coisa que transforma "eu tomei cuidado" em verificação.

⚠️ ESTE ARQUIVO DEPENDE DO CORPUS. Ele lê o SQLite real, e as contagens (15
docentes no DCC/IM, 26 na Física) são do retrato carregado em 4 set 2026. Uma
recarga do ETL que mude o corpus faz ele falhar **legitimamente** — e aí o
instantâneo precisa ser regravado de propósito, não o teste afrouxado.
"""

import json
from pathlib import Path

import pytest

INSTANTANEO = (
    Path(__file__).parent / "instantaneos" / "tools_antes_do_registro.json"
)


@pytest.fixture(scope="module")
def gravado() -> dict:
    if not INSTANTANEO.exists():
        pytest.skip(f"instantaneo ausente: {INSTANTANEO}")
    return json.loads(INSTANTANEO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def casos(gravado) -> list:
    return [(k, v) for k, v in gravado.items() if not k.startswith("__")]


class TestSaidaDasToolsEstruturadas:
    def test_todos_os_casos_batem(self, casos):
        from modulo2_inferencia import tools

        divergentes = []
        for chave, caso in casos:
            funcao = getattr(tools, caso["tool"])
            agora = funcao(caso["argumento"])
            if agora != caso["retorno"]:
                divergentes.append((chave, caso["retorno"], agora))

        if divergentes:
            partes = [
                f"\n--- {chave}\n  ANTES: {antes!r}\n  AGORA: {depois!r}"
                for chave, antes, depois in divergentes
            ]
            pytest.fail(
                f"{len(divergentes)} de {len(casos)} casos mudaram de saida."
                + "".join(partes)
            )

    def test_o_instantaneo_nao_esta_vazio(self, casos):
        # Um instantaneo vazio faria o teste acima passar sem medir nada --
        # exatamente o "teste que nao distingue o que mediu" da setima
        # ocorrencia do relatorio.
        assert len(casos) >= 11


class TestSchemaAnunciadoAoLLM:
    """
    As descrições são o que decide o roteamento. Comparadas inteiras, não por
    nome de tool: uma descrição reescrita mantém o nome e muda o comportamento.
    """

    def test_schema_identico(self, gravado):
        from modulo2_inferencia.tools import TOOLS_SCHEMA

        assert TOOLS_SCHEMA == gravado["__schema__"]
