"""
O registro de tipos concorda com a realidade — antes de alguém depender dele.

`interfaces/tipos.py` é o passo 3 da ordem do D0, e é declarativo de propósito.
Este arquivo é o que torna esse passo seguro: o registro é conferido contra o
`TOOLS_SCHEMA` que está no ar HOJE, e contra `interfaces/identidade.py`, antes
que o passo 4 passe a gerar as tools a partir dele.

⚠️ POR QUE A IGUALDADE É LITERAL
---------------------------------
As descrições das tools são o artefato mais medido do projeto. Cada uma foi
reescrita para consertar uma falha de roteamento observada — a de
`buscar_docente_por_nome` custou seis execuções na bateria de 5 set 2026 — e o
roteamento de 97,8% da fase 3 é uma propriedade **desses textos**, não do
sistema em abstrato.

Se o registro gerar uma descrição parecida em vez da mesma, o 97,8% deixa de
descrever o que está rodando, e ninguém fica sabendo. Por isso a comparação é
caractere a caractere.
"""

import pytest

from interfaces import identidade as ident
from interfaces.tipos import TIPOS, Tipo, tipos_do_chroma
from modulo2_inferencia.tools import TOOLS_SCHEMA

# A tool semântica não é uma "busca por campo": não filtra dado estruturado, e
# por isso não sai do registro de buscas. Fica de fora da comparação, e este
# nome existe para que a exclusão seja explícita em vez de acidental.
TOOL_SEMANTICA = "busca_vetorial_sigaa"


def _schema_por_nome() -> dict:
    return {
        s["function"]["name"]: s["function"]
        for s in TOOLS_SCHEMA
        if s["function"]["name"] != TOOL_SEMANTICA
    }


class TestRegistroCobreOQueEstaNoAr:
    def test_as_mesmas_tools_estruturadas(self):
        no_ar = set(_schema_por_nome())
        no_registro = {b.nome_tool for t in TIPOS.values() for b in t.buscas}
        assert no_registro == no_ar, (
            f"registro tem {no_registro}, TOOLS_SCHEMA tem {no_ar}. "
            "Uma tool que o registro nao conhece nao seria gerada no passo 4, "
            "e sumiria do agente sem erro nenhum."
        )

    @pytest.mark.parametrize(
        "busca",
        [b for t in TIPOS.values() for b in t.buscas],
        ids=lambda b: b.nome_tool,
    )
    def test_descricao_identica_caractere_a_caractere(self, busca):
        atual = _schema_por_nome()[busca.nome_tool]
        assert busca.descricao == atual["description"]

    @pytest.mark.parametrize(
        "busca",
        [b for t in TIPOS.values() for b in t.buscas],
        ids=lambda b: b.nome_tool,
    )
    def test_parametro_identico(self, busca):
        atual = _schema_por_nome()[busca.nome_tool]
        props = atual["parameters"]["properties"]
        assert list(props) == [busca.parametro]
        assert props[busca.parametro]["description"] == busca.descricao_parametro
        assert atual["parameters"]["required"] == [busca.parametro]


class TestRegistroConcordaComIdentidade:
    """
    O registro e `interfaces/identidade.py` calculam a mesma coisa por caminhos
    diferentes. Se divergirem, um documento teria uma identidade ao ser gravado
    e outra ao ser lido — e nada acusaria.
    """

    @staticmethod
    def _entidade_docente() -> dict:
        # A forma que sai do SQLite (`nome`) e a que sai do metadado do Chroma
        # (`nome_docente`) sao DIFERENTES. O registro tem de aceitar as duas.
        return {"nome": "FILIPE BRAIDA DO CARMO", "departamento": "DCC/IM",
                "siape": "1234567"}

    def test_identidade_bate_com_o_leitor(self):
        e = self._entidade_docente()
        assert TIPOS["docente"].identidade(e) == ident.identidade(e)

    def test_rotulo_aceita_as_duas_formas(self):
        e = self._entidade_docente()
        meta = {"nome_docente": e["nome"], "siape": e["siape"]}
        assert TIPOS["docente"].rotulo(e) == e["nome"]
        assert TIPOS["docente"].rotulo(meta) == e["nome"]
        assert ident.rotulo(meta) == e["nome"]


class TestEstruturaPuraNaoVaiParaOChroma:
    """
    A correção do D2: `texto_semantico=None` significa que o tipo não é
    vetorizado. Indexar um registro sem texto livre é vetorizar um nome — o
    defeito que o item 7 mediu e removeu para levar o recall de 14% a 27%.
    """

    def test_docente_vai(self):
        assert TIPOS["docente"].vai_para_o_chroma
        assert "docente" in tipos_do_chroma()

    def test_tipo_sem_texto_semantico_fica_de_fora(self):
        estrutural = Tipo(
            nome="departamento",
            identidade=lambda e: f"departamento:{e['id_sigaa']}",
            rotulo=lambda e: e["nome"],
            texto_semantico=None,
            buscas=(),
        )
        assert not estrutural.vai_para_o_chroma


class TestMetadadosTemOMesmoFormatoParaTodoTipo:
    def test_campos_obrigatorios_do_D1(self):
        meta = TIPOS["docente"].metadados(
            {"nome": "ANA SILVA", "siape": "99", "source_url": "http://x",
             "scraped_at": "2026-09-07"}
        )
        assert meta["tipo"] == "docente"
        assert meta["id_entidade"] == "docente:99"
        assert meta["rotulo"] == "ANA SILVA"
        assert set(meta) >= {"tipo", "id_entidade", "rotulo", "source_url",
                             "scraped_at", "instancia_dona"}
