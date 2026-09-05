"""
Suíte do gold set — o checker v2a só avança se passar em 100% dela.

ESTES TESTES FALHAM ATÉ A FASE 4 EXISTIR, e isso é o desenho. Eles são a
especificação executável escrita antes da implementação, para que a
implementação seja julgada contra eles em vez de o contrário. Se algum falhar
depois do v2a pronto, a pergunta a fazer é qual das duas coisas está errada:

    - BUG DE IMPLEMENTAÇÃO  -> conserta-se o código
    - A REGRA DISCORDA DO RÓTULO -> exige decisão humana, não conserto

São coisas diferentes e não podem ser confundidas na pressa de ficar verde.

INTERFACE QUE O v2a TEM DE OFERECER
    _conferir(checagem, verdade, resposta, afirmados, desempate_anaforico=False)
        devolve dict com "veredito" em {"APROVA", "REPROVA", "AMBIGUO"}
    _conferir("listagem", ..., recall=True)   # v2b, exploratória
"""

import pytest

from interfaces.conjunto_avaliacao import CONJUNTO, _docentes
from testes.gold_checker.casos import APROVA, GOLD, texto

POR_ID = {p.id: p for p in CONJUNTO}
POR_CASO = {c["id"]: c for c in GOLD}
CHECAGEM_ATRIBUICAO = "precisao_de_atribuicao_departamental"


def _checker():
    """Importa tardiamente: até a Fase 4 isto não existe, e o skip diz por quê."""
    from interfaces.comparar import _conferir

    import inspect

    if "desempate_anaforico" not in inspect.signature(_conferir).parameters:
        pytest.skip("checker v2a ainda não implementado (Fase 4)")
    return _conferir


def _julgar(caso, **kwargs):
    conferir = _checker()
    pergunta = POR_ID[caso["pergunta_id"]]
    resposta = texto(caso["arquivo"])
    from interfaces.comparar import nomes_afirmados

    return conferir(
        CHECAGEM_ATRIBUICAO,
        pergunta.verdade(),
        resposta,
        nomes_afirmados(resposta),
        **kwargs,
    )


class TestPremissasDoGoldSet:
    """
    O gold set afirma coisas sobre a base. Se a base mudar, os rótulos
    apodrecem em silêncio e a suíte passa a medir a defasagem do arquivo.
    """

    @pytest.mark.parametrize("caso", [c for c in GOLD if isinstance(c.get("base"), dict)],
                             ids=lambda c: c["id"])
    def test_o_departamento_afirmado_no_rotulo_confere_com_a_base(self, caso):
        from interfaces.comparar import _normalizar

        real = {_normalizar(r.get("nome", "")): r.get("departamento", "") for r in _docentes()}
        for nome, departamento in caso["base"].items():
            assert _normalizar(real.get(nome, "")) == _normalizar(departamento), (
                f"{caso['id']}: a base diz {real.get(nome)!r} para {nome}, "
                f"o gold set afirma {departamento!r}"
            )

    @pytest.mark.parametrize("caso", GOLD, ids=lambda c: c["id"])
    def test_todo_caso_tem_fixture_e_justificativa(self, caso):
        assert texto(caso["arquivo"]).strip(), caso["id"]
        assert caso["porque"].strip(), caso["id"]
        assert caso["origem"].startswith(("REAL", "SINTÉTICA")), caso["id"]

    @pytest.mark.parametrize("caso", [c for c in GOLD if c["origem"].startswith("SINTÉTICA")],
                             ids=lambda c: c["id"])
    def test_sintetica_declara_a_mutacao(self, caso):
        """Sintética sem diff registrado é texto inventado disfarçado de evidência."""
        assert caso.get("mutacao"), caso["id"]

    def test_o_gold_set_cobre_os_tres_vereditos(self):
        assert {c["esperado"] for c in GOLD} == {"APROVA", "REPROVA", "AMBIGUO"}

    def test_existe_reprovacao_no_gold_set(self):
        """
        Com ZERO reprovações em dado real, as sintéticas são a única prova de
        que o instrumento morde. Se sumirem, a suíte não testa mais nada.
        """
        assert sum(1 for c in GOLD if c["esperado"] == "REPROVA") >= 2


class TestVereditos:
    @pytest.mark.parametrize("caso", GOLD, ids=lambda c: c["id"])
    def test_veredito_primario(self, caso):
        resultado = _julgar(caso)
        assert resultado["veredito"] == caso["esperado"], (
            f"{caso['id']}: esperado {caso['esperado']}, obtido "
            f"{resultado['veredito']}\n  {caso['porque']}"
        )

    @pytest.mark.parametrize(
        "caso", [c for c in GOLD if "esperado_variante_anaforica" in c], ids=lambda c: c["id"]
    )
    def test_veredito_com_desempate_anaforico(self, caso):
        resultado = _julgar(caso, desempate_anaforico=True)
        assert resultado["veredito"] == caso["esperado_variante_anaforica"], caso["porque"]


class TestParDeUtilidade:
    """
    (k) A métrica é CEGA À UTILIDADE, e isso é fixado em código.

    k1 despeja 35 de 35 docentes do departamento; k2 traz 3 com evidência
    temática de verdade. Utilidades opostas, veredito idêntico. O teste passa se
    e somente se forem iguais — não mede correção da regra, mede o que ela
    ignora.
    """

    def test_despejo_e_resposta_util_recebem_o_mesmo_veredito(self):
        k1 = _julgar(POR_CASO["k1_despejo_do_departamento_inteiro"])
        k2 = _julgar(POR_CASO["k2_tres_uteis"])
        assert k1["veredito"] == k2["veredito"] == APROVA, (
            "35 indiscriminados e 3 criteriosos têm de receber o MESMO veredito. "
            f"k1={k1['veredito']} k2={k2['veredito']}"
        )

    def test_a_diferenca_de_utilidade_e_real_e_nao_so_de_tamanho(self):
        k1 = texto("real_amb02_r3_despejo")
        k2 = texto("sint_k2_tres_uteis")
        assert k1.count("\n- ") == 35
        assert k2.count("\n- ") == 3


class TestSeparacaoEntreV2aEV2b:
    """
    (h) Omitir docente do elenco não é erro de atribuição, e é erro de listagem.
    Se os dois vereditos coincidirem, as regras estão contaminadas.
    """

    def test_omissao_aprova_em_atribuicao(self):
        caso = POR_CASO["h_omissao_de_docente_do_elenco"]
        assert _julgar(caso)["veredito"] == APROVA

    def test_omissao_reprova_em_listagem_sob_v2b(self):
        from interfaces.comparar import _conferir, nomes_afirmados

        import inspect

        if "recall" not in inspect.signature(_conferir).parameters:
            pytest.skip("v2b ainda não implementada (Fase 4)")
        caso = POR_CASO["h_omissao_de_docente_do_elenco"]
        resposta = texto(caso["arquivo"])
        resultado = _conferir(
            "listagem", POR_ID[caso["pergunta_id"]].verdade(), resposta,
            nomes_afirmados(resposta), recall=True,
        )
        assert resultado["veredito"] == "REPROVA"
