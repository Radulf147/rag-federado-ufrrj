"""
Regressão dos defeitos de BUSCA e DEDUPLICAÇÃO — acentuação, duplicatas, `think`.
"""

import pytest
from haystack import Document

import db_manager
import deduplicacao


class TestBuscaCegaAAcentos:
    """
    O `LIKE` do SQLite é case-insensitive só para ASCII. Medido em Set/2026:
    LIKE '%Ciência da Computação%' devolvia 0 e '%CIÊNCIA DA COMPUTAÇÃO%'
    devolvia 6. Como o SIGAA grava em CAIXA ALTA e o LLM escreve o argumento em
    caixa mista com acento, a tool falhava em quase toda pergunta — e o agente
    respondia, honestamente, que não havia docentes no departamento.
    """

    def test_caixa_e_acento_convergem(self):
        assert db_manager.normalizar("Ciência da Computação") == db_manager.normalizar(
            "CIÊNCIA DA COMPUTAÇÃO"
        )

    def test_acento_decomposto_converge(self):
        # ATENÇÃO ao ler: as duas strings abaixo parecem iguais na tela e não
        # são. A primeira traz "e" + acento combinante (NFD); a segunda, "ê"
        # pré-composto (NFC). É a mesma armadilha do achado 01.
        assert db_manager.normalizar("Ciência") == db_manager.normalizar("Ciência")

    def test_palavras_diferentes_nao_convergem(self):
        assert db_manager.normalizar("Matemática") != db_manager.normalizar("Física")


class TestDeduplicacao:
    """
    A chave é (nome, departamento), NÃO a siape.

    Isso se provou certo por acidente feliz: o achado 08 mostrou que a siape não
    identifica a pessoa de forma confiável entre execuções, e uma deduplicação
    ancorada nela teria herdado o problema.
    """

    @staticmethod
    def _perfil(nome, depto, siape, conteudo="x"):
        return Document(
            content=conteudo,
            meta={
                "content_type": "docente_perfil",
                "nome_docente": nome,
                "departamento": depto,
                "siape": str(siape),
            },
        )

    def test_mesma_pessoa_sob_siapes_diferentes_colapsa(self):
        docs = [
            self._perfil("FULANO DE TAL", "DEPARTAMENTO X", 111),
            self._perfil("FULANO DE TAL", "DEPARTAMENTO X", 222),
        ]
        mantidos, removidos = deduplicacao.deduplicar_documentos(docs)
        assert removidos == 1
        assert len(mantidos) == 1

    def test_homonimos_em_departamentos_diferentes_sobrevivem(self):
        """Mesmo nome em departamentos distintos são duas pessoas diferentes."""
        docs = [
            self._perfil("FULANO DE TAL", "DEPARTAMENTO X", 111),
            self._perfil("FULANO DE TAL", "DEPARTAMENTO Y", 222),
        ]
        mantidos, removidos = deduplicacao.deduplicar_documentos(docs)
        assert removidos == 0
        assert len(mantidos) == 2

    def test_documento_que_nao_e_perfil_passa_intacto(self):
        outro = Document(content="qualquer coisa", meta={"content_type": "outra_coisa"})
        mantidos, removidos = deduplicacao.deduplicar_documentos([outro])
        assert removidos == 0
        assert mantidos == [outro]


class TestReasoningEffort:
    """
    `reasoning_effort` dentro de `options` era ignorado em silêncio pelo Ollama;
    o controle real é o parâmetro `think`, no topo do payload.
    """

    def test_auto_nao_envia_think(self):
        """
        `auto` devolve None de propósito: mandar `think` para um modelo sem
        raciocínio configurável — como o qwen2.5, o modelo atual — é erro, não
        no-op.
        """
        from modulo2_inferencia.llm_setup import _valor_de_think

        assert _valor_de_think("auto") is None
        assert _valor_de_think("") is None

    def test_desligado_vira_false(self):
        from modulo2_inferencia.llm_setup import _valor_de_think

        for valor in ["off", "false", "none", "no"]:
            assert _valor_de_think(valor) is False, valor

    def test_niveis_viram_string(self):
        from modulo2_inferencia.llm_setup import _valor_de_think

        for valor in ["low", "medium", "high"]:
            assert _valor_de_think(valor) == valor

    def test_valor_invalido_falha_alto(self):
        """Falha barulhenta em vez de degradação silenciosa."""
        from modulo2_inferencia.llm_setup import _valor_de_think

        with pytest.raises(ValueError):
            _valor_de_think("altissimo")
