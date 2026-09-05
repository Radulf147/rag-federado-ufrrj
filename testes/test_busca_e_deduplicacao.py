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


class TestOsDoisZeros:
    """
    ARMADILHA 4 — "nao ha docentes" tem duas causas muito diferentes.

    Nenhum docente naquele departamento e resposta legitima. Base vazia e falha
    de infraestrutura. Ate 5 set 2026 as duas saiam com o mesmo texto, e a
    segunda virava a resposta errada mais convincente que este sistema sabe dar:
    dita com seguranca, verificavel, e completamente falsa.
    """

    def test_base_vazia_denuncia_infraestrutura(self, tmp_path, monkeypatch):
        # PATCH NO MODULO CERTO. `import db_manager` (plano) e
        # `modulo1_etl.db_manager` (qualificado) sao DOIS objetos de modulo
        # distintos para o mesmo arquivo, cada um com seus proprios globais —
        # consequencia da convencao de imports mista. tools.py usa o
        # qualificado, entao e nele que o DB_PATH precisa ser trocado.
        import modulo1_etl.db_manager as dbm_qualificado
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(dbm_qualificado, "DB_PATH", str(tmp_path / "vazio.db"))
        resposta = tools.buscar_docentes_por_departamento("Matemática")

        assert "FALHA" in resposta
        assert "não foi carregado" in resposta or "caminho está errado" in resposta
        # O ponto todo: nao pode soar como ausencia real de docentes.
        assert "Não encontrei nenhum docente registrado" not in resposta

    def test_base_populada_e_departamento_inexistente_responde_ausencia(self, tmp_path, monkeypatch):
        import json
        import sqlite3

        import modulo2_inferencia.tools as tools

        import modulo1_etl.db_manager as dbm_qualificado

        caminho = tmp_path / "cheio.db"
        monkeypatch.setattr(dbm_qualificado, "DB_PATH", str(caminho))
        dbm_qualificado.init_db()
        with sqlite3.connect(caminho) as conn:
            conn.execute(
                "INSERT INTO entidades_sigaa (tipo_entidade, dados_brutos) VALUES (?, ?)",
                ("docente", json.dumps({"nome": "FULANO", "departamento": "DEPARTAMENTO DE FISICA"})),
            )

        resposta = tools.buscar_docentes_por_departamento("Veterinária")
        assert "FALHA" not in resposta
        assert "Não encontrei nenhum docente" in resposta
