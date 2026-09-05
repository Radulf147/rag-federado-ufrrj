"""
Testes do que MEDE — o instrumento da fase 3.

Um instrumento de medição errado é pior que nenhum: produz número plausível.
A primeira versão do calibrador de limiar mediu precisão sobre o corpus inteiro
ignorando o TOP_K, e recomendou um valor que descrevia um sistema inexistente.
Estes testes existem para que a bateria que decide o fim da fase 3 não sofra do
mesmo mal.
"""

import json

from interfaces.comparar import ROTA_POR_FONTES, avaliar, nomes_afirmados
from interfaces.conjunto_avaliacao import CONJUNTO, ROTAS_VALIDAS, validar
from modulo2_inferencia.pipelines import ResultadoPipeline

POR_ID = {p.id: p for p in CONJUNTO}


class TestConjuntoPreRegistrado:
    def test_o_conjunto_e_valido(self):
        validar()

    def test_toda_pergunta_tem_rota_conhecida(self):
        for pergunta in CONJUNTO:
            assert pergunta.rota in ROTAS_VALIDAS, pergunta.id

    def test_toda_pergunta_justifica_a_rota(self):
        """
        Rótulo sem justificativa é palpite; o pré-registro exige o porquê.

        O que se verifica é a EXISTÊNCIA de uma razão, não o tamanho dela.
        A primeira versão deste teste exigia mais de 20 caracteres — número
        que eu inventei e que reprovava "Listagem nominal.", uma justificativa
        curta e perfeitamente boa. Limiar arbitrário em teste mede o limiar,
        não a propriedade.
        """
        for pergunta in CONJUNTO:
            razao = pergunta.porque.strip()
            assert razao, pergunta.id
            assert razao.rstrip(".").lower() != pergunta.rota, (
                f"{pergunta.id}: a justificativa só repete o nome da rota"
            )

    def test_as_quatro_classes_estao_representadas(self):
        assert {p.rota for p in CONJUNTO} == ROTAS_VALIDAS

    def test_a_verdade_base_e_calculada_e_nao_literal(self):
        """
        Se fosse número escrito à mão, apodreceria na próxima recarga e a
        bateria mediria a defasagem do arquivo em vez do agente.
        """
        for pergunta in CONJUNTO:
            if pergunta.verdade is not None:
                assert callable(pergunta.verdade), pergunta.id


class TestRotaDeduzidaDasFerramentas:
    """A rota é fato registrado — que tool foi chamada —, não leitura do texto."""

    def test_mapeamento_completo(self):
        assert ROTA_POR_FONTES[frozenset()] == "nenhuma"
        assert ROTA_POR_FONTES[frozenset({"sqlite"})] == "estruturada"
        assert ROTA_POR_FONTES[frozenset({"chromadb"})] == "semantica"
        assert ROTA_POR_FONTES[frozenset({"sqlite", "chromadb"})] == "ambigua"


class TestChecagemDeAtribuicao:
    """
    Substitui a leitura no olho do critério de tolerância zero: todo docente que
    a resposta AFIRMA tem de aparecer no contexto que as ferramentas devolveram.
    """

    @staticmethod
    def _resultado(resposta, contexto, fontes=("chromadb",)):
        return ResultadoPipeline(
            pipeline="3-agente",
            pergunta="irrelevante",
            resposta=resposta,
            fontes=list(fontes),
            contexto=contexto,
        )

    def test_nome_sem_respaldo_e_flagrado(self):
        aval = avaliar(
            POR_ID["sem-01"],
            self._resultado(
                "Filipe Braida do Carmo pesquisa isso.",
                "Docente: BRUNO JOSE DEMBOGURSKI.",
            ),
        )
        assert not aval["atribuicao_ok"]
        assert "FILIPE BRAIDA DO CARMO" in aval["nomes_sem_respaldo"]

    def test_nome_com_respaldo_passa(self):
        aval = avaliar(
            POR_ID["sem-01"],
            self._resultado(
                "Filipe Braida do Carmo pesquisa isso.",
                "Docente: FILIPE BRAIDA DO CARMO. Areas: IA.",
            ),
        )
        assert aval["atribuicao_ok"]
        assert aval["nomes_sem_respaldo"] == []

    def test_resposta_sem_nome_nenhum_passa(self):
        """Dizer que não encontrou é resposta correta e não pode ser punida."""
        aval = avaliar(
            POR_ID["sem-01"], self._resultado("Não encontrei essa informação.", "")
        )
        assert aval["atribuicao_ok"]
        assert aval["nomes_afirmados"] == 0

    def test_detecta_nome_com_caixa_e_acento_diferentes(self):
        assert nomes_afirmados("segundo Marcel William Rocha da Silva, ...") == [
            "MARCEL WILLIAM ROCHA DA SILVA"
        ]


class TestVerdadeBaseDeDepartamento:
    """
    ACHADO 06 — somar departamentos homônimos é resposta errada.
    O nome "Geografia" casa com o de Seropédica e o do IM.
    """

    @staticmethod
    def _resultado(resposta):
        return ResultadoPipeline(
            pipeline="3-agente",
            pergunta="irrelevante",
            resposta=resposta,
            fontes=["sqlite"],
            contexto="",
        )

    def test_relatar_as_duas_contagens_e_correto(self):
        contagens = list(POR_ID["est-08"].verdade()["contagens"].values())
        resposta = f"São dois: um com {contagens[0]} e outro com {contagens[1]} docentes."
        aval = avaliar(POR_ID["est-08"], self._resultado(resposta))
        assert aval["verdade"]["faltando"] == []
        assert not aval["verdade"]["soma_indevida"]

    def test_somar_os_dois_e_flagrado(self):
        soma = sum(POR_ID["est-08"].verdade()["contagens"].values())
        aval = avaliar(POR_ID["est-08"], self._resultado(f"Tem {soma} docentes."))
        assert aval["verdade"]["soma_indevida"]
        assert aval["verdade"]["faltando"]

    def test_o_caso_de_geografia_e_mesmo_ambiguo(self):
        """Se o corpus mudar e deixar de ser ambíguo, este teste avisa."""
        assert POR_ID["est-08"].verdade()["ambiguo"]


class TestExecucaoQueFalhaNaoDerrubaABateria:
    """
    REGRESSÃO — a bateria de 5 set morreu na célula 53 de 150, meia hora depois
    de começar, num `print` de progresso.

    A correção que parou de contar ReadTimeout como rota "nenhuma" passou a
    gravar `rota_escolhida: None`. `calcular_metricas` aprendeu a lidar com o
    None em todos os pontos; a formatação da tela não, e `f"{None:12}"` levanta
    TypeError. Ou seja: o conserto da medição abriu um caminho de morte
    acionado pela mesma condição que ele existia para tratar.

    O teste exercita o LAÇO INTEIRO com todas as execuções falhando — que é o
    que o `print` recebia e ninguém nunca tinha rodado.
    """

    @staticmethod
    def _preparar(monkeypatch, tmp_path, pergunta):
        import interfaces.comparar as comparar

        def explode(componentes, texto):
            raise TimeoutError("timed out")

        monkeypatch.setattr(comparar, "CONJUNTO", [pergunta])
        monkeypatch.setattr(comparar, "validar", lambda: None)
        monkeypatch.setattr(comparar, "PIPELINES", dict.fromkeys(comparar.PIPELINES, explode))
        monkeypatch.setattr(comparar, "montar_componentes", lambda: None)
        monkeypatch.setattr(comparar, "_aguardar_ollama", lambda: 0.0)
        monkeypatch.setattr(comparar, "REPETICOES", 2)
        monkeypatch.setattr(comparar, "SAIDA", tmp_path / "relatorio.md")
        monkeypatch.setattr(comparar, "REGISTRO", tmp_path / "registro.jsonl")
        return comparar

    def test_o_laco_termina_com_todas_as_execucoes_falhando(self, tmp_path, monkeypatch):
        comparar = self._preparar(monkeypatch, tmp_path, POR_ID["sem-01"])

        comparar.executar_comparacao()  # antes: TypeError na primeira repetição

        registro = (tmp_path / "registro.jsonl").read_text(encoding="utf-8")
        assert registro.count("\n") == 4, "2 pipelines de base + 2 repetições"
        assert (tmp_path / "relatorio.md").exists()

    def test_o_relatorio_diz_FALHOU_e_nao_escreve_None(self, tmp_path, monkeypatch):
        """
        `None` no relatório não quebra nada — e é pior por isso: fica
        indistinguível de uma rota que o agente tivesse escolhido.
        """
        comparar = self._preparar(monkeypatch, tmp_path, POR_ID["sem-01"])
        comparar.executar_comparacao()

        relatorio = (tmp_path / "relatorio.md").read_text(encoding="utf-8")
        assert "FALHOU" in relatorio
        assert "`None`" not in relatorio

    def test_falha_de_infraestrutura_fica_fora_das_metricas(self, tmp_path, monkeypatch):
        """
        O ponto da correção original: timeout não é decisão de roteamento.
        Com todas as execuções falhando, não sobra nada para medir — e o
        relatório tem de dizer isso em vez de exibir 0%.
        """
        comparar = self._preparar(monkeypatch, tmp_path, POR_ID["sem-01"])
        comparar.executar_comparacao()

        registros = [
            json.loads(linha)
            for linha in (tmp_path / "registro.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        metricas = comparar.calcular_metricas(
            [{**r, "pipeline": r["pipeline"]} for r in registros]
        )
        assert metricas["execucoes_validas"] == 0
        assert metricas["falhas_de_infraestrutura"] == 2
        assert metricas["condicional_objetivas"] is None
