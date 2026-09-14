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


class TestChecagemDeVinculo:
    """
    A checagem de vínculo cobrava uma FORMA, não um fato.

    O store guarda `DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM`; a versão
    anterior exigia essa string literal dentro da resposta. O agente escreveu
    "Departamento de Ciência da Computação do Instituto de Matemática (IM)" —
    certo, e idêntico nas três repetições — e reprovou nas três. Zero de 3 numa
    pergunta que o sistema acertava.

    O conserto não pode ser um substring pelo núcleo: `COMPUTACAO` está contido
    em `CIENCIA DA COMPUTACAO`, que é OUTRO departamento. Isso trocaria um falso
    negativo por um falso positivo, que é estritamente pior.
    """

    CORPUS = {
        "CIENCIA DA COMPUTACAO": {"DEPARTAMENTO DE CIENCIA DA COMPUTACAO/IM"},
        "COMPUTACAO": {"DEPARTAMENTO DE COMPUTACAO"},
        "GEOGRAFIA": {"DEPARTAMENTO DE GEOGRAFIA", "DEPARTAMENTO DE GEOGRAFIA/IM"},
        "MATEMATICA": {"DEPARTAMENTO DE MATEMATICA"},
    }

    @staticmethod
    def _nomeado(monkeypatch, resposta, esperado):
        import interfaces.comparar as comparar

        monkeypatch.setattr(
            comparar, "_NUCLEOS_DEPARTAMENTO", TestChecagemDeVinculo.CORPUS
        )
        return comparar._departamento_nomeado(comparar._normalizar(resposta), esperado)

    def test_a_resposta_real_do_est_07_passa(self, monkeypatch):
        assert self._nomeado(
            monkeypatch,
            "O professor Marcel William Rocha da Silva trabalha no Departamento "
            "de Ciência da Computação do Instituto de Matemática (IM).",
            "DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
        )

    def test_a_forma_do_banco_tambem_passa(self, monkeypatch):
        """Quem responder copiando a fonte não pode ser punido por isso."""
        assert self._nomeado(
            monkeypatch,
            "Ele é do DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM.",
            "DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
        )

    def test_departamento_errado_reprova(self, monkeypatch):
        assert not self._nomeado(
            monkeypatch,
            "Ele trabalha no Departamento de Matemática.",
            "DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
        )

    def test_o_irmao_de_seropedica_nao_serve_pelo_do_IM(self, monkeypatch):
        """
        O FALSO POSITIVO que um substring ingênuo deixaria passar ao contrário:
        "Ciência da Computação" contém "Computação", mas são departamentos
        diferentes. Quem é do de Seropédica não pode ser dado como do IM.
        """
        assert not self._nomeado(
            monkeypatch,
            "Ele trabalha no Departamento de Computação.",
            "DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
        )

    def test_e_o_do_IM_nao_serve_pelo_de_seropedica(self, monkeypatch):
        assert not self._nomeado(
            monkeypatch,
            "Ele trabalha no Departamento de Ciência da Computação.",
            "DEPARTAMENTO DE COMPUTACAO",
        )

    def test_homonimo_sem_qualificador_reprova(self, monkeypatch):
        """
        ACHADO 06 — "Departamento de Geografia" existe duas vezes. Responder só
        o núcleo não decidiu entre os dois, e não decidir é errar.
        """
        assert not self._nomeado(
            monkeypatch,
            "Ele é do Departamento de Geografia.",
            "DEPARTAMENTO DE GEOGRAFIA/IM",
        )

    def test_homonimo_com_qualificador_passa(self, monkeypatch):
        for forma in ("do Departamento de Geografia/IM",
                      "do Departamento de Geografia do Instituto de Matemática"):
            assert self._nomeado(monkeypatch, forma, "DEPARTAMENTO DE GEOGRAFIA/IM"), forma

    def test_homonimo_sem_sufixo_reprova_se_a_resposta_aponta_o_irmao(self, monkeypatch):
        assert not self._nomeado(
            monkeypatch,
            "Ele é do Departamento de Geografia do IM.",
            "DEPARTAMENTO DE GEOGRAFIA",
        )
        assert self._nomeado(
            monkeypatch, "Ele é do Departamento de Geografia.", "DEPARTAMENTO DE GEOGRAFIA"
        )

    def test_sigla_nao_casa_dentro_de_palavra(self, monkeypatch):
        """"IM" em "IMPORTANTE" não é o Instituto de Matemática."""
        assert not self._nomeado(
            monkeypatch,
            "É importante dizer: ele é do Departamento de Geografia.",
            "DEPARTAMENTO DE GEOGRAFIA/IM",
        )


class TestContextoPersistidoNoRegistro:
    """
    Item 1 do `docs/backlog_avaliacao.md`, corrigido em 9 set 2026.

    `_gravar` trocava o texto do contexto pelo tamanho dele:

        linha["contexto"] = f"<{len(r.contexto)} caracteres>"

    Consequência: `nomes_sem_respaldo` deixava de ser recomputável, e o 100%
    de "interpretativas sem afirmação sem respaldo" da fase 3 virava o que a
    bateria produziu, não o que alguém conferiu (§9 de `relatorio_fase5.md`).

    É também o que impedia o grupo E de `docs/pre_registro_comparacao_30.md`
    de existir sobre os dados de 5 set — o critério é "nenhum nome afirmado
    fora do contexto recuperado", e o contexto recuperado tinha sido jogado
    fora.

    ⚠️ Este teste não confere formatação: confere que o texto SOBREVIVE. Um
    registro que guarda o tamanho passa em qualquer teste de esquema e falha
    na única coisa que o campo existe para permitir.
    """

    def _gravar_em_temporario(self, tmp_path, resultado):
        from types import SimpleNamespace

        import interfaces.comparar as comparar

        alvo = tmp_path / "registro.jsonl"
        original = comparar.REGISTRO
        comparar.REGISTRO = alvo
        try:
            comparar._gravar(
                "exec-teste", SimpleNamespace(id="x-01"), 1, resultado, {"ok": True}
            )
        finally:
            comparar.REGISTRO = original
        return json.loads(alvo.read_text(encoding="utf-8").strip())

    def test_o_texto_do_contexto_chega_inteiro_ao_registro(self, tmp_path):
        texto = "Docente: FULANO DE TAL. Departamento: DEP DE TESTE. " * 40
        linha = self._gravar_em_temporario(
            tmp_path,
            ResultadoPipeline(
                pipeline="3-agente", pergunta="p?", resposta="r", contexto=texto
            ),
        )
        assert linha["contexto"] == texto

    def test_nao_sobrou_o_placeholder_de_tamanho(self, tmp_path):
        # A regressão exata: o campo voltar a ser "<N caracteres>". Escrito
        # separado do teste acima porque um placeholder novo, com outro texto,
        # também precisa falhar.
        linha = self._gravar_em_temporario(
            tmp_path,
            ResultadoPipeline(
                pipeline="1-vetorial", pergunta="p?", resposta="r", contexto="ABC"
            ),
        )
        assert "caracteres>" not in linha["contexto"]

    def test_o_tamanho_continua_disponivel_em_campo_proprio(self, tmp_path):
        texto = "x" * 137
        linha = self._gravar_em_temporario(
            tmp_path,
            ResultadoPipeline(
                pipeline="3-agente", pergunta="p?", resposta="r", contexto=texto
            ),
        )
        assert linha["contexto_caracteres"] == 137

    def test_contexto_vazio_nao_quebra(self, tmp_path):
        # O `2-estruturado` pode devolver contexto vazio quando não casa
        # departamento nenhum. Sem este caso a correção passaria e a bateria
        # morreria na primeira pergunta fora de escopo.
        linha = self._gravar_em_temporario(
            tmp_path,
            ResultadoPipeline(
                pipeline="2-estruturado", pergunta="p?", resposta="r", contexto=""
            ),
        )
        assert linha["contexto"] == ""
        assert linha["contexto_caracteres"] == 0


class TestCaminhosDeSaidaSobreponiveis:
    """
    10 set 2026. `SAIDA` e `REGISTRO` eram constantes fixas, e rodar a bateria
    uma segunda vez SOBRESCREVIA `docs/avaliacao_fase3.md` — o relatório
    publicado da fase 3 — e ANEXAVA ao registro dela, misturando duas medições
    num arquivo só.

    O campo `execucao` distingue as duas, e é por isso que o defeito é
    perigoso em vez de fatal: nada se perde, mas uma apuração feita sem filtrar
    devolve um número plausível e errado.

    ⚠️ O teste confere que o caminho novo é REALMENTE usado por `_gravar`, e
    não só guardado numa variável. Um `main()` que aceitasse o argumento e
    escrevesse no lugar de sempre passaria em qualquer teste de parsing.
    """

    def _rodar_main(self, monkeypatch, argv):
        import interfaces.comparar as comparar

        monkeypatch.setattr(comparar, "SAIDA", comparar.SAIDA_PADRAO)
        monkeypatch.setattr(comparar, "REGISTRO", comparar.REGISTRO_PADRAO)
        visto = {}
        monkeypatch.setattr(
            comparar,
            "executar_comparacao",
            lambda: visto.update(saida=comparar.SAIDA, registro=comparar.REGISTRO),
        )
        monkeypatch.setattr("sys.argv", argv)
        comparar.main()
        return comparar, visto

    def test_sem_argumento_usa_os_caminhos_da_fase3(self, monkeypatch):
        comparar, visto = self._rodar_main(monkeypatch, ["comparar"])
        assert visto["saida"] == comparar.SAIDA_PADRAO
        assert visto["registro"] == comparar.REGISTRO_PADRAO

    def test_os_argumentos_trocam_os_caminhos(self, monkeypatch, tmp_path):
        md, jsonl = tmp_path / "novo.md", tmp_path / "novo.jsonl"
        _, visto = self._rodar_main(
            monkeypatch,
            ["comparar", "--saida", str(md), "--registro", str(jsonl)],
        )
        assert visto["saida"] == md
        assert visto["registro"] == jsonl

    def test_o_gravar_escreve_no_caminho_novo_e_nao_no_padrao(
        self, monkeypatch, tmp_path
    ):
        from types import SimpleNamespace

        md, jsonl = tmp_path / "novo.md", tmp_path / "novo.jsonl"
        comparar, _ = self._rodar_main(
            monkeypatch,
            ["comparar", "--saida", str(md), "--registro", str(jsonl)],
        )
        comparar._gravar(
            "exec-nova",
            SimpleNamespace(id="y-01"),
            1,
            ResultadoPipeline(pipeline="3-agente", pergunta="p", resposta="r"),
            {},
        )
        assert jsonl.exists()
        assert json.loads(jsonl.read_text(encoding="utf-8").strip())["execucao"] == (
            "exec-nova"
        )
        # E o padrão continua onde estava — este é o ponto do teste.
        assert comparar.REGISTRO_PADRAO != jsonl


class TestRessalvaDoValorAutomatico:
    """
    11 set 2026. A tabela de métricas do relatório gerado não dizia que os
    valores são os AUTOMÁTICOS, e a falta produziu um erro concreto: o 91,7%
    de `docs/avaliacao_fase3.md` quase foi citado num e-mail ao orientador
    como "o resultado da fase 3". O resultado é o intervalo auditado
    [95,83% ; 100%], de `docs/relatorio_fase5.md`.

    A ressalva mora no RENDERIZADOR, e não no .md, porque o .md é gerado —
    escrita à mão no arquivo, uma execução sem `--saida` a levaria junto.

    ⚠️ Este teste existe para que a ressalva sobreviva a refatoração. Um
    relatório sem ela volta a parecer um veredito.
    """

    def _render(self):
        import interfaces.comparar as comparar

        return comparar._renderizar(
            [], comparar.calcular_metricas([]), "exec-teste", [], False
        )

    def test_o_relatorio_diz_que_os_valores_sao_automaticos(self):
        assert "AUTOMÁTICOS" in self._render()

    def test_o_relatorio_aponta_para_o_documento_do_metodo(self):
        # Sem o ponteiro a ressalva vira só um aviso vago: quem lê precisa
        # saber ONDE está a política de denominador.
        assert "docs/criterios_avaliacao.md" in self._render()

    def test_o_relatorio_avisa_que_X_nao_e_criterio_reprovado(self):
        texto = self._render()
        assert "não significa critério reprovado" in texto

    def test_a_ressalva_vem_depois_da_tabela_das_metricas(self):
        # Antes da tabela ela seria lida como preâmbulo e pulada; o lugar que
        # funciona é logo abaixo dos números.
        texto = self._render()
        assert texto.index("Acurácia de roteamento") < texto.index("AUTOMÁTICOS")


class TestConsultaEmitidaFicaRegistrada:
    """
    A consulta que vai para a busca, e não a pergunta que o usuário fez.

    POR QUE (12 set 2026). Relendo a bateria de `main`, nas 7 perguntas
    semânticas o agente recuperou de 3 a 8 dos seus 10 documentos que o
    `1-vetorial` NÃO viu — mesmo ChromaDB, mesmo bge-m3, mesmo top_k=10, mesma
    execução, uma única chamada de ferramenta. Só se conclui disso que a
    consulta emitida foi outra. Qual, era impossível saber: o registro guardava
    o NOME da ferramenta e jogava fora o ARGUMENTO.

    ⚠️ Estes testes não verificam que o agente busca bem. Verificam que a
    escolha dele fica legível — que é a condição para decidir a correção com
    dado em vez de hipótese.
    """

    @staticmethod
    def _resultado_de_tool(nome, argumentos, texto="ok"):
        from types import SimpleNamespace

        return SimpleNamespace(
            origin=SimpleNamespace(tool_name=nome, arguments=argumentos),
            result=texto,
        )

    def _agente_com_historico(self, monkeypatch, historico, pergunta="pergunta do usuário?"):
        from types import SimpleNamespace

        import modulo2_inferencia.pipelines as pipelines

        monkeypatch.setattr(
            pipelines, "processar_pergunta", lambda **kwargs: ("resposta", historico)
        )
        # Os três atributos são lidos na montagem da chamada, ANTES do
        # `processar_pergunta` dublado rodar — um SimpleNamespace vazio quebra
        # em AttributeError sem nunca chegar ao que se quer medir.
        componentes = SimpleNamespace(chat_generator=None, embedder=None, retriever=None)
        return pipelines.responder_agente(componentes, pergunta)

    def test_o_argumento_do_agente_fica_gravado(self, monkeypatch):
        from types import SimpleNamespace

        historico = [
            SimpleNamespace(
                tool_call_results=[
                    self._resultado_de_tool(
                        "busca_vetorial_sigaa", {"pergunta": "formação docente currículo"}
                    )
                ]
            )
        ]
        r = self._agente_com_historico(monkeypatch, historico)
        # Campo a campo, e não o dicionário inteiro: a entrada de uma busca
        # semântica ganhou a chave `busca` em 14 set, que diz o que foi de fato
        # EMBUTIDO (ver TestVarianteDaConsultaSemantica). Igualdade exata aqui
        # transformaria qualquer campo novo em falha, sem que nada tivesse
        # quebrado — e o que este teste guarda é o argumento, não o formato.
        assert len(r.consultas) == 1
        assert r.consultas[0]["via"] == "busca_vetorial_sigaa"
        assert r.consultas[0]["argumentos"] == {"pergunta": "formação docente currículo"}

    def test_o_que_fica_gravado_NAO_e_a_pergunta_do_usuario(self, monkeypatch):
        """
        A regressão que tornaria o campo inútil.

        Um campo `consultas` preenchido com `pergunta` passaria em qualquer
        teste de esquema, pareceria certo no JSONL, e apagaria exatamente o
        fenômeno que ele existe para expor — a paráfrase. O caso abaixo é
        construído para que os dois textos sejam diferentes de propósito.
        """
        from types import SimpleNamespace

        historico = [
            SimpleNamespace(
                tool_call_results=[
                    self._resultado_de_tool("busca_vetorial_sigaa", {"pergunta": "didática"})
                ]
            )
        ]
        r = self._agente_com_historico(
            monkeypatch, historico, pergunta="Algum professor atua com didática?"
        )
        assert r.consultas[0]["argumentos"]["pergunta"] == "didática"
        assert r.consultas[0]["argumentos"]["pergunta"] != r.pergunta

    def test_varias_rodadas_viram_varias_consultas_na_ordem(self, monkeypatch):
        from types import SimpleNamespace

        historico = [
            SimpleNamespace(
                tool_call_results=[
                    self._resultado_de_tool(
                        "buscar_docentes_por_departamento", {"departamento": "MATEMÁTICA"}
                    )
                ]
            ),
            SimpleNamespace(
                tool_call_results=[
                    self._resultado_de_tool("busca_vetorial_sigaa", {"pergunta": "estatística"})
                ]
            ),
        ]
        r = self._agente_com_historico(monkeypatch, historico)
        assert [c["via"] for c in r.consultas] == [
            "buscar_docentes_por_departamento",
            "busca_vetorial_sigaa",
        ]

    def test_sem_tool_nenhuma_a_lista_fica_vazia(self, monkeypatch):
        from types import SimpleNamespace

        historico = [SimpleNamespace(tool_call_results=None)]
        r = self._agente_com_historico(monkeypatch, historico)
        assert r.consultas == []

    def test_argumento_ausente_nao_quebra(self, monkeypatch):
        # Um modelo pode emitir tool call sem argumentos. Isso é dado — e não
        # pode derrubar a bateria inteira na pergunta seguinte.
        from types import SimpleNamespace

        historico = [
            SimpleNamespace(
                tool_call_results=[self._resultado_de_tool("busca_vetorial_sigaa", None)]
            )
        ]
        r = self._agente_com_historico(monkeypatch, historico)
        assert len(r.consultas) == 1
        assert r.consultas[0]["via"] == "busca_vetorial_sigaa"
        assert r.consultas[0]["argumentos"] == {}

    def test_o_vetorial_grava_a_pergunta_literal(self):
        """
        No `1-vetorial` a consulta É a pergunta — é ela que vai ao embedder.

        Registrar isso parece redundante e é o termo de comparação: sem a
        linha do RAG puro no mesmo campo, "o agente perguntou outra coisa" não
        tem contra o quê ser lido.
        """
        from types import SimpleNamespace

        import modulo2_inferencia.pipelines as pipelines

        doc = SimpleNamespace(content="Docente: FULANO.", meta={"nome_docente": "FULANO"})
        componentes = SimpleNamespace(
            embedder=SimpleNamespace(run=lambda text: {"embedding": [0.0]}),
            retriever=SimpleNamespace(run=lambda query_embedding: {"documents": [doc]}),
            chat_generator=SimpleNamespace(
                run=lambda messages: {"replies": [SimpleNamespace(text="resposta")]}
            ),
        )
        r = pipelines.responder_vetorial(componentes, "Quem pesquisa ecologia?")
        assert r.consultas == [
            {"via": "retriever direto", "argumentos": {"pergunta": "Quem pesquisa ecologia?"}}
        ]

    def test_o_campo_chega_ao_registro_jsonl(self, tmp_path):
        # O campo só serve se sobreviver até o arquivo. `asdict` leva qualquer
        # campo novo do dataclass, mas isso é consequência de um detalhe de
        # implementação de `_gravar` — e detalhe de implementação muda.
        from types import SimpleNamespace

        import interfaces.comparar as comparar

        alvo = tmp_path / "registro.jsonl"
        original = comparar.REGISTRO
        comparar.REGISTRO = alvo
        try:
            comparar._gravar(
                "exec-teste",
                SimpleNamespace(id="sem-09"),
                1,
                ResultadoPipeline(
                    pipeline="3-agente",
                    pergunta="Algum professor atua com didática?",
                    resposta="r",
                    consultas=[
                        {"via": "busca_vetorial_sigaa", "argumentos": {"pergunta": "didática"}}
                    ],
                ),
                {"ok": True},
            )
        finally:
            comparar.REGISTRO = original

        linha = json.loads(alvo.read_text(encoding="utf-8").strip())
        assert linha["consultas"][0]["argumentos"]["pergunta"] == "didática"


class TestVarianteDaConsultaSemantica:
    """
    Experimento de `docs/pre_registro_consulta_semantica.md`.

    O agente reduz a pergunta ao termo nu — "didática" no lugar de "Algum
    professor atua com didática?" — em 21 de 21 execuções, porque o schema
    manda "otimizar". `VARIANTE_CONSULTA` decide qual texto é embutido.

    ⚠️ O PADRÃO TEM DE SER v0. Enquanto o experimento não decidir, nada muda em
    produção sem alguém escrever a variável. O primeiro teste desta classe é
    esse, e não é formalidade: um padrão trocado por engano mudaria o
    comportamento do agente em silêncio e invalidaria a comparação com as
    baterias de 5, 10 e 14 set.
    """

    @staticmethod
    def _componentes(docs_por_texto):
        """Dublê que devolve documentos diferentes conforme o texto embutido."""
        from types import SimpleNamespace

        embutidos = []

        def embed(text):
            embutidos.append(text)
            return {"embedding": text}

        def recuperar(query_embedding):
            return {"documents": docs_por_texto.get(query_embedding, [])}

        return (
            SimpleNamespace(run=embed),
            SimpleNamespace(run=recuperar),
            embutidos,
        )

    @staticmethod
    def _doc(id_, conteudo, score, nome="FULANO", depto="DEP"):
        from types import SimpleNamespace

        return SimpleNamespace(
            id=id_,
            content=conteudo,
            score=score,
            meta={"nome_docente": nome, "departamento": depto},
        )

    def test_o_padrao_e_v0(self):
        import modulo2_inferencia.tools as tools

        assert tools.VARIANTE_CONSULTA == "v0"

    def test_variante_invalida_estoura_em_vez_de_cair_no_v0(self, monkeypatch):
        # Um nome errado cairia silenciosamente no controle, e a bateria
        # reportaria "variante medida" tendo medido o de sempre.
        import importlib

        import modulo2_inferencia.tools as tools

        monkeypatch.setenv("VARIANTE_CONSULTA", "v9")
        try:
            import pytest

            with pytest.raises(ValueError, match="VARIANTE_CONSULTA"):
                importlib.reload(tools)
        finally:
            monkeypatch.delenv("VARIANTE_CONSULTA", raising=False)
            importlib.reload(tools)
        assert tools.VARIANTE_CONSULTA == "v0"

    def test_v0_embute_o_argumento_do_llm(self, monkeypatch):
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v0")
        assert tools._textos_a_embutir("didática", "Algum professor atua com didática?") == [
            "didática"
        ]

    def test_v1_embute_a_pergunta_do_usuario(self, monkeypatch):
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v1")
        assert tools._textos_a_embutir("didática", "Algum professor atua com didática?") == [
            "Algum professor atua com didática?"
        ]

    def test_v1_sem_pergunta_original_cai_no_argumento(self, monkeypatch):
        # Chamador antigo que não passa a original. Cair de volta é melhor que
        # estourar — mas então v1 É v0, e o registro tem de deixar isso visível.
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v1")
        assert tools._textos_a_embutir("didática", None) == ["didática"]

    def test_v2_embute_as_duas_com_a_original_primeiro(self, monkeypatch):
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v2")
        assert tools._textos_a_embutir("didática", "Algum professor atua com didática?") == [
            "Algum professor atua com didática?",
            "didática",
        ]

    def test_v2_nao_embute_duas_vezes_o_mesmo_texto(self, monkeypatch):
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v2")
        assert tools._textos_a_embutir("igual", "igual") == ["igual"]

    def test_v3_embute_o_argumento_do_llm_como_o_v0(self, monkeypatch):
        # A v3 só muda a DESCRIÇÃO do parâmetro; o caminho do dado é o do v0.
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v3")
        assert tools._textos_a_embutir("didática", "Algum professor atua com didática?") == [
            "didática"
        ]

    def test_P1_v1_recupera_o_mesmo_que_o_pipeline_vetorial(self, monkeypatch):
        """
        **P1 do pré-registro**, verificada em código.

        Se a v1 embute a pergunta do usuário e o `1-vetorial` também, os dois
        TÊM de receber os mesmos documentos. Divergência aqui não é achado: é
        bug, e foi para pegá-lo que a previsão foi escrita.
        """
        import modulo2_inferencia.pipelines as pipelines
        import modulo2_inferencia.tools as tools
        from types import SimpleNamespace

        pergunta = "Algum professor atua com didática?"
        ricos = [self._doc("d1", "Docente: ANA. Áreas: didática.", 0.5, "ANA")]
        pobres = [self._doc("d2", "Docente: BIA. Telefone: 1.", 0.4, "BIA")]
        mapa = {pergunta: ricos, "didática": pobres}

        # 1-vetorial
        emb, ret, _ = self._componentes(mapa)
        comp = SimpleNamespace(
            embedder=emb,
            retriever=ret,
            chat_generator=SimpleNamespace(
                run=lambda messages: {"replies": [SimpleNamespace(text="r")]}
            ),
        )
        do_vetorial = pipelines.responder_vetorial(comp, pergunta).contexto

        # v1: o LLM manda "didática", mas a v1 embute a pergunta original
        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v1")
        emb2, ret2, embutidos = self._componentes(mapa)
        tools.busca_vetorial_sigaa("didática", emb2, ret2, pergunta)

        assert embutidos == [pergunta], "a v1 não embutiu a pergunta do usuário"
        assert "ANA" in do_vetorial and "BIA" not in do_vetorial

    def test_v2_une_sem_repetir_e_ordena_por_distancia(self, monkeypatch):
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v2")
        monkeypatch.setattr(tools, "LIMIAR_DISTANCIA", None)
        comum = self._doc("d1", "Docente: ANA.", 0.9, "ANA")
        so_original = self._doc("d2", "Docente: BIA.", 0.1, "BIA")
        so_termo = self._doc("d3", "Docente: CIA.", 0.5, "CIA")
        emb, ret, _ = self._componentes(
            {"pergunta inteira?": [comum, so_original], "termo": [comum, so_termo]}
        )
        saida = tools.busca_vetorial_sigaa("termo", emb, ret, "pergunta inteira?")

        # os três aparecem, ANA uma vez só, e na ordem BIA(0.1) CIA(0.5) ANA(0.9)
        assert saida.count("Docente: ANA") == 1
        assert saida.index("BIA") < saida.index("CIA") < saida.index("ANA")

    def test_o_registro_diz_o_que_foi_embutido_e_nao_o_que_o_llm_pediu(self, monkeypatch):
        """
        A regressão que faria a instrumentação mentir.

        Na v1 o argumento do ToolCall é "didática" e o texto embutido é a
        pergunta inteira. Um registro que guardasse só o argumento descreveria
        a intenção do modelo, não o ato do sistema.
        """
        import modulo2_inferencia.tools as tools

        monkeypatch.setattr(tools, "VARIANTE_CONSULTA", "v1")
        pergunta = "Algum professor atua com didática?"
        emb, ret, _ = self._componentes({pergunta: [self._doc("d1", "Docente: ANA.", 0.2)]})
        registro = []
        tools.busca_vetorial_sigaa("didática", emb, ret, pergunta, registro)

        assert registro[0]["embutido"] == [pergunta]
        assert registro[0]["argumento_do_llm"] == "didática"
        assert registro[0]["variante"] == "v1"

    def test_v3_muda_a_descricao_do_parametro(self, monkeypatch):
        import importlib

        import modulo2_inferencia.tools as tools

        monkeypatch.setenv("VARIANTE_CONSULTA", "v3")
        try:
            importlib.reload(tools)
            desc = next(
                s["function"]["parameters"]["properties"]["pergunta_semantica"]["description"]
                for s in tools.TOOLS_SCHEMA
                if s["function"]["name"] == "busca_vetorial_sigaa"
            )
            assert "ÍNTEGRA" in desc
            assert "otimizada" not in desc
        finally:
            monkeypatch.delenv("VARIANTE_CONSULTA", raising=False)
            importlib.reload(tools)

    def test_o_carimbo_registra_a_variante(self):
        # Sem isto dois registros de variantes diferentes são indistinguíveis
        # no disco, e alguém os compara como se fossem a mesma coisa.
        import interfaces.comparar as comparar

        assert comparar._carimbo()["variante_consulta"] == "v0"
