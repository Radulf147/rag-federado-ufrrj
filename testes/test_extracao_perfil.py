"""
Regressão dos defeitos de EXTRAÇÃO de perfil — achados 01, 03 e 09.

Cada teste aqui é a memória de um defeito que de fato aconteceu e custou caro.
Nenhum deles toca em rede, banco ou container: são funções puras, e é por isso
que valem como teste — rodam em milissegundos e falham por um motivo só.
"""

import parte2_scraping_docentes as p2


class TestAreasDeInteresse:
    """
    ACHADO 01 — o campo não era capturado em nenhum dos 704 perfis.

    A causa não é o SIGAA omitir o espaço antes do parêntese: o `<dt>` desse
    campo é o único do perfil com uma tag aninhada, e `get_text(strip=True)`
    junta os nós SEM separador, produzindo "áreas de interesse(áreas...".
    """

    CHAVE_REAL = "áreas de interesse(áreas de interesse de ensino e pesquisa)"

    def test_casa_a_chave_que_o_bs4_produz(self):
        campos = {self.CHAVE_REAL: "Redes, Inteligência Artificial"}
        assert p2._valor_por_prefixo(campos, "áreas de interesse") == "Redes, Inteligência Artificial"

    def test_casa_tambem_a_variante_com_espaco(self):
        """Se o SIGAA mudar o HTML e o espaço sobreviver, tem de continuar casando."""
        campos = {"áreas de interesse (áreas de ensino)": "Ecologia"}
        assert p2._valor_por_prefixo(campos, "áreas de interesse") == "Ecologia"

    def test_casa_com_acento_decomposto(self):
        """"á" pode chegar como U+00E1 ou como "a" + acento combinante."""
        campos = {"a\u0301reas de interesse(x)": "Botânica"}
        assert p2._valor_por_prefixo(campos, "áreas de interesse") == "Botânica"

    def test_a_igualdade_exata_falharia(self):
        """
        Guarda contra alguém "simplificar" o casamento de volta para ==.
        Esta era a implementação antiga, e é por isso que davam 0 de 704.
        """
        chave_antiga = "áreas de interesse (áreas de interesse de ensino e pesquisa)"
        assert chave_antiga != self.CHAVE_REAL

    def test_prefixo_ausente_devolve_vazio(self):
        assert p2._valor_por_prefixo({"telefone/ramal": "1234"}, "áreas de interesse") == ""


class TestPlaceholderDoSigaa:
    """
    ACHADO 03 — o SIGAA grava "não informada" como VALOR do campo.

    Eram 30% dos campos do corpus. Como o texto é idêntico em centenas de
    perfis, ele funcionava como ímã genérico na busca semântica: os perfis mais
    vazios eram os que mais se pareciam entre si.
    """

    def test_reconhece_as_variantes_do_sigaa(self):
        for valor in ["não informada", "Não Informado", "NÃO INFORMADA", "nao informado"]:
            assert p2._e_placeholder(valor), valor

    def test_reconhece_marcadores_curtos(self):
        for valor in ["", "   ", "-", "--", "n/a"]:
            assert p2._e_placeholder(valor), repr(valor)

    def test_nao_descarta_conteudo_real(self):
        for valor in ["Redes de computadores", "Ecologia de anfíbios", "Doutorado pela UFRJ"]:
            assert not p2._e_placeholder(valor), valor

    def test_nao_descarta_ramal_curto(self):
        """
        O corte antigo era `len(val) > 3`, e jogava fora ramal legítimo —
        medido, 7 de 40 perfis. "677" e "3128" são telefones de verdade.
        """
        for valor in ["677", "3128", "12"]:
            assert not p2._e_placeholder(valor), valor


class TestIdentidadeDoDocente:
    """
    ACHADO 08/09 — o nome prometido pela listagem é a prova de identidade.

    É ela que denuncia uma página servida trocada pela corrida de sessão, e é
    ela que tornou desnecessário exigir chaves de perfil (achado 09).
    """

    def test_nomes_iguais_batem(self):
        assert p2._nomes_batem("BRUNO JOSE DEMBOGURSKI", "BRUNO JOSE DEMBOGURSKI")

    def test_ignora_caixa_acento_e_espaco_duplo(self):
        assert p2._nomes_batem("JOÃO DA SILVA", "joao  da   silva")

    def test_pessoas_diferentes_nao_batem(self):
        assert not p2._nomes_batem("BRUNO JOSE DEMBOGURSKI", "FILIPE BRAIDA DO CARMO")

    def test_sem_nome_esperado_aceita(self):
        """
        Listagem sem <span class="nome"> deixa a verificação cega. Aceitar é
        deliberado: a ausência já é denunciada alto na listagem, e reprovar
        todo mundo por isso seria pior.
        """
        assert p2._nomes_batem("", "QUALQUER PESSOA")


class TestMontagemDoConteudo:
    """O texto que vai para o índice — junção de tudo acima."""

    def test_perfil_esparso_mantem_nome_e_departamento(self):
        """
        ACHADO 09: docente sem seção descritiva era descartado inteiro. Hoje
        entra, porque nome e departamento são exatamente o que as perguntas de
        contagem e listagem precisam.
        """
        from bs4 import BeautifulSoup

        html = """
        <h3>BRUNO JOSE DEMBOGURSKI</h3>
        <h3 class="departamento">DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM</h3>
        <dl><dt>Telefone/Ramal</dt><dd>2124</dd>
            <dt>Endereço eletrônico</dt><dd>bruno@exemplo.br</dd></dl>
        """
        nome, depto, conteudo = p2._montar_conteudo_docente(BeautifulSoup(html, "lxml"))
        assert nome == "BRUNO JOSE DEMBOGURSKI"
        assert "CIÊNCIA DA COMPUTAÇÃO" in depto
        assert "Telefone: 2124" in conteudo
        assert "E-mail: bruno@exemplo.br" in conteudo

    def test_placeholder_nao_entra_no_conteudo(self):
        from bs4 import BeautifulSoup

        html = """
        <h3>FULANO DE TAL</h3>
        <h3 class="departamento">DEPARTAMENTO X</h3>
        <dl><dt>Descrição pessoal</dt><dd>não informada</dd>
            <dt>Telefone/Ramal</dt><dd>1234</dd></dl>
        """
        _, _, conteudo = p2._montar_conteudo_docente(BeautifulSoup(html, "lxml"))
        assert "não informada" not in conteudo
        assert "Perfil:" not in conteudo
        assert "Telefone: 1234" in conteudo

    def test_span_aninhado_no_dt_de_areas(self):
        """O caso exato do achado 01, com o HTML como o SIGAA o entrega."""
        from bs4 import BeautifulSoup

        html = """
        <h3>CICLANA</h3>
        <h3 class="departamento">DEPARTAMENTO Y</h3>
        <dl><dt> Áreas de Interesse <span class="info">\t(áreas de interesse de ensino e pesquisa) </span></dt>
            <dd>Inteligência Artificial, Mineração de Dados</dd></dl>
        """
        _, _, conteudo = p2._montar_conteudo_docente(BeautifulSoup(html, "lxml"))
        assert "Áreas de interesse: Inteligência Artificial, Mineração de Dados" in conteudo
