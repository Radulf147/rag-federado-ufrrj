"""
Testes do servidor da rede simulada, pelo cliente de teste do Flask.

Sem navegador e sem porta aberta. Exercitam o que de fato pode quebrar em
silêncio: rota que devolve 200 com a página errada, post gravado na instância
errada, e o aviso de "aguardando o bot" aparecendo onde não devia.
"""

import pytest

import config
from interfaces.rede import loja, servidor

COMP = "computacao.ufrrj"
MAT = "matematica.ufrrj"


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REDE_DB_PATH", str(tmp_path / "rede.db"))
    loja.init_db()


@pytest.fixture
def cliente():
    app = servidor.criar_app()
    app.config.update(TESTING=True)
    return app.test_client()


class TestRotas:
    def test_raiz_redireciona_para_a_instancia_padrao(self, cliente):
        resposta = cliente.get("/")
        assert resposta.status_code == 302
        assert COMP in resposta.headers["Location"]

    def test_feed_abre(self, cliente):
        assert cliente.get(f"/i/{COMP}").status_code == 200

    def test_instancia_inventada_da_404(self, cliente):
        """
        Sem isto a rota renderizaria feed vazio para qualquer nome — página
        200, bonita, de uma instância que não existe.
        """
        assert cliente.get("/i/naoexiste.ufrrj").status_code == 404

    def test_as_duas_instancias_aparecem_na_coluna(self, cliente):
        html = cliente.get(f"/i/{COMP}").get_data(as_text=True)
        assert COMP in html and MAT in html

    def test_thread_de_outra_instancia_da_404(self, cliente):
        """
        A separação por instância é a premissa do projeto; ela tem de valer
        também na URL, não só no banco.
        """
        pid = loja.publicar(COMP, "@raul", "post da computacao")
        assert cliente.get(f"/i/{MAT}/p/{pid}").status_code == 404
        assert cliente.get(f"/i/{COMP}/p/{pid}").status_code == 200

    def test_thread_inexistente_da_404(self, cliente):
        assert cliente.get(f"/i/{COMP}/p/9999").status_code == 404


class TestPublicar:
    def test_publica_e_aparece_no_feed(self, cliente):
        cliente.post("/publicar", data={
            "instancia": COMP, "autor": "@raul", "texto": "primeiro post"
        })
        assert "primeiro post" in cliente.get(f"/i/{COMP}").get_data(as_text=True)

    def test_publica_na_instancia_pedida_e_nao_na_outra(self, cliente):
        cliente.post("/publicar", data={
            "instancia": MAT, "autor": "@ana", "texto": "so na matematica"
        })
        assert "so na matematica" not in cliente.get(f"/i/{COMP}").get_data(as_text=True)
        assert "so na matematica" in cliente.get(f"/i/{MAT}").get_data(as_text=True)

    def test_arroba_e_acrescentada_ao_autor(self, cliente):
        cliente.post("/publicar", data={
            "instancia": COMP, "autor": "raul", "texto": "sem arroba"
        })
        assert loja.feed(COMP)[0]["autor"] == "@raul"

    def test_autor_vazio_vira_anonimo(self, cliente):
        cliente.post("/publicar", data={
            "instancia": COMP, "autor": "", "texto": "sem nome"
        })
        assert loja.feed(COMP)[0]["autor"] == "@anonimo"

    def test_texto_vazio_nao_cria_post(self, cliente):
        cliente.post("/publicar", data={
            "instancia": COMP, "autor": "@raul", "texto": "   "
        })
        assert loja.feed(COMP) == []

    def test_instancia_invalida_e_recusada(self, cliente):
        resposta = cliente.post("/publicar", data={
            "instancia": "hacker.local", "autor": "@x", "texto": "oi"
        })
        assert resposta.status_code == 400

    def test_resposta_cruzando_instancia_e_recusada(self, cliente):
        pid = loja.publicar(COMP, "@raul", "post da comp")
        resposta = cliente.post("/publicar", data={
            "instancia": MAT, "autor": "@ana", "texto": "cruzando",
            "responde_a": str(pid),
        })
        assert resposta.status_code == 400


class TestAvisoDeEspera:
    def test_post_que_menciona_o_bot_mostra_o_aviso(self, cliente):
        raiz = loja.publicar(COMP, "@raul", "quantos profs?")
        loja.publicar(COMP, "@ana", "@ufrrj responde ai", responde_a=raiz)
        assert "ainda não respondeu" in cliente.get(f"/i/{COMP}").get_data(as_text=True)

    def test_depois_da_resposta_o_aviso_some(self, cliente):
        raiz = loja.publicar(COMP, "@raul", "quantos profs?")
        mencao = loja.publicar(COMP, "@ana", "@ufrrj responde ai", responde_a=raiz)
        loja.publicar(COMP, loja.AUTOR_BOT, "sao 15", responde_a=mencao, e_bot=True)
        html = cliente.get(f"/i/{COMP}").get_data(as_text=True)
        assert "ainda não respondeu" not in html
        assert "sao 15" in html

    def test_thread_sem_mencao_nao_mostra_aviso(self, cliente):
        """
        O aviso é estado real da fila. Se aparecesse onde o bot não foi
        chamado, estaria prometendo uma resposta que nunca vem.
        """
        raiz = loja.publicar(MAT, "@pedro", "o cafe voltou a funcionar")
        loja.publicar(MAT, "@bia", "boa noticia", responde_a=raiz)
        assert "ainda não respondeu" not in cliente.get(f"/i/{MAT}").get_data(as_text=True)


class TestSeguranca:
    def test_texto_do_post_e_escapado(self, cliente):
        """
        O Jinja escapa por padrão, mas isto é a rede social do projeto: se
        alguém desligar o autoescape numa refatoração, este teste avisa.
        """
        cliente.post("/publicar", data={
            "instancia": COMP, "autor": "@mal",
            "texto": "<script>alert(1)</script>",
        })
        html = cliente.get(f"/i/{COMP}").get_data(as_text=True)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
