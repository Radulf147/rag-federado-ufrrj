"""
Testes da loja de posts da rede simulada.

Sem rede, sem LLM, sem container — a loja foi desenhada para ser testável
sozinha, e isto é a prova de que ela é.

Cada teste corresponde a um jeito concreto de este desenho dar resultado
plausível e errado, que é o modo de falha que o projeto trata como inaceitável.
"""

import pytest

import config
from interfaces.rede import loja

COMP = "computacao.ufrrj"
MAT = "matematica.ufrrj"


@pytest.fixture(autouse=True)
def banco_temporario(tmp_path, monkeypatch):
    """
    Cada teste com o seu banco.

    `monkeypatch` no `config`, e não no `loja`: `loja` lê `config.REDE_DB_PATH`
    na hora de conectar, de propósito. Guardar o valor num global do módulo faria
    o patch aqui não ter efeito nenhum e os testes passariam contra o banco de
    verdade — que é a armadilha do módulo importado duas vezes, registrada no
    CLAUDE.md.
    """
    monkeypatch.setattr(config, "REDE_DB_PATH", str(tmp_path / "rede.db"))
    loja.init_db()


class TestMencao:
    def test_mencao_simples_conta(self):
        assert loja.menciona_bot("@ufrrj quantos profs tem a computacao?")

    def test_mencao_no_meio_do_texto_conta(self):
        assert loja.menciona_bot("alguem sabe? @ufrrj me ajuda")

    def test_email_institucional_NAO_conta(self):
        """
        O falso positivo mais provável deste domínio: 92,9% dos perfis de
        docente têm e-mail `@ufrrj.br`, e uma resposta do agente que cite
        contatos mencionaria o próprio bot.
        """
        assert not loja.menciona_bot("escreve pra fulano@ufrrj.br")
        assert not loja.menciona_bot("meu email e raul@ufrrj.br, obrigado")

    def test_handle_maior_NAO_conta(self):
        assert not loja.menciona_bot("segue o @ufrrjnoticias")

    def test_caixa_nao_importa(self):
        assert loja.menciona_bot("@UFRRJ quantos docentes?")

    def test_texto_sem_mencao_devolve_a_pergunta_limpa(self):
        assert loja.texto_sem_mencao("@ufrrj quantos profs?") == "quantos profs?"


class TestPublicar:
    def test_publica_e_le_de_volta(self):
        pid = loja.publicar(COMP, "@raul", "bom dia")
        assert loja.post(pid)["texto"] == "bom dia"

    def test_post_vazio_e_recusado(self):
        with pytest.raises(ValueError):
            loja.publicar(COMP, "@raul", "   ")

    def test_responder_post_inexistente_e_recusado_com_o_id(self):
        """
        Sem esta checagem o SQLite diz só 'FOREIGN KEY constraint failed', que
        não informa QUAL id faltou.
        """
        with pytest.raises(ValueError, match="9999"):
            loja.publicar(COMP, "@raul", "oi", responde_a=9999)

    def test_thread_nao_atravessa_instancia(self):
        """
        A separação de dados por instância é a premissa do projeto. Se uma
        resposta pudesse cair noutra instância, a premissa deixaria de valer no
        dado sem ninguém perceber.
        """
        pid = loja.publicar(COMP, "@raul", "post da computacao")
        with pytest.raises(ValueError, match="instancia"):
            loja.publicar(MAT, "@ana", "respondendo de outra", responde_a=pid)


class TestThread:
    def test_pai_devolve_o_post_de_cima(self):
        raiz_id = loja.publicar(COMP, "@raul", "quem da comp pesquisa IA?")
        filho = loja.publicar(COMP, "@ana", "@ufrrj responde ai", responde_a=raiz_id)
        assert loja.pai(filho)["id"] == raiz_id

    def test_post_de_abertura_nao_tem_pai(self):
        pid = loja.publicar(COMP, "@raul", "oi")
        assert loja.pai(pid) is None

    def test_raiz_sobe_a_thread_inteira(self):
        a = loja.publicar(COMP, "@raul", "a")
        b = loja.publicar(COMP, "@ana", "b", responde_a=a)
        c = loja.publicar(COMP, "@joao", "c", responde_a=b)
        assert loja.raiz(c)["id"] == a

    def test_thread_vem_em_ordem_e_completa(self):
        a = loja.publicar(COMP, "@raul", "a")
        b = loja.publicar(COMP, "@ana", "b", responde_a=a)
        c = loja.publicar(COMP, "@joao", "c", responde_a=a)
        # pedida a partir de um FILHO, tem de devolver a thread toda
        assert [p["id"] for p in loja.thread(b)] == [a, b, c]

    def test_feed_traz_so_aberturas_da_instancia(self):
        a = loja.publicar(COMP, "@raul", "abertura comp")
        loja.publicar(COMP, "@ana", "resposta", responde_a=a)
        loja.publicar(MAT, "@ana", "abertura mat")
        assert [p["id"] for p in loja.feed(COMP)] == [a]


class TestFilaDoBot:
    def test_post_com_mencao_entra_na_fila(self):
        pid = loja.publicar(COMP, "@raul", "@ufrrj quantos docentes?")
        assert [p["id"] for p in loja.pendentes()] == [pid]

    def test_post_sem_mencao_nao_entra(self):
        loja.publicar(COMP, "@raul", "quantos docentes?")
        assert loja.pendentes() == []

    def test_post_ja_respondido_sai_da_fila(self):
        pid = loja.publicar(COMP, "@raul", "@ufrrj quantos docentes?")
        loja.publicar(COMP, loja.AUTOR_BOT, "sao 1302", responde_a=pid, e_bot=True)
        assert loja.pendentes() == []

    def test_resposta_de_humano_NAO_tira_da_fila(self):
        """
        Só a resposta DO BOT encerra o item. Se um humano respondendo bastasse,
        a pergunta ficaria sem resposta do agente e ninguém veria falha nenhuma.
        """
        pid = loja.publicar(COMP, "@raul", "@ufrrj quantos docentes?")
        loja.publicar(COMP, "@ana", "acho que uns mil", responde_a=pid)
        assert [p["id"] for p in loja.pendentes()] == [pid]

    def test_post_do_bot_com_mencao_NAO_entra_na_fila(self):
        """
        A barreira contra auto-resposta em laço, e ela é ESTRUTURAL (`e_bot`),
        não textual: a resposta do agente pode legitimamente conter o handle ao
        citar um e-mail institucional. Se a fila dependesse do texto, o bot
        responderia a si mesmo para sempre.
        """
        loja.publicar(COMP, loja.AUTOR_BOT, "fala com @ufrrj de novo", e_bot=True)
        assert loja.pendentes() == []

    def test_fila_em_ordem_de_chegada(self):
        primeiro = loja.publicar(COMP, "@raul", "@ufrrj um")
        segundo = loja.publicar(MAT, "@ana", "@ufrrj dois")
        assert [p["id"] for p in loja.pendentes()] == [primeiro, segundo]

    def test_fila_atravessa_instancias(self):
        """
        Um bot por rede, não um por instância: a fila junta as duas. É o que
        permite a mesma pergunta ser feita nas duas e as respostas serem
        comparadas.
        """
        loja.publicar(COMP, "@raul", "@ufrrj um")
        loja.publicar(MAT, "@ana", "@ufrrj dois")
        assert {p["instancia"] for p in loja.pendentes()} == {COMP, MAT}
