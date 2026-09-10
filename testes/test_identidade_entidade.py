"""
Identidade de entidade — o nome NÃO é chave (decisão D1).

Escrito ANTES da correção, de propósito, e **deve reprovar a versão atual**.
Um teste que passa nas duas versões não distingue o que mediu; é ritual, e a
sétima ocorrência do §10 já custou uma bateria inteira por isso.

O QUE ESTÁ SENDO AFIRMADO
-------------------------
Três funções puras chaveiam por `nome_docente`:

    medir_recuperacao.gabarito()  -> devolve um SET de nomes
    medir_hibrido.recall()        -> monta um SET de nomes
    medir_hibrido.fundir()        -> dicionário de RRF chaveado por nome

Dois documentos de pessoas diferentes com o MESMO nome colapsam em um. Isso não
é hipótese: `FERNANDA SILVA FERREIRA CHAER` existe duas vezes no corpus (dois
SIAPEs, dois departamentos — a duplicação é da fonte, item 10 do backlog).

Hoje o efeito é 1 em 1302 e não muda número nenhum das 6 medições, porque ela
não escreveu nenhum dos temas medidos. **O tamanho do efeito não é o
argumento.** O argumento é que os tipos novos pioram o mecanismo: na listagem de
cursos, `CIÊNCIAS BIOLÓGICAS` aparece duas vezes, mesmo campus, uma Bacharelado
e outra Licenciatura, com ids distintos.

POR QUE ESTES DOIS DOCUMENTOS E NÃO A FERNANDA REAL
---------------------------------------------------
A Fernanda não escreveu nenhum dos temas, então usá-la aqui mediria zero contra
zero. As fixtures abaixo são dois homônimos que **escreveram o tema**, que é o
caso em que o colapso muda o resultado.
"""

import pytest
from haystack import Document

from modulo2_inferencia import medir_hibrido, medir_recuperacao

TEMA = "inteligencia artificial"


def _docente(nome: str, siape: str, tema: str = TEMA) -> Document:
    """Perfil mínimo que `texto_descritivo` consegue recortar."""
    return Document(
        content=(
            f"Docente: {nome}. Departamento: DEPARTAMENTO DE TESTE. "
            f"Áreas de interesse: {tema}"
        ),
        meta={
            "nome_docente": nome,
            "siape": siape,
            "departamento": "DEPARTAMENTO DE TESTE",
            "id_entidade": f"docente:{siape}",
            "rotulo": nome,
        },
    )


@pytest.fixture
def homonimos() -> list[Document]:
    """Duas PESSOAS distintas, mesmo nome, ambas com o tema escrito."""
    return [_docente("ANA SILVA", "111"), _docente("ANA SILVA", "222")]


class TestGabaritoNaoColapsaHomonimos:
    """
    O gabarito responde "quem deveria ser encontrado". Duas pessoas que
    escreveram a frase são DUAS respostas certas, não uma.
    """

    def test_duas_pessoas_dao_gabarito_de_tamanho_dois(self, homonimos):
        esperados = medir_recuperacao.gabarito(homonimos, TEMA)
        assert len(esperados) == 2, (
            f"gabarito colapsou os homonimos em {len(esperados)}: {esperados}. "
            "Chaveado por nome, a segunda pessoa desaparece do denominador e o "
            "recall fica artificialmente melhor."
        )

    def test_pessoa_sem_o_tema_continua_fora(self):
        # Controle: o teste acima falharia tambem se o gabarito simplesmente
        # devolvesse tudo. Este fixa o outro lado.
        docs = [_docente("ANA SILVA", "111"), _docente("ANA SILVA", "222", "botanica")]
        assert len(medir_recuperacao.gabarito(docs, TEMA)) == 1


class TestRecallNaoColapsaHomonimos:
    def test_as_duas_contam(self, homonimos):
        esperados = {"docente:111", "docente:222"}
        assert medir_hibrido.recall(homonimos, esperados, k=10) == 2, (
            "recall() monta um set de nomes; os dois homonimos viram um, e o "
            "acerto da segunda pessoa e perdido."
        )


class TestFundirNaoColapsaHomonimos:
    """
    O RRF funde rankings somando 1/(K+posicao) por documento. Chaveado por
    nome, o segundo homonimo SOMA na pontuacao do primeiro e some da saida —
    dois erros de uma vez: um documento perdido e outro com nota inflada.
    """

    def test_os_dois_sobrevivem_a_fusao(self, homonimos):
        a, b = homonimos
        fundido = medir_hibrido.fundir([a], [b])
        assert len(fundido) == 2, (
            f"fundir() devolveu {len(fundido)} documento(s) para duas pessoas "
            "distintas. A chave do RRF e o nome."
        )

    def test_documentos_distintos_saem_distintos(self, homonimos):
        a, b = homonimos
        fundido = medir_hibrido.fundir([a], [b])
        ids = {d.meta.get("id_entidade") for d in fundido}
        assert ids == {"docente:111", "docente:222"}, (
            f"esperava as duas identidades, veio {ids}"
        )


class TestIdentidadeSobreviveAReindexacao:
    """
    Por que `id_entidade` e nao o `Document.id` do Haystack.

    O `Document.id` e hash do conteudo: reindexar o muda. Medido em 7 set 2026
    -- o mesmo FILIPE BRAIDA DO CARMO tem tres ids diferentes nas tres colecoes
    que existem hoje. Identidade que nao sobrevive a uma decisao de indexacao
    nao e identidade.
    """

    def test_id_entidade_e_estavel_entre_conteudos_diferentes(self):
        completo = _docente("ANA SILVA", "111")
        # o mesmo docente apos a reindexacao descritiva: outro conteudo
        so_descritivo = Document(
            content=f"Areas de interesse: {TEMA}",
            meta=dict(completo.meta),
        )
        assert completo.id != so_descritivo.id, (
            "premissa do teste: o Document.id do Haystack depende do conteudo"
        )
        assert completo.meta["id_entidade"] == so_descritivo.meta["id_entidade"]
