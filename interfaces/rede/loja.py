"""
Armazenamento dos posts da rede simulada.

Uma tabela só, `posts`, e a thread é reconstruída pela coluna `responde_a`.

POR QUE NÃO EXISTE TABELA DE FILA
---------------------------------
A "fila do bot" não é um estado gravado — é uma CONSULTA: posts que mencionam o
bot e ainda não têm resposta dele. Fila gravada precisaria ser mantida em
sincronia com os posts, e toda vez que dois lugares guardam a mesma verdade um
dos dois fica errado. Aqui a resposta publicada É a marca de que o item saiu da
fila, e não há como as duas coisas discordarem.

O QUE ESTE MÓDULO NÃO FAZ
-------------------------
Não chama o LLM, não sabe o que é um agente e não importa nada do
`modulo2_inferencia`. É proposital: assim a estrutura de posts e threads pode
ser testada inteira sem Ollama, sem túnel e sem container.
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

import config

BOT_HANDLE = config.BOT_HANDLE
AUTOR_BOT = BOT_HANDLE

# CASAMENTO DA MENÇÃO — o `(?<![\w.@])` não é decoração.
#
# A forma ingênua, `\B@ufrrj\b` ou um simples `in`, casa dentro de e-mail:
# "fulano@ufrrj.br" contém "@ufrrj". E e-mail é justamente o que mais aparece
# neste domínio — 92,9% dos perfis de docente têm um, e uma resposta do agente
# que cite contatos passaria a mencionar o próprio bot. Somado ao fato de que
# post do bot nunca entra na fila (ver `pendentes`), seriam duas barreiras, mas
# a primeira sozinha já evitaria um laço de auto-resposta.
#
# Bloqueia também o `.` anterior para não casar "algo.@ufrrj", e o `@` para não
# casar "@@ufrrj".
_RE_MENCAO = re.compile(
    r"(?<![\w.@])" + re.escape(BOT_HANDLE) + r"(?![\w])", re.IGNORECASE
)


def _conectar() -> sqlite3.Connection:
    caminho = Path(config.REDE_DB_PATH)
    caminho.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(caminho)
    conn.row_factory = sqlite3.Row
    # Sem isto o SQLite ACEITA `responde_a` apontando para post inexistente e
    # devolve thread truncada em silêncio — erro plausível, do tipo que este
    # projeto trata como inaceitável.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with _conectar() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS posts (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                instancia  TEXT    NOT NULL,
                autor      TEXT    NOT NULL,
                texto      TEXT    NOT NULL,
                responde_a INTEGER          REFERENCES posts(id),
                e_bot      INTEGER NOT NULL DEFAULT 0,
                criado_em  TEXT    NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_responde_a ON posts(responde_a)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_instancia ON posts(instancia)")


def menciona_bot(texto: str) -> bool:
    """O post chama o bot? Só isto decide se ele entra na fila."""
    return bool(_RE_MENCAO.search(texto or ""))


def texto_sem_mencao(texto: str) -> str:
    """O texto sem o handle — a pergunta como ela ficaria sem o gesto de chamar."""
    return " ".join(_RE_MENCAO.sub(" ", texto or "").split())


def publicar(
    instancia: str,
    autor: str,
    texto: str,
    responde_a: int | None = None,
    e_bot: bool = False,
) -> int:
    """
    Grava um post e devolve o id.

    `responde_a` tem de existir, e a checagem é explícita em vez de deixada
    para o PRAGMA: a mensagem do SQLite ("FOREIGN KEY constraint failed") não
    diz qual id faltou.
    """
    if not (texto or "").strip():
        raise ValueError("post sem texto")
    if not (instancia or "").strip() or not (autor or "").strip():
        raise ValueError("post precisa de instancia e autor")

    with _conectar() as conn:
        if responde_a is not None:
            alvo = conn.execute(
                "SELECT id, instancia FROM posts WHERE id = ?", (responde_a,)
            ).fetchone()
            if alvo is None:
                raise ValueError(f"responde_a={responde_a} nao existe")
            # A resposta vive na instância do post original. Sem isto, uma
            # thread poderia atravessar instâncias e a separação de dados —
            # que é a premissa do projeto — deixaria de valer no dado.
            if alvo["instancia"] != instancia:
                raise ValueError(
                    f"resposta na instancia {instancia!r} para post da "
                    f"instancia {alvo['instancia']!r}"
                )
        cursor = conn.execute(
            """
            INSERT INTO posts (instancia, autor, texto, responde_a, e_bot, criado_em)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                instancia,
                autor,
                texto,
                responde_a,
                int(bool(e_bot)),
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        return int(cursor.lastrowid)


def post(post_id: int) -> dict | None:
    with _conectar() as conn:
        linha = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    return dict(linha) if linha else None


def pai(post_id: int) -> dict | None:
    """
    O post ao qual este responde — o CONTEXTO que o bot vai ler.

    É a peça que separa este projeto de um chatbot comum: em rede social a
    pergunta chega incompleta ("quantos professores tem esse departamento?") e
    o que falta está no post de cima.
    """
    atual = post(post_id)
    if atual is None or atual["responde_a"] is None:
        return None
    return post(atual["responde_a"])


def raiz(post_id: int) -> dict | None:
    """
    O primeiro post da thread.

    Sobe pelo `responde_a`. Não pode entrar em laço: `responde_a` só aponta
    para post já existente, logo sempre para um id MENOR, e a sequência é
    estritamente decrescente. O teto abaixo é rede de segurança contra um banco
    editado à mão, não contra o código.
    """
    atual = post(post_id)
    if atual is None:
        return None
    for _ in range(1000):
        if atual["responde_a"] is None:
            return atual
        acima = post(atual["responde_a"])
        if acima is None:
            return atual
        atual = acima
    raise RuntimeError(f"thread do post {post_id} nao termina — banco inconsistente")


def thread(post_id: int) -> list[dict]:
    """A thread inteira a partir da raiz, em ordem de publicação."""
    inicio = raiz(post_id)
    if inicio is None:
        return []
    with _conectar() as conn:
        linhas = conn.execute(
            """
            WITH RECURSIVE descendentes(id) AS (
                SELECT id FROM posts WHERE id = ?
                UNION
                SELECT p.id FROM posts p JOIN descendentes d ON p.responde_a = d.id
            )
            SELECT * FROM posts WHERE id IN (SELECT id FROM descendentes)
            ORDER BY id
            """,
            (inicio["id"],),
        ).fetchall()
    return [dict(linha) for linha in linhas]


def feed(instancia: str, limite: int = 50) -> list[dict]:
    """Posts de abertura de uma instância, mais recentes primeiro."""
    with _conectar() as conn:
        linhas = conn.execute(
            """
            SELECT * FROM posts
            WHERE instancia = ? AND responde_a IS NULL
            ORDER BY id DESC LIMIT ?
            """,
            (instancia, limite),
        ).fetchall()
    return [dict(linha) for linha in linhas]


def instancias() -> list[str]:
    with _conectar() as conn:
        linhas = conn.execute(
            "SELECT DISTINCT instancia FROM posts ORDER BY instancia"
        ).fetchall()
    return [linha["instancia"] for linha in linhas]


def respostas_do_bot(post_id: int) -> list[dict]:
    with _conectar() as conn:
        linhas = conn.execute(
            "SELECT * FROM posts WHERE responde_a = ? AND e_bot = 1 ORDER BY id",
            (post_id,),
        ).fetchall()
    return [dict(linha) for linha in linhas]


def pendentes(limite: int = 20) -> list[dict]:
    """
    A fila do bot, calculada e não gravada: menciona o bot, não é do bot, e
    ainda não tem resposta dele.

    `e_bot = 0` é a barreira contra auto-resposta em laço, e ela é estrutural:
    a resposta do agente pode legitimamente conter o handle (ao citar o e-mail
    institucional de um docente, por exemplo). Depender só do texto para
    decidir seria depender de o agente nunca escrever a palavra errada.
    """
    with _conectar() as conn:
        linhas = conn.execute(
            """
            SELECT p.* FROM posts p
            WHERE p.e_bot = 0
              AND NOT EXISTS (
                  SELECT 1 FROM posts r WHERE r.responde_a = p.id AND r.e_bot = 1
              )
            ORDER BY p.id
            """
        ).fetchall()
    # A menção é filtrada em Python e não em SQL de propósito: o `LIKE` do
    # SQLite não distingue "@ufrrj" de "fulano@ufrrj.br", que é exatamente o
    # falso positivo que `_RE_MENCAO` existe para impedir. Ver a armadilha da
    # busca cega a acentos no CLAUDE.md — mesma família.
    fila = [dict(linha) for linha in linhas if menciona_bot(linha["texto"])]
    return fila[:limite]


def apagar_tudo() -> None:
    """Só para teste e para recarregar o cenário de exemplo."""
    with _conectar() as conn:
        conn.execute("DELETE FROM posts")
