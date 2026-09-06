"""
Servidor da rede simulada — a página que se abre no navegador.

    docker compose --profile rede up rede
    http://localhost:5000

RENDERIZADO NO SERVIDOR, DE PROPÓSITO
-------------------------------------
Sem framework de front-end e sem chamada de API pelo JavaScript. A página é
HTML montado aqui e devolvido pronto. Numa peça cujo papel é ser o cenário de um
experimento, cada camada a mais é uma camada a mais que pode estar errada
enquanto tudo parece funcionar.

POR QUE NÃO EXISTE ATUALIZAÇÃO AUTOMÁTICA
-----------------------------------------
É deliberado, e é fidelidade ao que se está simulando: no Grok, quem pergunta
não fica olhando — o comentário aparece na hora, a resposta chega depois e a
pessoa atualiza. Um spinner esperando o LLM daria a impressão errada sobre como
o sistema funciona, e é justamente a impressão que o TCC vai herdar.

O que a página mostra no lugar do spinner é ESTADO REAL, lido do banco: um post
que menciona o bot e ainda não tem resposta aparece marcado como aguardando.
Isso é fato sobre a fila, não animação.

ESTE MÓDULO NÃO CHAMA O AGENTE
------------------------------
Quem responde é `bot.py`, num processo separado. O servidor só grava o post e
devolve a página. Se o Ollama estiver fora, publicar continua funcionando e a
resposta chega quando o worker voltar — que é o comportamento certo para uma
rede social e o que evita o site inteiro depender do túnel SSH.
"""

from flask import Flask, abort, redirect, render_template, request, url_for

import config
from interfaces.rede import instancias as inst
from interfaces.rede import loja


def criar_app() -> Flask:
    app = Flask(__name__)
    loja.init_db()

    @app.context_processor
    def _comuns():
        return {
            "INSTANCIAS": inst.INSTANCIAS,
            "BOT": loja.BOT_HANDLE,
        }

    def _aguardando() -> set[int]:
        """Ids de posts que chamaram o bot e ainda não têm resposta dele."""
        return {p["id"] for p in loja.pendentes(limite=1000)}

    def _threads(instancia_id: str) -> list[dict]:
        espera = _aguardando()
        saida = []
        for raiz in loja.feed(instancia_id):
            posts = loja.thread(raiz["id"])
            saida.append(
                {
                    "raiz": posts[0],
                    "respostas": posts[1:],
                    "aguardando": {p["id"] for p in posts} & espera,
                }
            )
        return saida

    @app.route("/")
    def inicio():
        return redirect(url_for("feed", instancia_id=inst.PADRAO.id))

    @app.route("/i/<instancia_id>")
    def feed(instancia_id: str):
        if not inst.existe(instancia_id):
            abort(404)
        return render_template(
            "feed.html",
            atual=inst.POR_ID[instancia_id],
            threads=_threads(instancia_id),
        )

    @app.route("/i/<instancia_id>/p/<int:post_id>")
    def thread(instancia_id: str, post_id: int):
        if not inst.existe(instancia_id):
            abort(404)
        posts = loja.thread(post_id)
        if not posts or posts[0]["instancia"] != instancia_id:
            abort(404)
        espera = _aguardando()
        return render_template(
            "thread.html",
            atual=inst.POR_ID[instancia_id],
            raiz=posts[0],
            respostas=posts[1:],
            aguardando={p["id"] for p in posts} & espera,
        )

    @app.route("/publicar", methods=["POST"])
    def publicar():
        instancia_id = (request.form.get("instancia") or "").strip()
        autor = (request.form.get("autor") or "").strip()
        texto = (request.form.get("texto") or "").strip()
        responde_a = request.form.get("responde_a") or None

        if not inst.existe(instancia_id):
            abort(400, "instancia desconhecida")
        if not texto:
            # Campo vazio é engano de quem digita, não erro do sistema: volta
            # para a página em vez de mostrar tela de erro.
            return redirect(url_for("feed", instancia_id=instancia_id))
        if not autor:
            autor = "@anonimo"
        if not autor.startswith("@"):
            autor = "@" + autor

        try:
            loja.publicar(
                instancia=instancia_id,
                autor=autor,
                texto=texto,
                responde_a=int(responde_a) if responde_a else None,
            )
        except ValueError as erro:
            abort(400, str(erro))

        return redirect(url_for("feed", instancia_id=instancia_id))

    return app


app = criar_app()


if __name__ == "__main__":
    print(f"rede simulada | db={config.REDE_DB_PATH} | bot={loja.BOT_HANDLE}")
    # host 0.0.0.0 para a página ser alcançável de fora do container.
    app.run(host="0.0.0.0", port=5000, debug=False)
