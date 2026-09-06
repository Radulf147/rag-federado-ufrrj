"""
Cenário de exemplo da rede simulada.

    docker compose run --rm --no-deps rede python -m interfaces.rede.semear

APAGA E RECRIA. É cenário de demonstração, não dado de pesquisa — a mesma
disciplina do ETL, que trata cada execução como retrato completo em vez de
incremento (achado 10).

OS POSTS NÃO SÃO ENFEITE
------------------------
Cada thread aqui existe para exercitar um caso diferente, e as perguntas foram
escolhidas para que a diferença apareça ao vivo:

    1. pergunta que SÓ funciona com o contexto da thread ("esse", "deles")
    2. pergunta completa, que funcionaria sem contexto nenhum
    3. pergunta cuja resposta honesta é "não sei" (dado que o SIGAA não tem)
    4. thread sem menção nenhuma — o bot tem de ficar quieto

A quarta é a que se esquece de testar: um bot que responde onde não foi chamado
é tão defeituoso quanto um que não responde onde foi.

NENHUMA RESPOSTA DO BOT É SEMEADA. Se aparecesse resposta pronta aqui, a
demonstração mostraria texto escrito por mim com selo de bot — exatamente a
coisa que este projeto não pode fazer.
"""

from interfaces.rede import loja

COMP = "computacao.ufrrj"
MAT = "matematica.ufrrj"


def semear() -> None:
    loja.init_db()
    loja.apagar_tudo()

    # 1. Contexto é indispensável: "esse" e "deles" não significam nada soltos.
    t1 = loja.publicar(
        COMP, "@raul",
        "gente, quantos professores tem o Departamento de Computação hoje?",
    )
    loja.publicar(
        COMP, "@bia",
        "@ufrrj e a Matemática, tem mais ou menos que esse?",
        responde_a=t1,
    )

    # 2. Pergunta completa — funciona com ou sem contexto. É a comparação.
    t2 = loja.publicar(
        COMP, "@ana",
        "montando o horário do período que vem, alguém tem dica de optativa?",
    )
    loja.publicar(
        COMP, "@joao",
        "@ufrrj quais docentes do Departamento de Ciência da Computação "
        "trabalham com inteligência artificial?",
        responde_a=t2,
    )

    # 3. A resposta honesta é não saber: nota de disciplina não está no SIGAA
    #    público, e o agente tem de dizer isso em vez de inventar.
    t3 = loja.publicar(
        MAT, "@carla",
        "alguém sabe quando sai o resultado da monitoria?",
    )
    loja.publicar(
        MAT, "@carla",
        "@ufrrj qual a nota de corte da monitoria de Cálculo 1 esse ano?",
        responde_a=t3,
    )

    # 4. Thread sem menção: o bot NÃO pode responder aqui.
    t4 = loja.publicar(
        MAT, "@pedro",
        "o café do IM voltou a funcionar, aviso de utilidade pública",
    )
    loja.publicar(MAT, "@bia", "notícia boa demais pra ser verdade", responde_a=t4)

    fila = loja.pendentes()
    print(f"posts criados ......... {len(loja.thread(t1)) + len(loja.thread(t2)) + len(loja.thread(t3)) + len(loja.thread(t4))}")
    print(f"instâncias ............ {loja.instancias()}")
    print(f"na fila do bot ........ {[p['id'] for p in fila]}")
    for p in fila:
        print(f"    [{p['instancia']}] {p['autor']}: {p['texto'][:60]}")


if __name__ == "__main__":
    semear()
