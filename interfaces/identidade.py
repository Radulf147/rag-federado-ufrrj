"""
Identidade e rótulo de uma entidade — a decisão D1 de
`docs/arquitetura_multi_entidade.md`.

DOIS CAMPOS, DOIS TRABALHOS
---------------------------
`nome_docente` fazia os dois, e só um deles é exibição:

    identidade(meta) -> chave estável: dedupe, chave do RRF, junção do D4
    rotulo(meta)     -> texto para o LLM ler, e só isso

Separá-los não é arrumação. Chaveado por nome, dois documentos de pessoas
diferentes com o mesmo nome colapsam — e em `fundir()` é pior que colapsar: o
segundo some da saída **e a pontuação dele soma na do primeiro**. Documento
perdido e nota inflada, de uma vez. Demonstrado em
`testes/test_identidade_entidade.py`, que reprovou a versão anterior.

Não é hipótese: `FERNANDA SILVA FERREIRA CHAER` está duas vezes no corpus (dois
SIAPEs, dois departamentos — a duplicação é da fonte, item 10 do backlog). Hoje
isso não move número nenhum das 6 medições, porque ela não escreveu nenhum dos
temas medidos, e **o tamanho do efeito não é o argumento**: na listagem de
cursos, `CIÊNCIAS BIOLÓGICAS` aparece duas vezes, mesmo campus, Bacharelado e
Licenciatura, ids distintos.

⚠️ POR QUE NÃO O `Document.id` DO HAYSTACK
------------------------------------------
Ele é hash do conteúdo. O mesmo `FILIPE BRAIDA DO CARMO` tem três ids
diferentes nas três coleções que existem hoje (`rag_sigaa`,
`rag_sigaa_descritivo`, `rag_sigaa_filtrado`), porque reindexar muda o texto.
Uma identidade que não sobrevive a uma decisão de indexação não é identidade.

⚠️ O RECUO DERIVA, NÃO ADIVINHA — e é isso que torna a mudança segura
---------------------------------------------------------------------
Documento gravado antes desta decisão não tem `id_entidade`. Em vez de falhar
ou de inventar, o recuo **deriva a identidade do que já está lá**: todo docente
do corpus tem `siape`, e são 1302 siapes distintos para 1302 documentos
(conferido em 7 set 2026).

Consequência prática: quem LÊ funciona antes de qualquer migração, e o
`rotulo` recua para `nome_docente`, o que mantém a saída da tool **idêntica,
byte a byte**, para docente. É o critério de aceite do D0.
"""


def identidade(meta: dict) -> str:
    """
    Chave estável da entidade. Nunca é o rótulo.

    Ordem: o campo explícito; depois o que dá para derivar do metadado antigo;
    e só no fim o nome — que é chave ruim, e por isso vem marcado.
    """
    explicito = (meta or {}).get("id_entidade")
    if explicito:
        return str(explicito)

    siape = (meta or {}).get("siape")
    if siape:
        return f"docente:{siape}"

    # Último recuo. Um documento sem `id_entidade` e sem `siape` não tem
    # identidade nenhuma no metadado; usar o nome preserva o comportamento
    # anterior em vez de descartar o documento, e o prefixo deixa visível na
    # depuração que aquela chave é frágil.
    nome = (meta or {}).get("nome_docente") or (meta or {}).get("rotulo")
    return f"sem-id:{nome}" if nome else "sem-id:(anonimo)"


def rotulo(meta: dict) -> str:
    """Texto que identifica a entidade para quem lê a resposta."""
    meta = meta or {}
    return (
        meta.get("rotulo")
        or meta.get("nome_docente")
        # A frase abaixo é a do achado 02 e continua sendo a certa: o LLM tem
        # de ver que a atribuição está faltando, em vez de receber texto sem
        # dono e inventar de quem é.
        or "(nome ausente no metadado)"
    )
