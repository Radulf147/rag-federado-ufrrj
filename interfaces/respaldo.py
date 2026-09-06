"""
`respaldo_de_citacao` — existe respaldo no corpus para esta citação?

MÉTRICA EXPLORATÓRIA. Não julga a fase 3 e não é o contrapeso da v2a — a v2a é
relaxamento estrito e contrapeso interno a ela é impossível (ver
docs/criterios_avaliacao.md). Isto mede outra coisa: é proxy parcial e offline de
`nomes_sem_respaldo`, a verificação de tolerância zero que ficou congelada porque
o JSONL grava o contexto recuperado só como tamanho.

Parcial em dois sentidos, os dois declarados: o documento existir no corpus não
prova que chegou ao agente, e o casamento de palavra não decide o que um perfil
substantivo sem a palavra significa. A segunda limitação é o motivo de existirem
TRÊS classes.

O CASAMENTO É DELIBERADAMENTE CRU: frase completa do tema, como substring do
texto descritivo normalizado. Nada de radical, sinônimo ou similaridade. Um
casamento esperto acertaria mais e seria impossível de auditar — e a classe
INCONCLUSIVO existe exatamente para absorver o que a crueza não decide, em vez de
transformá-la em falha do sistema medido.
"""

import re
import unicodedata

SEM_RESPALDO = "SEM_RESPALDO"
COM_RESPALDO = "COM_RESPALDO"
INCONCLUSIVO = "INCONCLUSIVO"

# Corte ESTRUTURAL, fixado em docs/criterios_avaliacao.md antes de rodar: o
# perfil tem conteúdo substantivo se traz campo descritivo. Não é limiar de
# caracteres calibrado no dado.
CAMPOS_DESCRITIVOS = ("PERFIL", "FORMACAO", "AREAS DE INTERESSE")

# `Currículo Lattes` é institucional apesar de soar acadêmico: está em 52,3% dos
# perfis e, quando vazio, traz o literal "link não informado". É ponteiro, não
# conteúdo.
CAMPOS_INSTITUCIONAIS = (
    "DOCENTE", "DEPARTAMENTO", "CURRICULO LATTES", "ENDERECO", "SALA",
    "TELEFONE", "E-MAIL", "CEP", "ORIENTADOR", "TITULO",
)

TODOS_OS_CAMPOS = CAMPOS_DESCRITIVOS + CAMPOS_INSTITUCIONAIS


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.upper().split())


_RE_CAMPO = re.compile(
    r"(?:^|(?<=[.\s]))(" + "|".join(re.escape(c) for c in TODOS_OS_CAMPOS) + r"):"
)


def secoes(conteudo: str) -> dict[str, str]:
    """Divide o documento do docente nos campos que o ETL escreveu."""
    texto = _normalizar(conteudo)
    marcas = [(m.start(), m.group(1), m.end()) for m in _RE_CAMPO.finditer(texto)]
    out: dict[str, str] = {}
    for i, (_ini, campo, fim) in enumerate(marcas):
        prox = marcas[i + 1][0] if i + 1 < len(marcas) else len(texto)
        out[campo] = texto[fim:prox].strip(" .")
    return out


def texto_descritivo(conteudo: str) -> str:
    """
    Só o que o docente escreveu sobre si — Perfil, Formação, Áreas de interesse.

    O nome do departamento fica DE FORA por construção, e é o ponto todo: sem
    isso, "movimentos sociais" casaria com todo docente do DEPARTAMENTO DE
    EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE, que é precisamente a
    colisão que originou esta métrica.
    """
    campos = secoes(conteudo)
    return " ".join(campos[c] for c in CAMPOS_DESCRITIVOS if campos.get(c))


def tem_conteudo_substantivo(conteudo: str) -> bool:
    campos = secoes(conteudo)
    return any(campos.get(c) for c in CAMPOS_DESCRITIVOS)


def classificar(conteudo: str, tema: str) -> dict:
    """
    SEM_RESPALDO   perfil sem conteúdo substantivo — não há o que casar
    COM_RESPALDO   perfil substantivo e a frase do tema aparece nele
    INCONCLUSIVO   perfil substantivo e a frase não aparece — NÃO é falha
    """
    descritivo = texto_descritivo(conteudo)
    if not descritivo:
        return {
            "classe": SEM_RESPALDO,
            "chars_descritivos": 0,
            "chars_lidos": len(conteudo),
            "truncado": False,
        }
    alvo = _normalizar(tema)
    return {
        "classe": COM_RESPALDO if alvo in descritivo else INCONCLUSIVO,
        "chars_descritivos": len(descritivo),
        "chars_lidos": len(conteudo),
        "truncado": False,
    }


def resumir(classificacoes: list[str]) -> dict:
    """
    Intervalo [com respaldo ; com respaldo + inconclusivo].

    Mesma disciplina dos AMBÍGUOS: o que o instrumento não consegue decidir
    aparece como LARGURA do intervalo, nunca como veredito.
    """
    com = classificacoes.count(COM_RESPALDO)
    inconclusivos = classificacoes.count(INCONCLUSIVO)
    sem = classificacoes.count(SEM_RESPALDO)
    total = len(classificacoes)
    return {
        "citados": total,
        "com_respaldo": com,
        "inconclusivos": inconclusivos,
        "sem_respaldo": sem,
        "intervalo": [com, com + inconclusivos],
        "fracao_minima": f"{com} de {total}",
        "fracao_maxima": f"{com + inconclusivos} de {total}",
    }
