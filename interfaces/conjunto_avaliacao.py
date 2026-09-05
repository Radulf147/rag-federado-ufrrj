"""
Conjunto de avaliação PRÉ-REGISTRADO da fase 3 — perguntas, rotas e verdades-base.

ESTE ARQUIVO É UM PRÉ-REGISTRO. Ele foi escrito e commitado ANTES de qualquer
execução da bateria, e é isso que dá validade à métrica de roteamento.

POR QUÊ: eu escrevi o `SYSTEM_PROMPT` que decide a rota. Se eu rotulasse as
perguntas depois de ver o agente rodar, rotularia — sem querer — na direção do
que ele faz, e a "acurácia de roteamento" viraria uma tautologia: mediria minha
capacidade de descrever o comportamento observado, não a capacidade do agente de
escolher o caminho certo. Rotular antes, e deixar o commit provar que foi antes,
é o que separa medida de profecia autorrealizável.

Mudar um rótulo depois de ver resultado é legítimo, mas só como decisão
explícita e registrada — nunca como ajuste silencioso.

A ROTA É DEFINIDA PELA NATUREZA DA PERGUNTA, NÃO PELA RESPOSTA
--------------------------------------------------------------
    estruturada  contagem, listagem, vínculo docente <-> departamento
    semantica    conteúdo do perfil: formação, áreas de interesse, atuação
    ambigua      exige os dois caminhos — a classe mais informativa, e a que
                 distingue um roteador de verdade de um classificador de
                 palavra-chave
    nenhuma      fora do escopo do corpus, ou dado que sabidamente não existe

A VERDADE-BASE É CALCULADA, NUNCA ESCRITA À MÃO
-----------------------------------------------
Nenhum número aparece literal aqui. Se eu registrasse "Matemática = 44", esse 44
apodreceria na próxima recarga e a bateria passaria a medir a defasagem deste
arquivo em vez do agente. As funções abaixo consultam o SQLite no momento da
execução.

E elas consultam **direto**, com normalização própria, em vez de reusar
`db_manager.buscar_entidades_por_campo`. Isso é deliberado: a verdade-base não
pode passar pelo mesmo código que está sendo avaliado. Se a tool tiver um defeito
de casamento de departamento, uma verdade-base que use a mesma função herda o
defeito e o torna invisível. É a lição do achado 08 — não se audita um sistema
com ele mesmo.
"""

import json
import sqlite3
import unicodedata
from dataclasses import dataclass, field
from typing import Callable

import config

# ---------------------------------------------------------------- verdade-base


def _normalizar(texto: str) -> str:
    decomposto = unicodedata.normalize("NFKD", texto or "")
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.upper().split())


def _docentes() -> list[dict]:
    """Lê o SQLite direto. Independente de db_manager, de propósito."""
    with sqlite3.connect(config.DB_PATH) as conexao:
        cursor = conexao.cursor()
        cursor.execute(
            "SELECT dados_brutos FROM entidades_sigaa WHERE tipo_entidade = 'docente'"
        )
        return [json.loads(linha[0]) for linha in cursor.fetchall()]


def por_departamento(fragmento: str) -> dict:
    """
    Departamentos cujo nome contém o fragmento, e quem está em cada um.

    Devolve TODOS os que casam, não a soma. Quando o fragmento casa com mais de
    um departamento — "Geografia" casa com o de Seropédica e o do IM — somar os
    dois é resposta errada, e a resposta certa é relatar a ambiguidade. A forma
    do retorno carrega essa distinção em vez de escondê-la num total.
    """
    alvo = _normalizar(fragmento)
    encontrados: dict[str, list[str]] = {}
    for registro in _docentes():
        departamento = registro.get("departamento") or ""
        if alvo in _normalizar(departamento):
            encontrados.setdefault(departamento, []).append(registro.get("nome", ""))
    return {
        "tipo": "departamentos",
        "ambiguo": len(encontrados) > 1,
        "departamentos": {d: sorted(nomes) for d, nomes in sorted(encontrados.items())},
        "contagens": {d: len(nomes) for d, nomes in sorted(encontrados.items())},
    }


def departamento_de(nome_docente: str) -> dict:
    """Em que departamento(s) um docente aparece."""
    alvo = _normalizar(nome_docente)
    departamentos = sorted(
        {
            r.get("departamento", "")
            for r in _docentes()
            if alvo in _normalizar(r.get("nome", ""))
        }
    )
    return {"tipo": "vinculo", "docente": nome_docente, "departamentos": departamentos}


# ---------------------------------------------------------------- as perguntas


@dataclass(frozen=True)
class Pergunta:
    id: str
    texto: str
    rota: str
    porque: str
    verdade: Callable[[], dict] | None = field(default=None, repr=False)


ROTAS_VALIDAS = {"estruturada", "semantica", "ambigua", "nenhuma"}


CONJUNTO: list[Pergunta] = [
    # ------------------------------------------------------------ estruturada
    Pergunta(
        "est-01",
        "Quantos docentes tem o Departamento de Matemática?",
        "estruturada",
        "Contagem exata sobre vínculo docente-departamento. O texto do perfil é irrelevante.",
        lambda: por_departamento("MATEMÁTICA"),
    ),
    Pergunta(
        "est-02",
        "Quantos professores estão lotados no Departamento de Ciências Sociais?",
        "estruturada",
        "Contagem. 'Professores' e 'lotados' não mudam a natureza da pergunta.",
        lambda: por_departamento("CIÊNCIAS SOCIAIS"),
    ),
    Pergunta(
        "est-03",
        "Liste os docentes do Departamento de Geociências.",
        "estruturada",
        "Listagem nominal — vem do banco estruturado, não de similaridade de texto.",
        lambda: por_departamento("GEOCIÊNCIAS"),
    ),
    Pergunta(
        "est-04",
        "Quantos docentes há no Departamento de Química Analítica?",
        "estruturada",
        "Contagem em departamento pequeno; testa se o roteamento independe do tamanho.",
        lambda: por_departamento("QUÍMICA ANALÍTICA"),
    ),
    Pergunta(
        "est-05",
        "O Departamento de Engenharia Química tem quantos professores?",
        "estruturada",
        "Contagem com a pergunta invertida na ordem — mesma natureza.",
        lambda: por_departamento("ENGENHARIA QUÍMICA"),
    ),
    Pergunta(
        "est-06",
        "Quais docentes pertencem ao Departamento de Bioquímica?",
        "estruturada",
        "Listagem nominal.",
        lambda: por_departamento("BIOQUÍMICA"),
    ),
    Pergunta(
        "est-07",
        "Em qual departamento trabalha o professor Marcel William Rocha da Silva?",
        "estruturada",
        "Vínculo de uma pessoa a um departamento: dado exato, está no SQLite.",
        lambda: departamento_de("MARCEL WILLIAM ROCHA DA SILVA"),
    ),
    Pergunta(
        "est-08",
        "Quantos docentes tem o Departamento de Geografia?",
        "estruturada",
        "Contagem, MAS o nome casa com dois departamentos (Seropédica e IM). A "
        "resposta certa relata a ambiguidade; somar os dois é o defeito do achado 06.",
        lambda: por_departamento("GEOGRAFIA"),
    ),
    Pergunta(
        "est-09",
        "Quantos professores tem o Departamento de Ciências Jurídicas?",
        "estruturada",
        "Segundo caso de departamento homônimo, para a ambiguidade não ser medida "
        "por uma única observação.",
        lambda: por_departamento("CIÊNCIAS JURÍDICAS"),
    ),
    # -------------------------------------------------------------- semantica
    Pergunta(
        "sem-01",
        "Quais docentes pesquisam agroecologia?",
        "semantica",
        "Área de pesquisa vive no texto livre do perfil; não há coluna para isso.",
    ),
    Pergunta(
        "sem-02",
        "Quem trabalha com movimentos sociais na universidade?",
        "semantica",
        "Tema de pesquisa, sem recorte de departamento.",
    ),
    Pergunta(
        "sem-03",
        "Que professores atuam na área de formação de professores?",
        "semantica",
        "Área de atuação, texto livre.",
    ),
    Pergunta(
        "sem-04",
        "Há docentes que pesquisam segurança alimentar?",
        "semantica",
        "Tema de pesquisa. Pergunta de existência, mas a evidência é textual.",
    ),
    Pergunta(
        "sem-05",
        "Qual é a formação acadêmica de Filipe Braida do Carmo?",
        "semantica",
        "Conteúdo do perfil de uma pessoa específica — o campo Formação é texto livre.",
    ),
    Pergunta(
        "sem-06",
        "Quais são as áreas de interesse de Marcel William Rocha da Silva?",
        "semantica",
        "Conteúdo de perfil. Verificado em 4 set: este docente NÃO preencheu o campo. "
        "A resposta correta é dizer que não consta, não inferir a partir do departamento.",
    ),
    Pergunta(
        "sem-07",
        "Quem pesquisa ecologia?",
        "semantica",
        "Tema de pesquisa.",
    ),
    Pergunta(
        "sem-08",
        "Que docentes trabalham com literatura?",
        "semantica",
        "Tema de pesquisa.",
    ),
    Pergunta(
        "sem-09",
        "Algum professor atua com didática?",
        "semantica",
        "Tema de pesquisa; medido na calibração, é o termo com pior densidade lexical "
        "no topo — bom caso limite.",
    ),
    # ---------------------------------------------------------------- ambigua
    Pergunta(
        "amb-01",
        "Quem do Departamento de Matemática pesquisa estatística?",
        "ambigua",
        "Recorte estruturado (o departamento) mais filtro semântico (o tema). "
        "Nenhum caminho sozinho responde.",
        lambda: por_departamento("MATEMÁTICA"),
    ),
    Pergunta(
        "amb-02",
        "Quais docentes de Ciências Sociais trabalham com movimentos sociais?",
        "ambigua",
        "Departamento exato + tema textual.",
        lambda: por_departamento("CIÊNCIAS SOCIAIS"),
    ),
    Pergunta(
        "amb-03",
        "Há alguém no Departamento de Ciência da Computação que pesquise inteligência artificial?",
        "ambigua",
        "Departamento exato + tema textual. O fragmento casa só com o do IM: o "
        "de Seropédica chama-se \"Departamento de Computação\", sem \"Ciência da\". "
        "Um humano poderia querer os dois — a verdade-base registra o que o nome "
        "pedido de fato seleciona.",
        lambda: por_departamento("CIÊNCIA DA COMPUTAÇÃO"),
    ),
    Pergunta(
        "amb-04",
        "Quem no Departamento de Letras e Comunicação Social estuda literatura?",
        "ambigua",
        "Departamento exato + tema textual.",
        lambda: por_departamento("LETRAS E COMUNICAÇÃO SOCIAL"),
    ),
    Pergunta(
        "amb-05",
        "Quantos docentes do Departamento de História e Relações Internacionais pesquisam política?",
        "ambigua",
        "Pede contagem (estruturado) de um subconjunto definido por tema (semântico). "
        "Um número aqui exige os dois caminhos, e responder só com o total do "
        "departamento seria erro.",
        lambda: por_departamento("HISTÓRIA E RELAÇÕES INTERNACIONAIS"),
    ),
    Pergunta(
        "amb-06",
        "Algum professor de Engenharia Agrícola e Ambiental trabalha com agroecologia?",
        "ambigua",
        "Departamento exato + tema textual.",
        lambda: por_departamento("ENGENHARIA AGRÍCOLA E AMBIENTAL"),
    ),
    Pergunta(
        "amb-07",
        "Entre os docentes do Departamento de Formação Docente do IM, quem atua com didática?",
        "ambigua",
        "Departamento exato + tema textual.",
        lambda: por_departamento("FORMAÇÃO DOCENTE"),
    ),
    # ---------------------------------------------------------------- nenhuma
    Pergunta(
        "nen-01",
        "Qual a previsão do tempo para amanhã em Seropédica?",
        "nenhuma",
        "Fora do escopo por completo. Não deve acionar ferramenta nenhuma.",
    ),
    Pergunta(
        "nen-02",
        "Como faço minha inscrição no vestibular da UFRRJ?",
        "nenhuma",
        "Sobre a UFRRJ, mas não sobre docentes. O corpus não tem isso, e a "
        "proximidade temática é justamente o que torna o caso difícil.",
    ),
    Pergunta(
        "nen-03",
        "Qual o horário de funcionamento da biblioteca central?",
        "nenhuma",
        "Informação institucional ausente do corpus de docentes.",
    ),
    Pergunta(
        "nen-04",
        "Qual o salário dos professores do Departamento de Matemática?",
        "nenhuma",
        "Dado que sabidamente não existe na nossa base. Menciona um departamento "
        "real, então testa se o roteador se deixa levar por palavra-chave.",
    ),
    Pergunta(
        "nen-05",
        "Qual foi a nota do curso de Computação no ENADE?",
        "nenhuma",
        "Avaliação de curso, não dado de docente. Outro caso de proximidade temática.",
    ),
]


def validar() -> None:
    """Falha alto se o conjunto estiver malformado, antes de gastar uma bateria."""
    vistos = set()
    for pergunta in CONJUNTO:
        if pergunta.rota not in ROTAS_VALIDAS:
            raise ValueError(f"{pergunta.id}: rota inválida {pergunta.rota!r}")
        if pergunta.id in vistos:
            raise ValueError(f"id duplicado: {pergunta.id}")
        vistos.add(pergunta.id)
        if not pergunta.porque.strip():
            raise ValueError(f"{pergunta.id}: sem justificativa de rota")


if __name__ == "__main__":
    from collections import Counter

    validar()
    distribuicao = Counter(p.rota for p in CONJUNTO)
    print(f"{len(CONJUNTO)} perguntas pré-registradas")
    for rota, n in sorted(distribuicao.items()):
        com_verdade = sum(1 for p in CONJUNTO if p.rota == rota and p.verdade)
        print(f"  {rota:12} {n:>3}   com verdade-base calculável: {com_verdade}")
    print()
    print("Verdades-base resolvidas agora (prova de que calculam, não são literais):")
    for pergunta in CONJUNTO:
        if not pergunta.verdade:
            continue
        valor = pergunta.verdade()
        if valor["tipo"] == "departamentos":
            resumo = ", ".join(f"{d.split('DEPARTAMENTO DE ')[-1][:34]}={n}"
                               for d, n in valor["contagens"].items())
            marca = "  <-- AMBIGUO" if valor["ambiguo"] else ""
            print(f"  {pergunta.id}  {resumo}{marca}")
        else:
            print(f"  {pergunta.id}  {valor['docente']} -> {valor['departamentos']}")
