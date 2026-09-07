"""
As instâncias da rede simulada.

DUAS, e duas bastam. A premissa que o projeto quer demonstrar é que cada grupo
tem a sua infraestrutura, os seus dados e as suas regras — e isso ou é visível
com duas, ou não fica mais visível com cinquenta. Cada instância a mais é só
mais conteúdo de exemplo para escrever.

As "regras" de cada uma são texto, não código: aqui elas existem para deixar à
vista que instâncias diferentes têm políticas diferentes, que é a parte da tese
que o TCC vai implementar de verdade. Chamar isto de moderação seria mentira.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instancia:
    id: str
    nome: str
    descricao: str
    regra: str
    cor: str


INSTANCIAS: tuple[Instancia, ...] = (
    Instancia(
        id="computacao.ufrrj",
        nome="Computação",
        descricao="Alunos e docentes do Instituto Multidisciplinar e do DCC.",
        regra="Dados ficam nesta instância. Sem indexação externa.",
        cor="#3b4877",   # navy do SIGAA
    ),
    Instancia(
        id="matematica.ufrrj",
        nome="Matemática",
        descricao="Comunidade do Departamento de Matemática.",
        regra="Só membros verificados publicam. Histórico apagado a cada período.",
        cor="#7a5230",   # da familia do ambar do SIGAA
    ),
)

POR_ID = {i.id: i for i in INSTANCIAS}
PADRAO = INSTANCIAS[0]


def existe(instancia_id: str) -> bool:
    return instancia_id in POR_ID
