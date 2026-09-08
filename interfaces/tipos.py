"""
Registro de tipos de entidade — a decisão D0 de
`docs/arquitetura_multi_entidade.md`.

O PROBLEMA QUE ELE RESOLVE
--------------------------
O projeto foi construído sobre uma suposição nunca escrita: existe um tipo de
entidade, o docente. No caminho adaptativo, cada tipo novo toca **cinco
lugares** — o coletor, a `tools.py`, o `SYSTEM_PROMPT`, a indexação e os
medidores. Vêm seis tipos: trinta edições, cada uma uma chance do erro
silencioso que `docs/backlog_avaliacao.md` vem catalogando.

Com o registro, tipo novo é **uma entrada**.

⚠️ O REGISTRO GERA TOOLS NOMEADAS, NÃO UMA TOOL GENÉRICA
---------------------------------------------------------
A forma elegante seria `buscar(tipo, campo, valor)`: bonita no nosso código e
**pior para o LLM** — três parâmetros livres no lugar de uma ferramenta nomeada
e descrita. O item 8 do backlog mediu em 7 set 2026 que o LLM, neste projeto, é
um mau planejador de consulta: com a pergunta do usuário inteira o
`FILIPE BRAIDA` vem em posição 1; com a reescrita que o próprio agente emitiu,
sai do TOP_10.

O registro tira trabalho nosso **sem transferir trabalho para o LLM**.

⚠️ AS DESCRIÇÕES SÃO LITERAIS, E ISSO NÃO É PREGUIÇA
-----------------------------------------------------
Cada `descricao` abaixo é o texto EXATO que hoje está em `TOOLS_SCHEMA`, e cada
uma foi reescrita para consertar uma falha de roteamento medida — a de
`buscar_docente_por_nome` custou seis execuções na bateria de 5 set, e a de
`busca_vetorial_sigaa` carrega a correção de "perguntas genéricas
interpretativas". São o artefato mais medido do projeto: o roteamento de 97,8%
da fase 3 é uma propriedade **destes textos**.

Gerá-las a partir de um molde trocaria uma medição por uma conveniência.
`testes/test_registro_tipos.py` fixa a igualdade contra `TOOLS_SCHEMA`.

O QUE ESTE MÓDULO AINDA NÃO FAZ (7 set 2026)
--------------------------------------------
**Ninguém o lê.** É o passo 3 da ordem do D0, e é declarativo de propósito: o
registro existe e é conferido contra a realidade ANTES de qualquer código
depender dele. A geração das tools é o passo 4, e é lá que `formatar` — como o
resultado de cada busca vira texto — entra.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Busca:
    """
    Uma forma de procurar entidades de um tipo, e a tool que ela vira.

    `campo` é a chave dentro do JSON de `entidades_sigaa.dados_brutos`, que é o
    que `db_manager.buscar_entidades_por_campo` recebe.

    ⚠️ OS SUBSTANTIVOS SÃO DECLARADOS, NÃO DERIVADOS. `plural_do_campo` não sai
    de `campo + "s"`, e `artigo` não sai de heurística de terminação: português
    tem "unidade"/"unidades" feminino ao lado de "centro"/"centros" masculino, e
    uma regra esperta acertaria hoje e erraria calada no primeiro tipo novo. O
    texto que o LLM lê é medido — declarar é mais barato que depurar.
    """

    campo: str
    nome_tool: str
    descricao: str
    parametro: str
    descricao_parametro: str
    # "agrupado": conta e lista por grupo, relatando ambiguidade
    # "um_ou_ambiguo": espera um; se vier mais de um, relata em vez de escolher
    formato: str
    singular_do_campo: str
    plural_do_campo: str
    artigo: str = "o"


@dataclass(frozen=True)
class Tipo:
    """Tudo que o resto do sistema precisa saber sobre um tipo de entidade."""

    nome: str
    identidade: Callable[[dict], str]
    rotulo: Callable[[dict], str]
    # Substantivos que entram no texto devolvido ao LLM. Declarados pelo mesmo
    # motivo dos de `Busca`. `referente` é como a entidade é retomada numa
    # frase — "a pessoa", "o departamento" —, e existe porque a guarda dos dois
    # zeros diz "não responda como se a pessoa não existisse".
    singular: str
    plural: str
    referente: str
    # Campo que identifica a entidade numa listagem, e o que é mostrado ao lado
    # dela quando o resultado é único.
    campo_rotulo: str
    campo_vinculo: str
    # None significa ESTRUTURA PURA: não vai para o Chroma. É a correção do D2
    # — indexar um registro sem texto livre é vetorizar um nome, que é
    # exatamente o defeito que o item 7 mediu e removeu para levar o recall de
    # 14% para 27%.
    texto_semantico: Callable[[dict], str] | None
    buscas: tuple[Busca, ...]

    @property
    def vai_para_o_chroma(self) -> bool:
        return self.texto_semantico is not None

    def metadados(self, entidade: dict) -> dict:
        """
        Metadado padrão do Chroma — o MESMO formato para todo tipo.

        É isto que impede o D1 de virar um `if tipo == ...` espalhado: a forma
        do metadado é decidida aqui, uma vez.
        """
        return {
            "tipo": self.nome,
            "id_entidade": self.identidade(entidade),
            "rotulo": self.rotulo(entidade),
            "instancia_dona": entidade.get("instancia_dona", "sigaa"),
            "source_url": entidade.get("source_url", ""),
            "scraped_at": entidade.get("scraped_at", ""),
        }


# --------------------------------------------------------------------------
# DOCENTE — o tipo que já existia, agora declarado
#
# ⚠️ `texto_semantico` devolve o conteúdo do documento COMO ESTÁ HOJE, e não o
# texto descritivo. Não é descuido: a troca para a coleção descritiva está
# SUSPENSA por decisão registrada (item 7 do backlog — o 27% é real, e se paga
# por ele com uma afirmação falsa sobre uma pessoa). O registro é justamente
# onde essa decisão vira uma linha, quando os itens 8 e 9 estiverem corrigidos.
# --------------------------------------------------------------------------
DOCENTE = Tipo(
    nome="docente",
    identidade=lambda e: f"docente:{e.get('siape')}",
    rotulo=lambda e: e.get("nome") or e.get("nome_docente") or "",
    singular="docente",
    plural="docentes",
    referente="a pessoa",
    campo_rotulo="nome",
    campo_vinculo="departamento",
    texto_semantico=lambda e: e.get("conteudo") or "",
    buscas=(
        Busca(
            campo="departamento",
            nome_tool="buscar_docentes_por_departamento",
            formato="agrupado",
            singular_do_campo="departamento",
            plural_do_campo="departamentos",
            descricao=(
                "Utilize esta ferramenta APENAS quando o usuário pedir para "
                "contar ou listar os professores/docentes de um departamento "
                "específico (ex: Computação, Física). Retorna dados exatos."
            ),
            parametro="departamento",
            descricao_parametro=(
                "Nome ou sigla do departamento que o usuário deseja "
                "buscar (ex: Ciência da Computação, Matemática)"
            ),
        ),
        Busca(
            campo="nome",
            nome_tool="buscar_docente_por_nome",
            formato="um_ou_ambiguo",
            singular_do_campo="nome",
            plural_do_campo="nomes",
            # A redação anterior abria com "quando o usuário perguntar sobre UM
            # docente específico pelo nome", e casava com QUALQUER pergunta que
            # citasse uma pessoa. Na bateria de 5 set derrubou "qual é a
            # formação acadêmica de Filipe Braida?" e "quais são as áreas de
            # interesse de Marcel?" — todos os erros de roteamento da rodada.
            descricao=(
                "Vínculo de UMA pessoa: dado o nome, diz a que departamento "
                "ela pertence, ou que não está cadastrada. Isso é tudo o que "
                "devolve. NÃO tem formação acadêmica, áreas de interesse, "
                "atuação, contato nem qualquer outro texto do perfil — para "
                "esses use busca_vetorial_sigaa, inclusive quando a pergunta "
                "nomear a pessoa."
            ),
            parametro="nome",
            descricao_parametro=(
                "Nome, ou parte do nome, do docente procurado "
                "(ex: Marcel William Rocha da Silva)"
            ),
        ),
    ),
)


TIPOS: dict[str, Tipo] = {
    DOCENTE.nome: DOCENTE,
}


def tipos_do_chroma() -> tuple[str, ...]:
    """Os tipos que têm texto livre — os únicos que a busca semântica alcança."""
    return tuple(nome for nome, t in TIPOS.items() if t.vai_para_o_chroma)
