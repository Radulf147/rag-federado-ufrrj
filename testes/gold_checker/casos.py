"""
Gold set do checker de atribuição departamental — rótulos feitos à mão.

CADA RÓTULO CARREGA A EVIDÊNCIA. Não basta afirmar "este caso aprova": o caso
traz o trecho literal da resposta que o justifica e o que a base diz sobre cada
docente envolvido. Isso existe porque, ao revisar as 4 falhas do v1 na mão, eu
descrevi DUAS delas erradas — `amb-04` e `amb-06` — por ter lido só o começo do
texto. Rótulo sem citação é asserção, e minhas asserções já falharam.

SINTÉTICAS SÃO MUTAÇÃO MÍNIMA de resposta gravada, nunca texto inventado do
zero. O diff de cada uma está no campo `mutacao`. Motivo: com zero reprovações
em dado real, as sintéticas carregam sozinhas a prova de que o instrumento
morde — e uma sintética escrita à mão provaria apenas que o checker entende o
texto que eu escrevi para ele entender.

VEREDITOS POSSÍVEIS
    APROVA    nenhum nome de fora sem vínculo declarado e correto
    REPROVA   nome de fora sem declaração, ou com declaração que não bate
    AMBIGUO   mais de um departamento em escopo para o mesmo nome
"""

from pathlib import Path

CASOS_DIR = Path(__file__).parent / "casos"


def texto(nome: str) -> str:
    return (CASOS_DIR / f"{nome}.txt").read_text(encoding="utf-8")


APROVA, REPROVA, AMBIGUO = "APROVA", "REPROVA", "AMBIGUO"

GOLD = [
    # ------------------------------------------------------------------ (a)
    {
        "id": "a_certa_formato_esperado",
        "arquivo": "real_amb03_r1_limpa",
        "origem": "REAL — amb-03#1, veredito gravado ok=True",
        "pergunta_id": "amb-03",
        "esperado": APROVA,
        "porque": "Nenhum nome fora do elenco. A cláusula do rótulo nem chega a ser exercida.",
        "base": "todos os nomes citados pertencem ao DEPARTAMENTO DE CIÊNCIA DA COMPUTAÇÃO/IM",
    },
    # ------------------------------------------------------------------ (b)
    {
        "id": "b_intruso_corretamente_rotulado",
        "arquivo": "real_amb06_r1_rotulado",
        "origem": "REAL — amb-06#1, veredito gravado ok=False (v1 reprovava)",
        "pergunta_id": "amb-06",
        "esperado": APROVA,
        "trecho": "- ADRIANA DE MAGALHÃES CHAVES MARTINS (DEPARTAMENTO DE CIÊNCIAS SOCIAIS)",
        "porque": (
            "Três nomes de fora, os três com vínculo declarado em parêntese "
            "imediato (Nível 1) e os três batendo com a base."
        ),
        "base": {
            "ANELISE DIAS": "DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE",
            "MARCOS BACIS CEDDIA": "DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE",
            "ADRIANA DE MAGALHAES CHAVES MARTINS": "DEPARTAMENTO DE CIÊNCIAS SOCIAIS",
        },
    },
    # ------------------------------------------------------------------ (c)
    {
        "id": "c_departamento_declarado_incorreto",
        "arquivo": "sint_c_departamento_incorreto",
        "origem": "SINTÉTICA — mutação mínima de real_amb06_r1_rotulado",
        "mutacao": (
            "- ADRIANA DE MAGALHÃES CHAVES MARTINS (DEPARTAMENTO DE CIÊNCIAS SOCIAIS)\n"
            "+ ADRIANA DE MAGALHÃES CHAVES MARTINS (DEPARTAMENTO DE MATEMÁTICA)"
        ),
        "pergunta_id": "amb-06",
        "esperado": REPROVA,
        "porque": (
            "TERCEIRO BRAÇO — vínculo declarado que não bate com a base. É "
            "alucinação com aparência de rigor. NÃO EXISTE INSTÂNCIA REAL disto "
            "nos 21 itens (auditado, inclusive para nomes dentro do elenco), "
            "então este caso é a única prova de que a regra reprova."
        ),
        "base": {"ADRIANA DE MAGALHAES CHAVES MARTINS": "DEPARTAMENTO DE CIÊNCIAS SOCIAIS"},
    },
    # ------------------------------------------------------------------ (d)
    {
        "id": "d_nome_de_fora_sem_rotulo",
        "arquivo": "sint_d_sem_rotulo",
        "origem": "SINTÉTICA — mutação mínima de real_amb06_r1_rotulado",
        "mutacao": (
            "- - ANELISE DIAS (DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE)\n"
            "+ - ANELISE DIAS"
        ),
        "pergunta_id": "amb-06",
        "esperado": REPROVA,
        "porque": (
            "SEGUNDO BRAÇO — ausência de declaração reprova. A regra não pode "
            "ser satisfeita por vagueza. Note que os outros dois nomes seguem "
            "rotulados: um único nome sem rótulo basta para reprovar o item."
        ),
        "base": {"ANELISE DIAS": "DEPARTAMENTO DE AGROTECNOLOGIAS E SUSTENTABILIDADE"},
    },
    # ------------------------------------------------------------------ (e)
    {
        "id": "e_declaracao_com_escopo_lista_depois",
        "arquivo": "real_amb02_r1_escopo",
        "origem": "REAL — amb-02#1, veredito gravado ok=False (v1 reprovava)",
        "pergunta_id": "amb-02",
        "esperado": APROVA,
        "trecho": (
            "encontrei diversos docentes vinculados ao 'DEPARTAMENTO DE EDUCAÇÃO DO "
            "CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE' que trabalham com essa temática:"
        ),
        "porque": (
            "NÍVEL 2 para a frente — uma declaração governa os dez nomes que a "
            "seguem, porque nenhum outro departamento conhecido aparece entre "
            "ela e eles. Os dez nomes estão em itens de lista, sem departamento "
            "próprio, então o Nível 1 não resolve."
        ),
        "base": "os 10 citados pertencem ao DEPARTAMENTO DE EDUCAÇÃO DO CAMPO, MOVIMENTOS SOCIAIS E DIVERSIDADE",
    },
    # ------------------------------------------------------------------ (f)
    {
        "id": "f_anafora_dois_departamentos_em_escopo",
        "arquivo": "real_amb04_r3_anafora",
        "origem": "REAL — amb-04#3, veredito gravado ok=False (v1 reprovava)",
        "pergunta_id": "amb-04",
        "esperado": AMBIGUO,
        "esperado_variante_anaforica": APROVA,
        "trecho": (
            "Estes docentes estão listados no Departamento de Letras/IM, que "
            "parece ser um departamento afiliado ou parte do mesmo grupo."
        ),
        "porque": (
            "Posições medidas no texto normalizado (892 chars): LETRAS E "
            "COMUNICAÇÃO SOCIAL em 29, os três nomes em 539/565/583, LETRAS/IM "
            "em 640. Os nomes ficam ENTRE os dois e nada intervém em nenhuma "
            "direção, então ambos entram em escopo → AMBÍGUO. Com o desempate "
            "anafórico ('Estes docentes' na lista fechada, imediatamente antes "
            "da declaração) o escopo resolve para trás → APROVA."
        ),
        "base": {
            "VALERIA ROSITO FERREIRA": "DEPARTAMENTO DE LETRAS/IM",
            "CARMEN PIMENTEL": "DEPARTAMENTO DE LETRAS/IM",
            "ROSEMARY GONCALO AFONSO": "DEPARTAMENTO DE LETRAS/IM",
        },
    },
    # ------------------------------------------------------------------ (g)
    {
        "id": "g_dois_em_escopo_sem_marcador",
        "arquivo": "sint_g_sem_marcador",
        "origem": "SINTÉTICA — mutação mínima de real_amb04_r3_anafora",
        "mutacao": (
            "- Estes docentes estão listados no Departamento de Letras/IM, ...\n"
            "+ Há registro de vínculo com o Departamento de Letras/IM, ..."
        ),
        "pergunta_id": "amb-04",
        "esperado": AMBIGUO,
        "esperado_variante_anaforica": AMBIGUO,
        "porque": (
            "Mesma geometria do caso (f), sem o marcador da lista fechada. "
            "Prova que o desempate anafórico depende do MARCADOR e não da "
            "posição: aqui ele não se aplica, e o veredito é AMBÍGUO nas duas "
            "variantes. Sem este caso, (f) sozinho não distinguiria 'o "
            "desempate funcionou' de 'a regra sempre resolve para trás'."
        ),
        "base": {"VALERIA ROSITO FERREIRA": "DEPARTAMENTO DE LETRAS/IM"},
    },
    # ------------------------------------------------------------------ (h)
    {
        "id": "h_omissao_de_docente_do_elenco",
        "arquivo": "sint_h_omissao",
        "origem": "SINTÉTICA — mutação mínima de real_est06_r3_listagem",
        "mutacao": "- - MILANE DE SOUZA LEITE      (linha removida)",
        "pergunta_id": "est-06",
        "esperado": APROVA,
        "esperado_v2b_listagem": REPROVA,
        "porque": (
            "PROVA QUE AS DUAS REGRAS ESTÃO SEPARADAS. Omitir docente do elenco "
            "não afeta precisão de atribuição — nada foi afirmado de errado — "
            "então a v2a APROVA. A v2b em listagem REPROVA, porque ali o elenco "
            "É a resposta. Se este caso der o mesmo veredito nas duas, as "
            "regras estão contaminadas uma pela outra."
        ),
        "base": "MILANE DE SOUZA LEITE pertence ao DEPARTAMENTO DE BIOQUÍMICA, elenco da est-06",
    },
    # ------------------------------------------------------------------ (i)
    {
        "id": "i_nome_que_nao_casa_com_o_corpus",
        "arquivo": "real_amb04_r1_luiz",
        "origem": "REAL — amb-04#1, veredito gravado ok=True",
        "pergunta_id": "amb-04",
        "esperado": APROVA,
        "trecho": "LUIZ CARLOS ALVES DE MELO** é Doutor em Letras",
        "porque": (
            "LIMITAÇÃO FIXADA EM TESTE, não defeito a consertar aqui. O agente "
            "escreveu LUIZ; o ETL gravou LUIS (conferido no metadado e no "
            "content do Chroma), logo a corrupção é do GERADOR. O nome "
            "corrompido não casa com nenhum dos 1297 e some da detecção: nunca "
            "vira intruso, só pode inflar a nota. Aprova, e o teste existe para "
            "que a aprovação seja uma decisão registrada e não um descuido."
        ),
        "base": {"LUIS CARLOS ALVES DE MELO": "DEPARTAMENTO DE LETRAS E COMUNICAÇÃO SOCIAL"},
    },
    # ------------------------------------------------------------------ (j)
    {
        "id": "j_dentro_do_elenco_com_departamento_errado",
        "arquivo": "sint_j_dentro_do_elenco_errado",
        "origem": "SINTÉTICA — mutação mínima de real_amb06_r1_rotulado",
        "mutacao": (
            "- - HENRIQUE VIEIRA DE MENDONCA.\n"
            "+ - HENRIQUE VIEIRA DE MENDONCA (DEPARTAMENTO DE FÍSICA)."
        ),
        "pergunta_id": "amb-06",
        "esperado": APROVA,
        "porque": (
            "PONTO CEGO POR DESENHO, fixado em teste executável para não ser "
            "esquecido. A cláusula do rótulo só alcança nomes FORA do elenco; "
            "HENRIQUE está dentro, então a atribuição falsa não é conferida e o "
            "item aprova. Auditados os 21 itens, 3 nomes de dentro recebem "
            "atribuição explícita e 0 divergem — o ponto cego existe e não está "
            "ocupado. Este caso NÃO existe para consertar a limitação; existe "
            "para que ela quebre um teste no dia em que alguém 'melhorar' a "
            "regra sem perceber que mudou o escopo."
        ),
        "base": {"HENRIQUE VIEIRA DE MENDONCA": "DEPARTAMENTO DE ENGENHARIA AGRÍCOLA E AMBIENTAL"},
    },
    # ------------------------------------------------------------- (k1, k2)
    {
        "id": "k1_despejo_do_departamento_inteiro",
        "arquivo": "real_amb02_r3_despejo",
        "origem": "REAL — amb-02#3, veredito gravado ok=True",
        "pergunta_id": "amb-02",
        "esperado": APROVA,
        "par_de_utilidade": "k2_tres_uteis",
        "porque": (
            "Cita 35 de 35 docentes do departamento. Zero intrusos, nota "
            "perfeita. Medido: 25 dos 35 não têm uma palavra sobre o tema no "
            "perfil (mediana 166 chars). É a estratégia ótima sob precisão "
            "pura, executada — e a resposta menos útil possível."
        ),
        "base": "os 35 citados pertencem ao DEPARTAMENTO DE CIÊNCIAS SOCIAIS (elenco = 35)",
    },
    {
        "id": "k2_tres_uteis",
        "arquivo": "sint_k2_tres_uteis",
        "origem": "SINTÉTICA — mutação mínima de real_amb02_r3_despejo",
        "mutacao": "removidos 32 dos 35 itens de lista; mantidos os 3 com maior evidência temática",
        "pergunta_id": "amb-02",
        "esperado": APROVA,
        "par_de_utilidade": "k1_despejo_do_departamento_inteiro",
        "porque": (
            "PAR DE UTILIDADE. Mesma pergunta, veredito idêntico ao k1, "
            "utilidade oposta: 3 docentes com evidência temática própria "
            "(ELISA GUARANA DE CASTRO 8 menções, EDSON MIAGUSKO 7, GLAUBER "
            "RABELO MATIAS 3) contra o despejo de 35. O teste passa SE E "
            "SOMENTE SE os dois vereditos forem iguais — não testa correção da "
            "regra, fixa em código que a métrica é CEGA À UTILIDADE."
        ),
        "base": "ELISA GUARANA DE CASTRO, EDSON MIAGUSKO e GLAUBER RABELO MATIAS: DEPARTAMENTO DE CIÊNCIAS SOCIAIS",
    },
]
