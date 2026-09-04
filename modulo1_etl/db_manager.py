import sqlite3
import unicodedata
import json
import os
import logging

log = logging.getLogger(__name__)

# O banco de dados ficará na raiz do projeto (ou na pasta que definir)
DB_PATH = os.getenv("DB_PATH", "sigaa.db")

def init_db():
    """
    Inicializa o SQLite criando a tabela genérica schema-less.
    A coluna 'dados_brutos' armazena o JSON completo extraído.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS entidades_sigaa (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                tipo_entidade TEXT NOT NULL,
                dados_brutos TEXT NOT NULL
            )
        ''')
        # Cria um índice na coluna tipo_entidade para agilizar as buscas futuras
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_tipo ON entidades_sigaa(tipo_entidade)')
        conn.commit()

def salvar_entidades(tipo_entidade: str, lista_de_dicionarios: list[dict], substituir: bool = False):
    """
    Recebe uma lista de dicionários Python e guarda-os como JSON no SQLite.
    Qualquer scraper futuro poderá usar esta função genérica.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        if substituir:
            # Uma execução completa do ETL é um retrato COMPLETO daquele tipo de
            # entidade, não um incremento. Sem isto a recarga somava-se ao que já
            # havia: o Chroma era limpo (limpar_antes=True em parte5_carga) mas o
            # SQLite não, então a busca estruturada passaria a contar os docentes
            # duas vezes — e é justamente ela que responde "quantos docentes tem o
            # departamento X". Erro plausível, silencioso e do tamanho do corpus.
            cursor.execute("DELETE FROM entidades_sigaa WHERE tipo_entidade = ?", (tipo_entidade,))
            log.info(f"[SQLITE] {cursor.rowcount} registos antigos do tipo '{tipo_entidade}' removidos antes da recarga.")
        for item in lista_de_dicionarios:
            cursor.execute(
                "INSERT INTO entidades_sigaa (tipo_entidade, dados_brutos) VALUES (?, ?)",
                (tipo_entidade, json.dumps(item, ensure_ascii=False))
            )
        conn.commit()
    log.info(f"[SQLITE] Guardados {len(lista_de_dicionarios)} registos do tipo '{tipo_entidade}'.")

def normalizar(texto) -> str:
    """
    Caixa e acentos fora, para comparar 'Ciência da Computação' com
    'CIÊNCIA DA COMPUTAÇÃO'.

    Existe porque o LIKE do SQLite é case-insensitive apenas para ASCII: medido
    em Set/2026, LIKE '%Ciência da Computação%' devolvia 0 resultados enquanto
    '%CIÊNCIA DA COMPUTAÇÃO%' devolvia 6. Como o SIGAA grava departamento em
    CAIXA ALTA e o LLM escreve o argumento da tool em caixa mista com acento,
    a tool estruturada falhava em quase toda pergunta com acento — e o agente
    respondia, honestamente, que não havia docentes no departamento.
    """
    if texto is None:
        return ""
    decomposto = unicodedata.normalize("NFKD", str(texto))
    return "".join(c for c in decomposto if not unicodedata.combining(c)).casefold()


def buscar_entidades_por_campo(tipo_entidade: str, campo_json: str, valor_busca: str) -> list[dict]:
    """
    Busca entidades de forma tolerante dentro do JSON armazenado.
    Ex: buscar_entidades_por_campo('docente', 'departamento', 'Computação')

    A comparação é por substring, ignorando caixa e acentos. O filtro por
    tipo_entidade continua no SQL (é indexado); o casamento do campo é feito em
    Python porque o SQLite não normaliza acentos. Com a ordem de grandeza deste
    banco (centenas de linhas) isso é irrelevante em custo, e é correto —
    o que a versão em SQL puro não era.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT dados_brutos FROM entidades_sigaa WHERE tipo_entidade = ?",
            (tipo_entidade,),
        )
        linhas = cursor.fetchall()

    alvo = normalizar(valor_busca)
    return [
        registro
        for registro in (json.loads(linha[0]) for linha in linhas)
        if alvo in normalizar(registro.get(campo_json))
    ]