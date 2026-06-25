import sqlite3
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

def salvar_entidades(tipo_entidade: str, lista_de_dicionarios: list[dict]):
    """
    Recebe uma lista de dicionários Python e guarda-os como JSON no SQLite.
    Qualquer scraper futuro poderá usar esta função genérica.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        for item in lista_de_dicionarios:
            cursor.execute(
                "INSERT INTO entidades_sigaa (tipo_entidade, dados_brutos) VALUES (?, ?)",
                (tipo_entidade, json.dumps(item, ensure_ascii=False))
            )
        conn.commit()
    log.info(f"[SQLITE] Guardados {len(lista_de_dicionarios)} registos do tipo '{tipo_entidade}'.")

def buscar_entidades_por_campo(tipo_entidade: str, campo_json: str, valor_busca: str) -> list[dict]:
    """
    Busca entidades de forma tolerante dentro do JSON armazenado.
    Utiliza a função json_extract nativa do SQLite.
    Ex: buscar_entidades_por_campo('docente', 'departamento', 'Computação')
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        # O LIKE '%valor%' garante que nomes parciais (ex: 'dcc' -> 'Ciência da Computação') possam ser encontrados se o LLM normalizar minimamente.
        query = f"""
            SELECT dados_brutos FROM entidades_sigaa
            WHERE tipo_entidade = ? AND json_extract(dados_brutos, '$.' || ?) LIKE ?
        """
        cursor.execute(query, (tipo_entidade, campo_json, f"%{valor_busca}%"))
        resultados = cursor.fetchall()
        
        return [json.loads(row[0]) for row in resultados]