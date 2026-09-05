"""
Ponte para a convenção de imports mista do projeto.

`modulo1_etl/` usa imports planos entre seus arquivos (`from db_manager import
...`). No container isso é resolvido pelo `ENV PYTHONPATH=/app:/app/modulo1_etl`
do Dockerfile; aqui a pasta é acrescentada de novo para o pytest funcionar
também fora dele. Ver "Convenção de imports mista" no CLAUDE.md.
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
for caminho in (RAIZ, RAIZ / "modulo1_etl"):
    if str(caminho) not in sys.path:
        sys.path.insert(0, str(caminho))
