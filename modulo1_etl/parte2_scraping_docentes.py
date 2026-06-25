# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 2: scraping de docentes (Atualizado para SQLite)

import json
import re
import asyncio
import time
import logging
import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path
from haystack import Document

# IMPORTAÇÃO DO NOSSO NOVO GESTOR DE DADOS MODULAR
from db_manager import salvar_entidades

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/scraping_docentes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)
log = logging.info

BASE_URL  = "https://sigaa.ufrrj.br"
INSTANCIA = "sigaa"
URL_BUSCA_DOCENTES = f"{BASE_URL}/sigaa/public/docente/busca_docentes.jsf?aba=p-academico"

HEADERS = {
    "User-Agent":      "UFRRJ-IC-RAG/1.0 (Iniciacao Cientifica - pesquisa academica)",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Content-Type":    "application/x-www-form-urlencoded",
}

DELAY_ENTRE_REQUISICOES   = 2
DELAY_ENTRE_DEPARTAMENTOS = 3
DELAY_NIVEL3              = 1
MAX_WORKERS               = 8
MIN_CHARS       = 20
PADRAO_SIAPE    = re.compile(r"/docente/portal\.jsf\?siape=(\d+)")
NOMES_INVALIDOS = {"Sistema Integrado de Gestão de Atividades Acadêmicas"}

# --- AS FUNÇÕES DE SCRAPING PERMANECEM INTACTAS ---
def criar_cliente_sync() -> httpx.Client:
    return httpx.Client(headers=HEADERS, timeout=30, follow_redirects=True, cookies=httpx.Cookies())

def acessar_pagina_sync(cliente: httpx.Client, url: str) -> str | None:
    try:
        r = cliente.get(url)
        r.raise_for_status()
        try:
            return r.content.decode("iso-8859-1")
        except UnicodeDecodeError:
            return r.content.decode("utf-8", errors="replace")
    except Exception as e:
        log(f"  [ERRO] Acesso: {url} | {e}")
        return None

def extrair_campos_formulario(soup: BeautifulSoup) -> dict:
    return {inp.get("name"): inp.get("value", "") for inp in soup.find("form").find_all("input", type="hidden") if inp.get("name")} if soup.find("form") else {}

def extrair_info_formulario(soup: BeautifulSoup) -> dict:
    form = soup.find("form")
    if not form: return {}
    form_id = form.get("id", "form")
    form_action = form.get("action", URL_BUSCA_DOCENTES)
    url_post = f"{BASE_URL}{form_action}" if form_action.startswith("/") else (form_action if form_action.startswith("http") else URL_BUSCA_DOCENTES)
    select = form.find("select")
    btn = form.find("input", {"type": "submit"}) or form.find("button")
    return {
        "url_post": url_post,
        "nome_sel": select.get("name", f"{form_id}:departamento") if select else f"{form_id}:departamento",
        "nome_btn": btn.get("name", f"{form_id}:btnBuscar") if btn else f"{form_id}:btnBuscar",
        "valor_btn": btn.get("value", "Buscar") if btn else "Buscar",
    }

def extrair_departamentos(cliente: httpx.Client) -> list[dict]:
    log("[NÍVEL 1] Descobrindo departamentos...")
    html = acessar_pagina_sync(cliente, URL_BUSCA_DOCENTES)
    if not html: return []
    select = BeautifulSoup(html, "lxml").find("select")
    if not select: return []
    deptos = [{"id": o.get("value", "").strip(), "nome": o.get_text(strip=True)} for o in select.find_all("option") if o.get("value", "").strip() not in ("", "0") and "--" not in o.get_text(strip=True)]
    log(f"[NÍVEL 1] {len(deptos)} departamentos encontrados.")
    return deptos

def extrair_siapes_via_post(cliente: httpx.Client, id_departamento: str, nome_departamento: str) -> list[int]:
    html_get = acessar_pagina_sync(cliente, URL_BUSCA_DOCENTES)
    if not html_get: return []
    time.sleep(DELAY_ENTRE_REQUISICOES)
    soup_get = BeautifulSoup(html_get, "lxml")
    campos, info_form = extrair_campos_formulario(soup_get), extrair_info_formulario(soup_get)
    if "javax.faces.ViewState" not in campos: return []
    payload = {**campos, info_form["nome_sel"]: id_departamento, info_form["nome_btn"]: info_form["valor_btn"]}
    try:
        resp = cliente.post(info_form["url_post"], data=payload)
        resp.raise_for_status()
        html_post = resp.content.decode("iso-8859-1") if "iso-8859-1" in resp.headers.get("content-type", "").lower() else resp.content.decode("utf-8", errors="replace")
    except Exception:
        return []
    return sorted(set(int(PADRAO_SIAPE.search(a["href"]).group(1)) for a in BeautifulSoup(html_post, "lxml").find_all("a", href=True) if PADRAO_SIAPE.search(a["href"])))

def _montar_conteudo_docente(soup: BeautifulSoup) -> tuple[str, str, str]:
    h3_tags = soup.find_all("h3")
    nome = h3_tags[0].get_text(strip=True) if len(h3_tags) >= 1 else "Não informado"
    departamento = h3_tags[1].get_text(strip=True) if len(h3_tags) >= 2 else "Não informado"
    campos = {dt.get_text(strip=True).rstrip(":").lower(): dd.get_text(separator=" ", strip=True) for dl in soup.find_all("dl") for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd"))}
    partes = [f"Docente: {nome}.", f"Departamento: {departamento}."]
    for key, label in [("descrição pessoal", "Perfil"), ("formação acadêmica/profissional (onde obteve os títulos, atuação profissional, etc.)", "Formação"), ("áreas de interesse (áreas de interesse de ensino e pesquisa)", "Áreas de interesse"), ("endereço profissional", "Endereço"), ("telefone/ramal", "Telefone")]:
        val = campos.get(key, campos.get("formação acadêmica/profissional", "") if "formação" in key else "")
        if val and len(val) > 3: partes.append(f"{label}: {val}")
    return nome, departamento, " ".join(partes)

async def extrair_perfil_docente_async(cliente_async: httpx.AsyncClient, semaforo: asyncio.Semaphore, siape: int, timestamp: str) -> Document | None:
    url = f"{BASE_URL}/sigaa/public/docente/portal.jsf?siape={siape}"
    async with semaforo:
        await asyncio.sleep(DELAY_NIVEL3)
        try:
            r = await cliente_async.get(url)
            r.raise_for_status()
            html = r.content.decode("iso-8859-1")
        except Exception:
            return None
    nome, departamento, conteudo = _montar_conteudo_docente(BeautifulSoup(html, "lxml"))
    if len(conteudo) < MIN_CHARS or nome in NOMES_INVALIDOS: return None
    log(f"  ✓ siape={siape} — {nome}")
    return Document(content=conteudo, meta={"instancia_dona": INSTANCIA, "content_type": "docente_perfil", "source_url": url, "scraped_at": timestamp, "nome_docente": nome, "departamento": departamento, "siape": str(siape)})

async def coletar_perfis_async(siapes: list[int], timestamp: str) -> list[Document]:
    semaforo, headers_async = asyncio.Semaphore(MAX_WORKERS), {k: v for k, v in HEADERS.items() if k != "Content-Type"}
    async with httpx.AsyncClient(headers=headers_async, timeout=30, follow_redirects=True) as cliente_async:
        resultados = await asyncio.gather(*(extrair_perfil_docente_async(cliente_async, semaforo, siape, timestamp) for siape in siapes), return_exceptions=True)
    return [r for r in resultados if isinstance(r, Document)]

# --- FIM DAS FUNÇÕES INTACTAS ---

def processar_e_salvar_estruturado(documentos: list[Document]):
    """
    Substitui a antiga geração de JSON.
    Extrai apenas os dados estruturados e delega a gravação ao db_manager.
    """
    dados_para_sqlite = []
    
    for doc in documentos:
        depto = doc.meta.get("departamento")
        nome = doc.meta.get("nome_docente")
        siape = doc.meta.get("siape")
        
        if depto and nome:
            dados_para_sqlite.append({
                "nome": nome,
                "departamento": depto,
                "siape": siape
            })

    # Chama o módulo central para gravar os dicionários
    salvar_entidades(tipo_entidade="docente", lista_de_dicionarios=dados_para_sqlite)

def scrape_docentes() -> list[Document]:
    timestamp = datetime.now(timezone.utc).isoformat()
    siapes_vistos: set[int] = set()

    with criar_cliente_sync() as cliente:
        departamentos = extrair_departamentos(cliente)
        if not departamentos: return []
        for i, depto in enumerate(departamentos, 1):
            log(f"\n[NÍVEL 2] ({i}/{len(departamentos)}) {depto['nome']}")
            novos = [s for s in extrair_siapes_via_post(cliente, depto["id"], depto["nome"]) if s not in siapes_vistos]
            siapes_vistos.update(novos)
            time.sleep(DELAY_ENTRE_DEPARTAMENTOS)

    log(f"\n[NÍVEL 3] {len(siapes_vistos)} perfis a coletar...")
    documentos = asyncio.run(coletar_perfis_async(sorted(siapes_vistos), timestamp))
    
    # Nova chamada Desacoplada para o SQLite
    processar_e_salvar_estruturado(documentos)

    return documentos

if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 2 — SCRAPING DE DOCENTES (Modo SQLite)")
    log("=" * 60)
    docs_docentes = scrape_docentes()
    if not docs_docentes: exit(1)
    log("[PARTE 2 CONCLUÍDA]")  