# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 2: scraping de docentes + chunking (v3 — async paralelo)
# Autor: Raul Nascimento

import re
import asyncio
import time
import logging
import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from pathlib import Path
from haystack import Document
from haystack.components.preprocessors import DocumentSplitter


Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(
            f"logs/scraping_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
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
MAX_WORKERS               = 8   # reduzir para 4 se o servidor retornar 429/503

MIN_CHARS       = 20
CHUNK_SENTENCES = 5
CHUNK_OVERLAP   = 1

PADRAO_SIAPE    = re.compile(r"/docente/portal\.jsf\?siape=(\d+)")
NOMES_INVALIDOS = {"Sistema Integrado de Gestão de Atividades Acadêmicas"}


def criar_cliente_sync() -> httpx.Client:
    """Cookie jar persistente — obrigatório para manter JSESSIONID no fluxo JSF."""
    return httpx.Client(
        headers=HEADERS,
        timeout=30,
        follow_redirects=True,
        cookies=httpx.Cookies(),
    )


def acessar_pagina_sync(cliente: httpx.Client, url: str) -> str | None:
    try:
        r = cliente.get(url)
        r.raise_for_status()
        try:
            return r.content.decode("iso-8859-1")
        except UnicodeDecodeError:
            return r.content.decode("utf-8", errors="replace")
    except httpx.TimeoutException:
        log(f"  [ERRO] Timeout: {url}")
        return None
    except httpx.HTTPStatusError as e:
        log(f"  [ERRO] HTTP {e.response.status_code}: {url}")
        return None
    except httpx.RequestError as e:
        log(f"  [ERRO] Rede: {e}")
        return None


def extrair_campos_formulario(soup: BeautifulSoup) -> dict:
    campos = {}
    form = soup.find("form")
    if not form:
        return campos
    for inp in form.find_all("input", type="hidden"):
        nome = inp.get("name")
        if nome:
            campos[nome] = inp.get("value", "")
    return campos


def extrair_info_formulario(soup: BeautifulSoup) -> dict:
    form = soup.find("form")
    if not form:
        return {}

    form_id     = form.get("id", "form")
    form_action = form.get("action", URL_BUSCA_DOCENTES)

    if form_action.startswith("/"):
        url_post = f"{BASE_URL}{form_action}"
    elif form_action.startswith("http"):
        url_post = form_action
    else:
        url_post = URL_BUSCA_DOCENTES

    select   = form.find("select")
    nome_sel = select.get("name", f"{form_id}:departamento") if select else f"{form_id}:departamento"

    btn = form.find("input", {"type": "submit"}) or form.find("button")
    nome_btn  = btn.get("name",  f"{form_id}:btnBuscar") if btn else f"{form_id}:btnBuscar"
    valor_btn = btn.get("value", "Buscar")               if btn else "Buscar"

    return {
        "url_post":  url_post,
        "nome_sel":  nome_sel,
        "nome_btn":  nome_btn,
        "valor_btn": valor_btn,
    }


def extrair_departamentos(cliente: httpx.Client) -> list[dict]:
    """Extrai todos os departamentos do <select> dinamicamente — sem IDs hardcoded."""
    log("[NÍVEL 1] Descobrindo departamentos...")
    html = acessar_pagina_sync(cliente, URL_BUSCA_DOCENTES)
    if not html:
        log("[NÍVEL 1] FALHA: página de busca inacessível.")
        return []

    soup   = BeautifulSoup(html, "lxml")
    select = soup.find("select")

    if not select:
        log("[NÍVEL 1] FALHA: <select> não encontrado.")
        return []

    departamentos = []
    for option in select.find_all("option"):
        id_depto   = option.get("value", "").strip()
        nome_depto = option.get_text(strip=True)
        if not id_depto or id_depto == "0" or "--" in nome_depto:
            continue
        departamentos.append({"id": id_depto, "nome": nome_depto})

    log(f"[NÍVEL 1] {len(departamentos)} departamentos encontrados.")
    return departamentos


def extrair_siapes_via_post(
    cliente: httpx.Client,
    id_departamento: str,
    nome_departamento: str,
) -> list[int]:
    """
    POST JSF para listar docentes de um departamento.
    GET antes de cada POST para obter ViewState fresco.
    """
    html_get = acessar_pagina_sync(cliente, URL_BUSCA_DOCENTES)
    if not html_get:
        return []

    soup_get  = BeautifulSoup(html_get, "lxml")
    campos    = extrair_campos_formulario(soup_get)
    info_form = extrair_info_formulario(soup_get)

    if "javax.faces.ViewState" not in campos:
        log(f"  [ERRO] ViewState ausente — {nome_departamento}")
        return []

    payload = {
        **campos,
        info_form["nome_sel"]: id_departamento,
        info_form["nome_btn"]: info_form["valor_btn"],
    }

    time.sleep(DELAY_ENTRE_REQUISICOES)

    try:
        resp = cliente.post(info_form["url_post"], data=payload)
        resp.raise_for_status()
        try:
            html_post = resp.content.decode("iso-8859-1")
        except UnicodeDecodeError:
            html_post = resp.content.decode("utf-8", errors="replace")
    except httpx.HTTPError as e:
        log(f"  [ERRO] POST falhou: {e}")
        return []

    soup_post = BeautifulSoup(html_post, "lxml")
    siapes    = set()

    for tag_a in soup_post.find_all("a", href=True):
        match = PADRAO_SIAPE.search(tag_a["href"])
        if match:
            siapes.add(int(match.group(1)))

    return sorted(siapes)


def _montar_conteudo_docente(soup: BeautifulSoup) -> tuple[str, str, str]:
    """Parseia HTML do portal de um docente. Separado do I/O para facilitar testes."""
    nome         = "Não informado"
    departamento = "Não informado"
    h3_tags      = soup.find_all("h3")
    if len(h3_tags) >= 1:
        nome = h3_tags[0].get_text(strip=True)
    if len(h3_tags) >= 2:
        departamento = h3_tags[1].get_text(strip=True)

    campos = {}
    for dl in soup.find_all("dl"):
        for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd")):
            chave = dt.get_text(strip=True).rstrip(":").lower()
            valor = dd.get_text(separator=" ", strip=True)
            if valor and len(valor) > 3:
                campos[chave] = valor

    descricao = campos.get("descrição pessoal", "")
    formacao  = campos.get(
        "formação acadêmica/profissional (onde obteve os títulos, atuação profissional, etc.)",
        campos.get("formação acadêmica/profissional", "")
    )
    areas    = campos.get("áreas de interesse (áreas de interesse de ensino e pesquisa)", "")
    endereco = campos.get("endereço profissional", "")
    telefone = campos.get("telefone/ramal", "")

    partes = [f"Docente: {nome}.", f"Departamento: {departamento}."]
    if descricao:
        partes.append(f"Perfil: {descricao}")
    if formacao:
        partes.append(f"Formação: {formacao}")
    if areas:
        partes.append(f"Áreas de interesse: {areas}")
    if endereco:
        partes.append(f"Endereço profissional: {endereco}.")
    if telefone:
        partes.append(f"Telefone/ramal: {telefone}.")

    return nome, departamento, " ".join(partes)


async def extrair_perfil_docente_async(
    cliente_async: httpx.AsyncClient,
    semaforo: asyncio.Semaphore,
    siape: int,
    timestamp: str,
) -> Document | None:
    """Busca e parseia o perfil de um docente. Semáforo limita a MAX_WORKERS requisições."""
    url = f"{BASE_URL}/sigaa/public/docente/portal.jsf?siape={siape}"

    async with semaforo:
        await asyncio.sleep(DELAY_NIVEL3)
        try:
            r = await cliente_async.get(url)
            r.raise_for_status()
            try:
                html = r.content.decode("iso-8859-1")
            except UnicodeDecodeError:
                html = r.content.decode("utf-8", errors="replace")
        except httpx.TimeoutException:
            log(f"  [ERRO] Timeout: siape={siape}")
            return None
        except httpx.HTTPStatusError as e:
            log(f"  [ERRO] HTTP {e.response.status_code}: siape={siape}")
            return None
        except httpx.RequestError as e:
            log(f"  [ERRO] Rede siape={siape}: {e}")
            return None

    soup = BeautifulSoup(html, "lxml")
    nome, departamento, conteudo = _montar_conteudo_docente(soup)

    if len(conteudo) < MIN_CHARS or nome in NOMES_INVALIDOS:
        return None

    log(f"  ✓ siape={siape} — {nome} ({len(conteudo)} chars)")

    return Document(
        content=conteudo,
        meta={
            "instancia_dona": INSTANCIA,
            "content_type":   "docente_perfil",
            "source_url":     url,
            "scraped_at":     timestamp,
            "nome_docente":   nome,
            "departamento":   departamento,
            "siape":          str(siape),
        }
    )


async def coletar_perfis_async(siapes: list[int], timestamp: str) -> list[Document]:
    """Coleta todos os perfis em paralelo. Deduplica por hash de conteúdo."""
    semaforo      = asyncio.Semaphore(MAX_WORKERS)
    headers_async = {k: v for k, v in HEADERS.items() if k != "Content-Type"}

    async with httpx.AsyncClient(
        headers=headers_async,
        timeout=30,
        follow_redirects=True,
    ) as cliente_async:
        tarefas    = [
            extrair_perfil_docente_async(cliente_async, semaforo, siape, timestamp)
            for siape in siapes
        ]
        resultados = await asyncio.gather(*tarefas, return_exceptions=True)

    documentos       = []
    conteudos_vistos = set()
    ignorados        = 0

    for r in resultados:
        if isinstance(r, Exception):
            ignorados += 1
        elif r is None:
            ignorados += 1
        else:
            hash_conteudo = hash(r.content)
            if hash_conteudo in conteudos_vistos:
                ignorados += 1
            else:
                conteudos_vistos.add(hash_conteudo)
                documentos.append(r)

    if ignorados:
        log(f"  [INFO] {ignorados} siape(s) ignorados (vazio, erro ou duplicata).")

    return documentos


def scrape_docentes() -> list[Document]:
    """
    Orquestra os 3 níveis de coleta.
    Níveis 1 e 2 síncronos (JSF), Nível 3 assíncrono e paralelo.
    """
    timestamp     = datetime.now(timezone.utc).isoformat()
    siapes_vistos: set[int] = set()

    with criar_cliente_sync() as cliente:
        departamentos = extrair_departamentos(cliente)
        if not departamentos:
            log("[SCRAPER] Nenhum departamento encontrado. Abortando.")
            return []

        total = len(departamentos)

        for i, depto in enumerate(departamentos, 1):
            log(f"\n[NÍVEL 2] ({i}/{total}) {depto['nome']}")
            siapes = extrair_siapes_via_post(cliente, depto["id"], depto["nome"])
            novos  = [s for s in siapes if s not in siapes_vistos]
            siapes_vistos.update(novos)
            log(f"  → {len(siapes)} docente(s) | {len(novos)} novo(s).")
            time.sleep(DELAY_ENTRE_DEPARTAMENTOS)

    todos_siapes = sorted(siapes_vistos)
    log(f"\n[NÍVEL 3] {len(todos_siapes)} perfis a coletar (workers={MAX_WORKERS})...")
    documentos = asyncio.run(coletar_perfis_async(todos_siapes, timestamp))
    log(f"\n[SCRAPER] {len(documentos)} perfis extraídos.")
    return documentos


def chunkar_documentos(documentos: list[Document]) -> list[Document]:
    """Divide documentos em chunks de CHUNK_SENTENCES sentenças com overlap."""
    if not documentos:
        log("[CHUNKING] Nenhum documento recebido.")
        return []

    log(f"\n[CHUNKING] {len(documentos)} documentos...")

    import warnings
    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=CHUNK_SENTENCES,
        split_overlap=CHUNK_OVERLAP,
        language="pt",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        splitter.warm_up()

    resultado = splitter.run(documents=documentos)
    chunks    = resultado["documents"]

    log(f"[CHUNKING] {len(documentos)} docs → {len(chunks)} chunks.")
    return chunks


def validar_documentos(documentos: list[Document]) -> bool:
    """Verifica presença dos campos de governança obrigatórios."""
    if not documentos:
        log("[VALIDAÇÃO] ✗ Nenhum documento.")
        return False

    erros = 0
    for campo in ["instancia_dona", "source_url", "scraped_at"]:
        faltando = [i for i, d in enumerate(documentos) if campo not in d.meta]
        if faltando:
            log(f"[VALIDAÇÃO] ✗ '{campo}' ausente em {len(faltando)} docs.")
            erros += 1
        else:
            log(f"[VALIDAÇÃO] ✓ '{campo}' OK.")

    errados = [d for d in documentos if d.meta.get("instancia_dona") != INSTANCIA]
    if errados:
        log(f"[VALIDAÇÃO] ✗ {len(errados)} docs com instancia_dona incorreto.")
        erros += 1
    else:
        log(f"[VALIDAÇÃO] ✓ instancia_dona = '{INSTANCIA}'.")

    return erros == 0


if __name__ == "__main__":
    log("=" * 60)
    log("PARTE 2 — SCRAPING DE DOCENTES + CHUNKING")
    log("=" * 60)

    docs_docentes = scrape_docentes()

    if not docs_docentes:
        log("[RESULTADO] Nenhum docente coletado.")
        exit(1)

    validar_documentos(docs_docentes)

    log("\n--- Amostra (primeiros 3 docentes) ---")
    for i, doc in enumerate(docs_docentes[:3]):
        log(f"  [{i+1}] {doc.meta.get('nome_docente')} — {doc.content[:100]}...")

    # Integração com Parte 1:
    #   from parte1_scraping import scrape_sigaa
    #   chunks = chunkar_documentos(scrape_sigaa() + docs_docentes)
    chunks = chunkar_documentos(docs_docentes)
    validar_documentos(chunks)

    log(f"\n[RESUMO] Departamentos: {len(set(d.meta['departamento'] for d in docs_docentes))} | "
        f"Docentes: {len(docs_docentes)} | Chunks: {len(chunks)}")
    log("[PARTE 2 CONCLUÍDA]")