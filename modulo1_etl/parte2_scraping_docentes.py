# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 2: scraping de docentes (Atualizado para SQLite)

import unicodedata
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

# Cria um logger específico para este arquivo
logger = logging.getLogger("scraping_docentes")
logger.setLevel(logging.INFO)
logger.propagate = False # Impede que o log vaze para a Parte 5

# Cria os manipuladores (onde o log vai ser salvo/mostrado)
arquivo_handler = logging.FileHandler(
    f"logs/scraping_docentes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log", 
    encoding="utf-8"
)
console_handler = logging.StreamHandler()

# Define o padrão de texto do log
formatador = logging.Formatter("%(asctime)s  %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
arquivo_handler.setFormatter(formatador)
console_handler.setFormatter(formatador)

# Conecta tudo ao logger
logger.addHandler(arquivo_handler)
logger.addHandler(console_handler)

# Mantém a sua variável 'log' funcionando perfeitamente no resto do código
log = logger.info

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

def extrair_docentes_via_post(cliente: httpx.Client, id_departamento: str, nome_departamento: str) -> list[tuple[int, str]]:
    """Devolve os pares (siape, nome) que a listagem do departamento anuncia."""
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
    soup_post = BeautifulSoup(html_post, "lxml")

    # ACHADO 08: a listagem devolve o PAR (id, nome), não só o id. O nome daqui
    # é a única testemunha independente de quem aquele id deveria ser — é ele
    # que permite flagrar, no perfil, uma página servida trocada. Estrutura
    # verificada em 4 set 2026:
    #   <tr><td><span class="nome">FULANO</span>
    #           <span class="departamento">...</span>
    #           <span class="pagina"><a href="...portal.jsf?siape=NNN">…</a>
    docentes: list[tuple[int, str]] = []
    sem_nome = 0
    for a in soup_post.find_all("a", href=True):
        casamento = PADRAO_SIAPE.search(a["href"])
        if not casamento:
            continue
        linha = a.find_parent("tr")
        span_nome = linha.find("span", class_="nome") if linha else None
        nome = span_nome.get_text(strip=True) if span_nome else ""
        if not nome:
            sem_nome += 1
        docentes.append((int(casamento.group(1)), nome))

    # Se o SIGAA mudar o HTML da listagem, o nome some e a verificação de
    # identidade vira no-op silencioso — que é como o achado 08 sobreviveu
    # tanto tempo. Falar alto é barato.
    if sem_nome:
        log(f"  [AVISO] {sem_nome} de {len(docentes)} linhas da listagem sem <span class='nome'> — "
            f"verificação de identidade fica cega nelas (mudou o HTML do SIGAA?)")

    return sorted(set(docentes))

# Campos do perfil, casados por PREFIXO NORMALIZADO — não por igualdade.
#
# ACHADO 01 (Set/2026): a versão anterior procurava a chave literal
# "áreas de interesse (áreas de interesse de ensino e pesquisa)", com espaço
# antes do parêntese, e nunca casava — 0 de 704 perfis, num sistema cuja
# pergunta central é "quem pesquisa o quê".
#
# A causa não é o SIGAA omitir o espaço. Verificado no HTML real (4 set 2026),
# este é o ÚNICO <dt> do perfil que tem uma tag aninhada dentro dele:
#
#   <dt> Áreas de Interesse <span class="info">(áreas de interesse de ensino
#   e pesquisa) </span></dt>
#
# (no original ainda há quebras de linha e tabulações entre os nós de texto,
# omitidas acima por legibilidade)
#
# O espaço existe na página. O que acontece é que `get_text(strip=True)`
# faz strip de cada nó de texto e junta os
# pedaços SEM separador: "Áreas de Interesse" + "(áreas...)" vira
# "áreas de interesse(áreas..." — o espaço morre no parser, não na origem.
# Registrar isso importa: quem conferir o código-fonte da página vai ver o
# espaço lá e concluir, errado, que o diagnóstico estava furado.
#
# Casar por prefixo resolve as duas variantes e ainda sobrevive a mudança no
# texto de ajuda entre parênteses, que é rótulo de interface. É a mesma técnica
# que _motivo_de_rejeicao() logo abaixo já usava — a extração é que estava mais
# frágil que a validação ao lado dela.
CAMPOS_DO_PERFIL = [
    ("descrição pessoal", "Perfil"),
    ("formação acadêmica", "Formação"),
    ("áreas de interesse", "Áreas de interesse"),
    # ACHADO 05, os "campos grátis": já vêm no mesmo <dl> da mesma requisição
    # que já fazemos, então custam zero acesso extra ao servidor da UFRRJ.
    # Medidos na auditoria de 4 set 2026: Lattes em 31 de 40 perfis, e-mail em
    # 38 de 40, sala em 4 de 40. O Lattes é especialmente valioso — é o
    # identificador externo estável que a siape não é (achado 08).
    ("currículo lattes", "Currículo Lattes"),
    ("endereço profissional", "Endereço"),
    ("sala", "Sala"),
    ("telefone", "Telefone"),
    ("endereço eletrônico", "E-mail"),
]


def _normalizar_chave(texto: str) -> str:
    """Minúscula, sem acento e sem espaço supérfluo, para casar rótulo do SIGAA.

    Sem acento porque "á" pode chegar como U+00E1 ou como "a"+U+0301 conforme a
    normalização Unicode da página — idênticos na tela, diferentes para o `==`.
    Mesma razão de normalizar() em db_manager.py.
    """
    decomposto = unicodedata.normalize("NFKD", texto)
    sem_acento = "".join(c for c in decomposto if not unicodedata.combining(c))
    return " ".join(sem_acento.lower().split())


# O SIGAA não deixa campo vazio: grava "não informada" como VALOR. Um perfil
# sem nada preenchido virava, no texto indexado, "Perfil: não informada Formação:
# não informada Endereço: não informado" — e nós vetorizávamos isso. Medido em
# 4 set 2026: 36% dos campos do corpus eram esse marcador. Como o texto é
# idêntico em centenas de perfis, ele funciona como ímã genérico na busca
# semântica: uma pergunta sem relação nenhuma com o corpus recupera justamente
# os perfis mais vazios, porque são os que mais se parecem entre si. É a causa
# direta do achado 03.
PLACEHOLDERS_SIGAA = ("nao informad", "nao ha ", "nao possui", "nenhum")


def _e_placeholder(valor: str) -> bool:
    """True quando o SIGAA gravou um marcador de vazio em vez de deixar em branco."""
    normalizado = _normalizar_chave(valor)
    if not normalizado or normalizado in {"-", "--", "n/a", "na", "nao informada", "nao informado"}:
        return True
    return normalizado.startswith(PLACEHOLDERS_SIGAA)


def _valor_por_prefixo(campos: dict[str, str], prefixo: str) -> str:
    alvo = _normalizar_chave(prefixo)
    for chave, valor in campos.items():
        if _normalizar_chave(chave).startswith(alvo):
            return valor
    return ""


def _montar_conteudo_docente(soup: BeautifulSoup) -> tuple[str, str, str]:
    h3_tags = soup.find_all("h3")
    nome = h3_tags[0].get_text(strip=True) if len(h3_tags) >= 1 else "Não informado"
    departamento = h3_tags[1].get_text(strip=True) if len(h3_tags) >= 2 else "Não informado"
    campos = {dt.get_text(strip=True).rstrip(":").lower(): dd.get_text(separator=" ", strip=True) for dl in soup.find_all("dl") for dt, dd in zip(dl.find_all("dt"), dl.find_all("dd"))}
    partes = [f"Docente: {nome}.", f"Departamento: {departamento}."]
    for prefixo, label in CAMPOS_DO_PERFIL:
        val = _valor_por_prefixo(campos, prefixo)
        # O corte por len(val) > 3 saiu junto: descartava ramal curto legítimo
        # (medido: 7 de 40 perfis com telefone de 2 a 3 dígitos). _e_placeholder
        # cobre vazio, "-" e "n/a", que era o que aquele corte queria pegar.
        if val.strip() and not _e_placeholder(val):
            partes.append(f"{label}: {val}")
    return nome, departamento, " ".join(partes)

MOTIVO_IDENTIDADE = "pagina de outro docente (identidade nao confere)"


def _nomes_batem(esperado: str, obtido: str) -> bool:
    """
    O nome do perfil tem que ser o nome que a listagem prometeu para aquele id.

    Comparação normalizada (sem acento, sem caixa, sem espaço duplo) porque a
    listagem e o <h3> do perfil vêm de templates diferentes do SIGAA. Sem nome
    esperado — listagem sem <span class="nome"> — não há o que verificar, e
    devolver True aqui é deliberado: a ausência já foi denunciada alto na
    listagem, e reprovar todo mundo por isso seria pior.
    """
    if not esperado:
        return True
    return _normalizar_chave(esperado) == _normalizar_chave(obtido)


# Chaves que só existem numa página de perfil de docente de verdade.
CHAVES_DE_PERFIL = ("descrição pessoal", "formação acadêmica", "áreas de interesse")


def _motivo_de_rejeicao(html: str, soup: BeautifulSoup, siape: int, nome: str, departamento: str, nome_esperado: str = "") -> str | None:
    """
    Devolve o motivo pelo qual a página NÃO é um perfil, ou None se for válida.

    POR QUE ISSO EXISTE: para uma SIAPE inexistente o SIGAA responde HTTP 200
    com a home do portal, não 404. O raise_for_status() passa, e o h3 da home
    rende nome="Docentes" e departamento="Autenticação de Documentos" — ambos
    não-vazios, então checar preenchimento não basta. A guarda antiga
    (MIN_CHARS=20) era inútil no sentido oposto: medido em Set/2026, a home
    tem 31126 caracteres contra 8069 de um perfil real.

    A verificação decisiva é de identidade: a página tem que provar que é
    daquele docente. A home nunca referencia a SIAPE pedida.
    """
    if f"siape={siape}" not in html:
        return "pagina nao referencia a siape pedida (provavel home do portal)"
    if not nome.strip() or not departamento.strip():
        return "nome ou departamento vazios"
    if nome in NOMES_INVALIDOS:
        return "nome na lista de invalidos"
    # ACHADO 09 (4 set 2026): quando a listagem nos deu um nome, ele é a prova
    # de identidade — e ela é ESTRITAMENTE MAIS FORTE que procurar chaves de
    # perfil. Se a listagem disse que este id é do Fulano e o <h3> diz Fulano,
    # é a página do Fulano, tenha ele preenchido a bio ou não.
    #
    # A checagem por CHAVES_DE_PERFIL abaixo custava caro: docente que não
    # preenche a seção descritiva faz o SIGAA omitir aquele <dl> inteiro,
    # sobrando só o bloco de contato — e a pessoa era descartada. Medido nos
    # dois departamentos de computação: 8 de 30 docentes reais recusados assim,
    # entre eles BRUNO JOSE DEMBOGURSKI e LEANDRO GUIMARAES MARQUES ALVIM.
    # Contraria o princípio 1 do projeto (perfil vazio não é falha nossa) e
    # tira da contagem justamente quem só tem nome e departamento — que é o
    # dado de que as perguntas de contagem precisam.
    if nome_esperado:
        if not _nomes_batem(nome_esperado, nome):
            return MOTIVO_IDENTIDADE
        return None

    # Sem nome esperado (a listagem mudou de HTML), volta a valer a guarda
    # antiga: é o melhor sinal que sobra para distinguir um perfil da home.
    chaves = {dt.get_text(strip=True).rstrip(":").lower()
              for dl in soup.find_all("dl") for dt in dl.find_all("dt")}
    if not any(any(k in c for c in chaves) for k in CHAVES_DE_PERFIL):
        return "nenhuma chave de perfil encontrada"
    return None


async def extrair_perfil_docente_async(semaforo: asyncio.Semaphore, siape: int, nome_esperado: str, timestamp: str, headers_async: dict) -> Document | str:
    """Devolve o Document do perfil, ou uma string com o motivo da rejeição."""
    url = f"{BASE_URL}/sigaa/public/docente/portal.jsf?siape={siape}"
    async with semaforo:
        await asyncio.sleep(DELAY_NIVEL3)
        try:
            # UM CLIENTE POR REQUISIÇÃO — ACHADO 08, e é o ponto todo desta função.
            #
            # Antes havia um AsyncClient compartilhado entre as MAX_WORKERS
            # requisições simultâneas. O portal público do SIGAA é JSF e guarda o
            # docente corrente em ESTADO DE SESSÃO NO SERVIDOR, atrelado ao
            # JSESSIONID — que era um só, porque o cliente era um só. As
            # requisições concorrentes disputavam essa sessão: a última a
            # escrever vencia e as outras recebiam a página DE OUTRA PESSOA, com
            # HTTP 200 e HTML impecável.
            #
            # Medido em 4 set 2026, 15 docentes de Ciência da Computação/IM:
            #   cliente compartilhado ....... 6 de 15 com a pessoa errada
            #   um cliente por requisição ... 0 de 15, em duas rodadas
            #
            # Cliente próprio = cookie próprio = sessão própria no servidor. O
            # custo é um handshake por perfil, e latência não é critério aqui.
            async with httpx.AsyncClient(headers=headers_async, timeout=30, follow_redirects=True) as cliente_async:
                r = await cliente_async.get(url)
                r.raise_for_status()
                html = r.content.decode("iso-8859-1")
        except Exception as e:
            return f"erro de rede: {type(e).__name__}"

    soup = BeautifulSoup(html, "lxml")
    nome, departamento, conteudo = _montar_conteudo_docente(soup)

    motivo = _motivo_de_rejeicao(html, soup, siape, nome, departamento, nome_esperado)
    if motivo == MOTIVO_IDENTIDADE:
        log(f"  [IDENTIDADE] siape={siape} — listagem prometeu '{nome_esperado}', perfil trouxe '{nome}'")
    if motivo:
        return motivo

    if len(conteudo) < MIN_CHARS:
        return "conteudo abaixo do minimo"

    log(f"  ✓ siape={siape} — {nome}")
    return Document(content=conteudo, meta={"instancia_dona": INSTANCIA, "content_type": "docente_perfil", "source_url": url, "scraped_at": timestamp, "nome_docente": nome, "departamento": departamento, "siape": str(siape)})

async def coletar_perfis_async(docentes: list[tuple[int, str]], timestamp: str) -> list[Document]:
    """Coleta os perfis dos pares (siape, nome) vindos das listagens."""
    from collections import Counter

    semaforo = asyncio.Semaphore(MAX_WORKERS)
    headers_async = {k: v for k, v in HEADERS.items() if k != "Content-Type"}

    resultados = await asyncio.gather(
        *(extrair_perfil_docente_async(semaforo, siape, nome, timestamp, headers_async) for siape, nome in docentes),
        return_exceptions=True,
    )

    perfis = [r for r in resultados if isinstance(r, Document)]

    # SEGUNDA PASSADA, EM SÉRIE, só para quem falhou na verificação de
    # identidade. Com cliente isolado a corrida não deve mais acontecer, mas se
    # o SIGAA arranjar outra forma de embaralhar sessão, uma refeita em série
    # (semáforo de 1) recupera o perfil em vez de perdê-lo em silêncio. Se ainda
    # assim não bater, aí é rejeição contada e visível.
    a_refazer = [
        docentes[i] for i, r in enumerate(resultados)
        if isinstance(r, str) and r == MOTIVO_IDENTIDADE
    ]
    recuperados = 0
    if a_refazer:
        log(f"[IDENTIDADE] {len(a_refazer)} perfis com identidade divergente — refazendo em série...")
        semaforo_serial = asyncio.Semaphore(1)
        for siape, nome in a_refazer:
            r = await extrair_perfil_docente_async(semaforo_serial, siape, nome, timestamp, headers_async)
            if isinstance(r, Document):
                perfis.append(r)
                recuperados += 1
        log(f"[IDENTIDADE] {recuperados} de {len(a_refazer)} recuperados na segunda passada.")

    # Contabilizar as rejeições, e não só descartá-las: antes um perfil perdido
    # sumia em silêncio, e foi assim que um registro com nome vazio chegou até
    # o vetor store sem ninguém notar.
    motivos = Counter(r for r in resultados if isinstance(r, str))
    if recuperados:
        motivos[MOTIVO_IDENTIDADE] -= recuperados
        if motivos[MOTIVO_IDENTIDADE] <= 0:
            del motivos[MOTIVO_IDENTIDADE]
    if motivos:
        log(f"[VALIDAÇÃO] {sum(motivos.values())} de {len(docentes)} páginas rejeitadas:")
        for motivo, n in motivos.most_common():
            log(f"             {n:>5}  {motivo}")

    log(f"[NÍVEL 3] {len(perfis)} de {len(docentes)} perfis coletados "
        f"({len(perfis) / len(docentes) * 100:.0f}% de captura).")

    return perfis

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
    salvar_entidades(tipo_entidade="docente", lista_de_dicionarios=dados_para_sqlite, substituir=True)

def scrape_docentes() -> list[Document]:
    timestamp = datetime.now(timezone.utc).isoformat()
    # id -> nome prometido pela listagem. Era um set de ids; virou dicionário
    # porque o nome da listagem é a testemunha que valida a identidade do perfil
    # (achado 08). O primeiro departamento que anuncia um id fica com ele.
    docentes_vistos: dict[int, str] = {}

    with criar_cliente_sync() as cliente:
        departamentos = extrair_departamentos(cliente)
        if not departamentos: return []
        for i, depto in enumerate(departamentos, 1):
            log(f"\n[NÍVEL 2] ({i}/{len(departamentos)}) {depto['nome']}")
            for siape, nome in extrair_docentes_via_post(cliente, depto["id"], depto["nome"]):
                docentes_vistos.setdefault(siape, nome)
            time.sleep(DELAY_ENTRE_DEPARTAMENTOS)

    log(f"\n[NÍVEL 3] {len(docentes_vistos)} perfis a coletar...")
    documentos = asyncio.run(coletar_perfis_async(sorted(docentes_vistos.items()), timestamp))

    # Deduplica aqui, na fonte, porque esta função grava o SQLite logo abaixo e
    # devolve os documentos para o vetor store — é o único ponto que corrige as
    # duas saídas de uma vez.
    #
    # ⚠️ A interpretação mudou (achado 08). O "1278 SIAPEs para 703 pessoas" era
    # atribuído a cadastro duplicado no SIGAA; a maior parte era a corrida de
    # sessão deste próprio scraper devolvendo a MESMA resposta HTTP para ids
    # diferentes — daí os perfis byte a byte idênticos. Com o cliente isolado,
    # espera-se que sobrem pouquíssimas duplicatas. Se ainda sobrarem muitas, há
    # outra fonte de repetição e vale investigar, não deduplicar por cima.
    from deduplicacao import deduplicar_documentos

    documentos, removidos = deduplicar_documentos(documentos)
    if removidos:
        log(f"[DEDUP] {removidos} perfis duplicados descartados -> {len(documentos)} pessoas.")

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