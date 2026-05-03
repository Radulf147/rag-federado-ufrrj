# Projeto: Agente RAG Federado — UFRRJ
# Módulo 1, Parte 1: scraping do portal público do SIGAA
# Autor: Raul Nascimento

import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from haystack import Document


SIGAA_URL        = "https://sigaa.ufrrj.br/sigaa/public/home.jsf"
INSTANCIA        = "sigaa"
MIN_CHARS_TITULO = 4
MIN_CHARS_DESC   = 30

HEADERS = {
    "User-Agent": "UFRRJ-IC-RAG/1.0 (Iniciacao Cientifica - pesquisa academica)",
    "Accept-Language": "pt-BR,pt;q=0.9",
}


def acessar_pagina(url: str) -> str | None:
    print(f"[SCRAPER] Acessando: {url}")
    try:
        r = httpx.get(url, headers=HEADERS, timeout=30, follow_redirects=True)
        r.raise_for_status()
        # SIGAA serve latin-1, não UTF-8
        return r.content.decode("iso-8859-1")
    except httpx.TimeoutException:
        print("[SCRAPER] ERRO: timeout ao acessar o SIGAA (>30s). Tente novamente.")
        return None
    except httpx.HTTPStatusError as e:
        print(f"[SCRAPER] ERRO HTTP {e.response.status_code}: {url}")
        return None
    except httpx.RequestError as e:
        print(f"[SCRAPER] ERRO de rede: {e}")
        return None


def extrair_cards(html: str) -> list[dict]:
    soup  = BeautifulSoup(html, "lxml")
    cards = []

    for h3 in soup.find_all("h3"):
        titulo = h3.get_text(strip=True)
        if not titulo or len(titulo) < MIN_CHARS_TITULO:
            continue

        link = h3.find_next("a")
        if not link:
            continue

        descricao = link.get_text(strip=True)
        href      = link.get("href", "")

        if href.startswith("/"):
            url_completa = f"https://sigaa.ufrrj.br{href}"
        elif href.startswith("http"):
            url_completa = href
        else:
            url_completa = SIGAA_URL

        if descricao and len(descricao) >= MIN_CHARS_DESC:
            cards.append({"titulo": titulo, "descricao": descricao, "url": url_completa})

    return cards


def cards_para_documents(cards: list[dict], timestamp: str) -> list[Document]:
    docs = []
    for card in cards:
        doc = Document(
            content=f"{card['titulo']}: {card['descricao']}",
            meta={
                "instancia_dona": INSTANCIA,
                "content_type":   "sigaa_public_home",
                "source_url":     card["url"],
                "origin_url":     SIGAA_URL,
                "scraped_at":     timestamp,
                "titulo":         card["titulo"],
                "descricao":      card["descricao"],
            },
        )
        docs.append(doc)
    return docs


def scrape_sigaa() -> list[Document]:
    timestamp = datetime.now(timezone.utc).isoformat()

    html = acessar_pagina(SIGAA_URL)
    if html is None:
        print("[SCRAPER] Falha na coleta — retornando lista vazia.")
        print("[SCRAPER] A base de conhecimento anterior será mantida intacta.")
        return []

    cards = extrair_cards(html)
    print(f"[SCRAPER] {len(cards)} documentos extraídos com sucesso.")

    return cards_para_documents(cards, timestamp)


def validar_documentos(docs: list[Document]) -> bool:
    if not docs:
        print("[VALIDAÇÃO] ✗ Nenhum documento foi extraído.")
        return False

    erros = 0
    for campo in ["instancia_dona", "source_url", "scraped_at"]:
        faltando = [i for i, d in enumerate(docs) if campo not in d.meta]
        if faltando:
            print(f"[VALIDAÇÃO] ✗ Campo '{campo}' ausente em {len(faltando)} documentos")
            erros += 1
        else:
            print(f"[VALIDAÇÃO] ✓ Campo '{campo}' presente em todos os documentos")

    errados = [d for d in docs if d.meta.get("instancia_dona") != INSTANCIA]
    if errados:
        print(f"[VALIDAÇÃO] ✗ {len(errados)} documentos com instancia_dona incorreto!")
        erros += 1
    else:
        print(f"[VALIDAÇÃO] ✓ instancia_dona = '{INSTANCIA}' em todos os documentos")

    curtos = [d for d in docs if len(d.content) < MIN_CHARS_DESC]
    if curtos:
        print(f"[VALIDAÇÃO] ⚠ {len(curtos)} documentos com conteúdo muito curto")

    return erros == 0


if __name__ == "__main__":
    print("=" * 60)
    print("PARTE 1 — SCRAPING DO SIGAA PÚBLICO")
    print("=" * 60)

    docs = scrape_sigaa()

    if not docs:
        print("\n[RESULTADO] Scraping falhou. Verifique sua conexão com a internet.")
        exit(1)

    print("\n--- Validação ---")
    ok = validar_documentos(docs)

    print("\n--- Amostra dos primeiros 5 documentos ---")
    for i, doc in enumerate(docs[:5]):
        print(f"\nDocumento {i+1}:")
        print(f"  Conteúdo : {doc.content}")
        print(f"  URL      : {doc.meta['source_url']}")
        print(f"  Coletado : {doc.meta['scraped_at']}")

    print("\n--- Resumo ---")
    print(f"Total de documentos: {len(docs)}")
    print(f"Instancia_dona:      {docs[0].meta['instancia_dona']}")
    print(f"Status da validação: {'✓ OK' if ok else '✗ COM ERROS'}")

    print("\n[PARTE 1 CONCLUÍDA]")
    print("Próximo passo: passar 'docs' para a Parte 2 (chunking).")
    print("  from parte2_chunking import chunkar_documentos")
    print("  chunks = chunkar_documentos(docs)")