# =============================================================================
# MÓDULO 1 — PARTE 1: SCRAPING DO SIGAA PÚBLICO
# Projeto: Agente RAG Federado — UFRRJ
# Autor: Raul Nascimento
# =============================================================================
#
# O que este script faz:
#   1. Acessa o portal público do SIGAA (sigaa.ufrrj.br/sigaa/public/home.jsf)
#   2. Extrai todos os cards de conteúdo (título + descrição + URL)
#   3. Converte cada card em um objeto Document do Haystack
#   4. Adiciona metadados de governança obrigatórios (instancia_dona, etc.)
#   5. Imprime um relatório de validação no terminal
#
# Instalação das dependências:
#   pip install haystack-ai httpx beautifulsoup4 lxml
#
# Como executar:
#   python parte1_scraping.py
#
# Saída esperada:
#   [SCRAPER] Acessando: https://sigaa.ufrrj.br/sigaa/public/home.jsf
#   [SCRAPER] 47 documentos extraídos com sucesso.
#   [VALIDAÇÃO] ✓ Campo instancia_dona presente em todos os documentos
#   ... (amostra dos documentos)
# =============================================================================

import httpx
from bs4 import BeautifulSoup
from datetime import datetime, timezone
from haystack import Document


# =============================================================================
# CONFIGURAÇÃO — altere apenas estas constantes se necessário
# =============================================================================

SIGAA_URL    = "https://sigaa.ufrrj.br/sigaa/public/home.jsf"
INSTANCIA    = "sigaa"   # etiqueta de governança — NUNCA alterar em runtime

# User-Agent identificado: boa prática para scraping acadêmico
HEADERS = {
    "User-Agent": "UFRRJ-IC-RAG/1.0 (Iniciacao Cientifica - pesquisa academica)",
    "Accept-Language": "pt-BR,pt;q=0.9",
}

# Mínimo de caracteres para um texto ser considerado conteúdo válido
MIN_CHARS = 30


# =============================================================================
# FUNÇÕES
# =============================================================================

def acessar_pagina(url: str) -> str | None:
    """
    Faz a requisição HTTP para a URL do SIGAA e retorna o HTML bruto.

    Retorna None em caso de falha — o chamador decide o que fazer.
    Nunca lança exceção: falha suave para não derrubar o ETL inteiro.
    """
    print(f"[SCRAPER] Acessando: {url}")
    try:
        resposta = httpx.get(
            url,
            headers=HEADERS,
            timeout=30,
            follow_redirects=True,
        )
        resposta.raise_for_status()  # lança erro se status != 2xx

        # O SIGAA usa ISO-8859-1 (Latin-1) — garantimos a decodificação correta
        return resposta.content.decode("iso-8859-1")

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
    """
    Analisa o HTML do SIGAA e extrai os cards de conteúdo público.

    A home do SIGAA organiza o conteúdo em cards com esta estrutura HTML:
        <h3>Título do Card</h3>
        <a href="/sigaa/public/...">Descrição do card.</a>

    Retorna uma lista de dicionários com:
        - titulo:    texto do <h3>
        - descricao: texto do <a> (descrição do serviço)
        - url:       href do <a> (link para a página do serviço)
    """
    soup = BeautifulSoup(html, "lxml")
    cards = []

    # Cada h3 é o título de um serviço público no SIGAA
    for h3 in soup.find_all("h3"):
        titulo = h3.get_text(strip=True)

        if not titulo or len(titulo) < MIN_CHARS:
            continue  # ignora títulos vazios ou muito curtos

        # O link descritivo geralmente é o próximo elemento após o h3
        link = h3.find_next("a")
        if not link:
            continue  # sem link = sem destino útil para o usuário

        descricao = link.get_text(strip=True)
        href = link.get("href", "")

        # Monta URL completa se o href for relativo
        if href.startswith("/"):
            url_completa = f"https://sigaa.ufrrj.br{href}"
        elif href.startswith("http"):
            url_completa = href
        else:
            url_completa = SIGAA_URL  # fallback: aponta para a home

        if descricao and len(descricao) >= MIN_CHARS:
            cards.append({
                "titulo":    titulo,
                "descricao": descricao,
                "url":       url_completa,
            })

    return cards


def cards_para_documents(cards: list[dict], timestamp: str) -> list[Document]:
    """
    Converte cada card extraído em um objeto Document do Haystack.

    O conteúdo de cada Document combina título + descrição em um texto
    completo e natural, que o modelo de embedding vai entender bem.

    Metadados de governança são adicionados aqui — nunca depois.
    """
    documentos = []

    for card in cards:
        # Texto do documento: frase completa combinando título e descrição
        # Formato pensado para o embedding entender o contexto completo
        conteudo = f"{card['titulo']}: {card['descricao']}"

        doc = Document(
            content=conteudo,
            meta={
                # ── Governança (obrigatório) ──────────────────────────────
                "instancia_dona": INSTANCIA,     # filtro central de isolamento
                "content_type":   "sigaa_public_home",

                # ── Rastreabilidade ───────────────────────────────────────
                "source_url":     card["url"],   # link para o serviço original
                "origin_url":     SIGAA_URL,     # página onde foi coletado
                "scraped_at":     timestamp,     # quando foi coletado (UTC)

                # ── Conteúdo estruturado (útil para debug e filtros) ──────
                "titulo":         card["titulo"],
                "descricao":      card["descricao"],
            }
        )
        documentos.append(doc)

    return documentos


def scrape_sigaa() -> list[Document]:
    """
    Função principal da Parte 1.

    Orquestra as etapas: acesso → extração → conversão.
    Retorna a lista de Documents pronta para a Parte 2 (chunking).
    Em caso de falha na coleta, retorna lista vazia.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    # Etapa 1: acesso HTTP
    html = acessar_pagina(SIGAA_URL)
    if html is None:
        print("[SCRAPER] Falha na coleta — retornando lista vazia.")
        print("[SCRAPER] A base de conhecimento anterior será mantida intacta.")
        return []

    # Etapa 2: extração dos cards
    cards = extrair_cards(html)
    print(f"[SCRAPER] {len(cards)} documentos extraídos com sucesso.")

    # Etapa 3: conversão para Documents do Haystack
    documentos = cards_para_documents(cards, timestamp)

    return documentos


# =============================================================================
# VALIDAÇÃO — verifica a qualidade dos documentos extraídos
# =============================================================================

def validar_documentos(documentos: list[Document]) -> bool:
    """
    Executa verificações básicas de qualidade nos documentos extraídos.

    Retorna True se todos os critérios forem atendidos.
    Imprime warnings para problemas encontrados.
    """
    if not documentos:
        print("[VALIDAÇÃO] ✗ Nenhum documento foi extraído.")
        return False

    erros = 0

    # Verifica campos obrigatórios de governança em todos os documentos
    campos_obrigatorios = ["instancia_dona", "source_url", "scraped_at"]
    for campo in campos_obrigatorios:
        faltando = [i for i, d in enumerate(documentos) if campo not in d.meta]
        if faltando:
            print(f"[VALIDAÇÃO] ✗ Campo '{campo}' ausente em {len(faltando)} documentos")
            erros += 1
        else:
            print(f"[VALIDAÇÃO] ✓ Campo '{campo}' presente em todos os documentos")

    # Verifica que todos têm instancia_dona correto
    errados = [d for d in documentos if d.meta.get("instancia_dona") != INSTANCIA]
    if errados:
        print(f"[VALIDAÇÃO] ✗ {len(errados)} documentos com instancia_dona incorreto!")
        erros += 1
    else:
        print(f"[VALIDAÇÃO] ✓ instancia_dona = '{INSTANCIA}' em todos os documentos")

    # Verifica tamanho mínimo de conteúdo
    curtos = [d for d in documentos if len(d.content) < MIN_CHARS]
    if curtos:
        print(f"[VALIDAÇÃO] ⚠ {len(curtos)} documentos com conteúdo muito curto")

    return erros == 0


# =============================================================================
# EXECUÇÃO DIRETA
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PARTE 1 — SCRAPING DO SIGAA PÚBLICO")
    print("=" * 60)

    # Executa o scraping
    docs = scrape_sigaa()

    if not docs:
        print("\n[RESULTADO] Scraping falhou. Verifique sua conexão com a internet.")
        exit(1)

    # Valida a qualidade
    print("\n--- Validação ---")
    ok = validar_documentos(docs)

    # Exibe amostra dos primeiros 5 documentos
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
