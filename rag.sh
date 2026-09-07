#!/bin/bash
# Projeto: Agente RAG Federado — UFRRJ
# Script de conveniência para operação via Docker

set -e

PROJETO="RAG Federado UFRRJ"

mostrar_ajuda() {
    echo "=== $PROJETO ==="
    echo ""
    echo "Uso: ./rag.sh [comando]"
    echo ""
    echo "Comandos:"
    echo "  build     Constrói a imagem Docker do projeto"
    echo "  etl       Sobe o ChromaDB e roda o pipeline ETL completo"
    echo "  agente    Sobe o ChromaDB e abre o agente interativo"
    echo "  comparar  Roda a comparação dos 3 pipelines (gera docs/testes_pipelines.md)"
    echo "  testes    Roda a suíte de testes (reconstrói a imagem antes)"
    echo "  tunel     Gerencia o túnel SSH até o Ollama (./tunel.sh up|status|down)"
    echo "  chroma    Sobe apenas o ChromaDB em background"
    echo "  logs      Exibe logs do ETL em tempo real"
    echo "  limpar    Remove containers e volumes (APAGA o banco)"
    echo "  status    Mostra containers rodando"
    echo ""
    echo "Rede simulada (o cenário onde o agente é exercitado):"
    echo "  rede        Sobe a página e o bot -> http://localhost:5000"
    echo "  rede-parar  Derruba a página e o bot"
    echo "  rede-logs   Segue os logs do worker do bot"
    echo "  semear      APAGA os posts e recria o cenário de exemplo"
    echo ""
}

case "$1" in

    build)
        echo "[BUILD] Construindo imagem..."
        docker compose build
        echo "[BUILD] Concluído."
        ;;

    chroma)
        echo "[CHROMA] Subindo ChromaDB..."
        docker compose up -d chromadb
        echo "[CHROMA] ChromaDB disponível em localhost:8000"
        ;;

    etl)
        echo "[ETL] Subindo ChromaDB e rodando pipeline ETL..."
        # Sobe o ChromaDB em background e aguarda ficar saudável
        docker compose up -d chromadb
        echo "[ETL] Aguardando ChromaDB ficar pronto..."
        docker compose run --rm etl
        echo "[ETL] Pipeline concluído. Verifique os logs em ./logs/"
        ;;

    tunel)
        shift
        ./tunel.sh "$@"
        ;;

    agente)
        echo "[AGENTE] Subindo ChromaDB e abrindo agente interativo..."
        docker compose up -d chromadb
        # O agente é inútil sem o Ollama da invaders. Levantar o túnel aqui é o
        # que permite disparar tudo remotamente, sem abrir terminal à mão.
        # Não aborta se falhar: o REPL ainda sobe, e a mensagem de erro do
        # túnel é mais informativa que um timeout lá dentro.
        ./tunel.sh up || echo "[AVISO] Sem túnel — o agente vai subir, mas não conseguirá responder."
        docker compose --profile agente run --rm agente
        ;;

    comparar)
        echo "[COMPARAR] Rodando a bateria dos 3 pipelines..."
        docker compose up -d chromadb
        ./tunel.sh up || echo "[AVISO] Sem túnel — os pipelines 1 e 3 vão falhar."
        docker compose --profile agente run --rm agente python -m interfaces.comparar
        ;;

    testes)
        # RECONSTRÓI ANTES, e não é zelo: `testes/` não é volume montado, o
        # código vem do COPY da imagem. Rodar pytest sem build executa a versão
        # anterior à edição e devolve verde de código obsoleto — aconteceu em
        # 5 set 2026 (armadilha 1). O id da imagem é impresso para que a saída
        # diga de onde os testes vieram.
        echo "[TESTES] Reconstruindo a imagem..."
        docker compose build agente
        echo "[TESTES] imagem $(docker image inspect rag-federado-ufrrj-agente --format '{{slice .Id 7 19}}')"
        docker compose run --rm --no-deps agente python -m pytest testes/ -q
        ;;

    rede)
        echo "[REDE] Subindo a rede simulada..."
        docker compose up -d chromadb
        # O `bot` precisa do Ollama; a `rede` não. Por isso o túnel não aborta
        # nada aqui: a página sobe e funciona de qualquer jeito, e os posts
        # ficam na fila esperando o worker conseguir responder.
        ./tunel.sh up || echo "[AVISO] Sem túnel — a página sobe, mas o bot não responde."
        docker compose --profile rede up -d rede bot
        echo ""
        echo "[REDE] Página:  http://localhost:5000"
        echo "[REDE] Bot:     ./rag.sh rede-logs   (o warm-up do embedding demora)"
        echo "[REDE] Sem posts? rode ./rag.sh semear"
        ;;

    rede-parar)
        docker compose --profile rede stop rede bot
        echo "[REDE] Página e bot parados. Os posts continuam em dados/rede.db."
        ;;

    rede-logs)
        docker compose --profile rede logs -f bot
        ;;

    semear)
        echo "[SEMEAR] Isso APAGA todos os posts da rede simulada."
        read -p "Confirma? (s/N): " confirmacao
        if [[ "$confirmacao" == "s" || "$confirmacao" == "S" ]]; then
            docker compose --profile rede run --rm --no-deps rede \
                python -m interfaces.rede.semear
        else
            echo "[SEMEAR] Cancelado."
        fi
        ;;

    logs)
        docker compose logs -f etl
        ;;

    status)
        docker compose ps
        ;;

    limpar)
        echo "[AVISO] Isso vai APAGAR o banco ChromaDB e todos os logs."
        read -p "Confirma? (s/N): " confirmacao
        if [[ "$confirmacao" == "s" || "$confirmacao" == "S" ]]; then
            docker compose down -v
            echo "[LIMPAR] Volumes removidos."
        else
            echo "[LIMPAR] Cancelado."
        fi
        ;;

    *)
        mostrar_ajuda
        ;;

esac
