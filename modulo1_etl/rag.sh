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
    echo "  chroma    Sobe apenas o ChromaDB em background"
    echo "  logs      Exibe logs do ETL em tempo real"
    echo "  limpar    Remove containers e volumes (APAGA o banco)"
    echo "  status    Mostra containers rodando"
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

    agente)
        echo "[AGENTE] Subindo ChromaDB e abrindo agente interativo..."
        docker compose up -d chromadb
        docker compose --profile agente run --rm agente
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
