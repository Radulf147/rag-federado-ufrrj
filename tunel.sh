#!/bin/bash
# Projeto: Agente RAG Federado — UFRRJ
# Gerencia o túnel SSH até o Ollama da invaders, sem interação humana.
#
# POR QUE ISSO EXISTE: o túnel era aberto à mão num terminal, o que obrigava
# a estar fisicamente no PC. Este script é feito para ser rodado por um
# agente/despacho remoto — nunca faz pergunta, nunca abre prompt de senha,
# e sempre sai com código de status honesto.
#
# EXIGE AUTENTICAÇÃO POR CHAVE. Com senha, o ssh abriria um prompt que numa
# sessão não-interativa trava para sempre; por isso passamos BatchMode=yes,
# que transforma "pediria senha" em "falha em segundos com mensagem clara".

set -uo pipefail

PORTA_LOCAL="${PORTA_TUNEL:-9999}"
HOST_OLLAMA="invaders.dcc.ufrrj.br"
PORTA_OLLAMA=11434
JUMP_HOST="www.dcc.ufrrj.br"
LOG="${TMPDIR:-/tmp}/tunel_dcc.log"

# Socket de controle do ssh (ControlMaster). É o que permite consultar e
# encerrar o túnel pela própria API do ssh, em vez de caçar PID no netstat.
#
# O caminho NÃO pode conter espaço: o ssh do Windows quebra com
# "keyword controlpath extra arguments at end of line". Isso exclui ~/.ssh
# neste perfil de usuário ("Raul Nascimento"), daí ficar em /tmp.
SOCKET="${CONTROL_PATH:-/tmp/rag_tunel_dcc}"

# Usuário do DCC — vem do .env para não ficar hardcoded no repositório.
if [ -f .env ]; then
    DCC_USUARIO="${DCC_USUARIO:-$(grep -E '^DCC_USUARIO=' .env 2>/dev/null | cut -d= -f2- | tr -d '\r')}"
fi
DCC_USUARIO="${DCC_USUARIO:-}"

# --- estado ---------------------------------------------------------------

no_ar() {
    # A prova real não é "existe processo ssh", é "o Ollama responde".
    curl -s -m 4 "http://localhost:${PORTA_LOCAL}/api/version" >/dev/null 2>&1
}

master_vivo() {
    # Pergunta ao próprio ssh se o master do socket está de pé.
    #
    # Substitui a busca de PID via netstat, que era inconfiável: medido em
    # Set/2026, o netstat reportava para a porta 9999 um PID (24568) que não
    # existia no tasklist, e o taskkill respondia "processo não encontrado".
    # O `ssh -f` bifurca e o PID que o netstat associa ao socket fica obsoleto.
    # Resultado: `down` não derrubava nada e ainda assim reportava sucesso, e
    # processos ssh órfãos se acumulavam.
    [ -S "$SOCKET" ] || return 1
    ssh -o ControlPath="$SOCKET" -O check "${DCC_USUARIO}@${JUMP_HOST}" >/dev/null 2>&1
}

# --- comandos -------------------------------------------------------------

cmd_status() {
    # Três estados distintos, que a versão anterior confundia num só: o master
    # ssh estar vivo, o encaminhamento funcionar, e o Ollama responder.
    if no_ar; then
        local versao
        versao=$(curl -s -m 4 "http://localhost:${PORTA_LOCAL}/api/version")
        echo "[OK] Túnel no ar em localhost:${PORTA_LOCAL} — Ollama responde: ${versao}"
        return 0
    fi
    if master_vivo; then
        echo "[FALHA] O master ssh está vivo, mas o Ollama não responde pela porta ${PORTA_LOCAL}."
        echo "        Ou a invaders está fora, ou o Ollama parou nela, ou o"
        echo "        encaminhamento caiu. Rode: ./tunel.sh down && ./tunel.sh up"
        return 1
    fi
    echo "[FECHADO] Nenhum túnel ativo (sem master em ${SOCKET})."
    return 1
}

cmd_up() {
    if no_ar; then
        echo "[OK] Túnel já estava no ar — nada a fazer."
        return 0
    fi

    if [ -z "$DCC_USUARIO" ]; then
        echo "[ERRO] DCC_USUARIO não definido."
        echo "       Adicione ao .env do projeto:  DCC_USUARIO=seu_usuario_do_dcc"
        return 2
    fi

    # Master vivo mas sem resposta = túnel meio-morto. Agora dá para limpar de
    # verdade: o `-O exit` encerra pelo socket, sem depender de PID.
    if master_vivo; then
        echo "[LIMPEZA] Master anterior sem resposta — encerrando antes de reabrir..."
        cmd_down >/dev/null 2>&1
    fi
    # Socket órfão de um master já morto impediria o ssh de subir.
    [ -S "$SOCKET" ] && rm -f "$SOCKET"

    echo "[TÚNEL] Abrindo ${DCC_USUARIO}@${JUMP_HOST} -L ${PORTA_LOCAL}:${HOST_OLLAMA}:${PORTA_OLLAMA} ..."

    # -f          desacopla do shell (sobrevive ao fim desta sessão)
    # -N          só encaminha portas, não abre shell remoto
    # BatchMode   falha rápido em vez de pedir senha
    # ExitOnForwardFailure  não finge sucesso se a porta não puder ser aberta
    # ServerAlive derruba o túnel se a rede cair, em vez de deixar zumbi
    # ControlMaster cria o socket que torna `status` e `down` determinísticos
    ssh -f -N \
        -o BatchMode=yes \
        -o ConnectTimeout=10 \
        -o ExitOnForwardFailure=yes \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ControlMaster=yes \
        -o ControlPath="$SOCKET" \
        -L "${PORTA_LOCAL}:${HOST_OLLAMA}:${PORTA_OLLAMA}" \
        "${DCC_USUARIO}@${JUMP_HOST}" 2>"$LOG"

    local codigo=$?
    if [ $codigo -ne 0 ]; then
        echo "[ERRO] ssh falhou (código ${codigo}). Saída:"
        sed 's/^/       /' "$LOG"
        echo
        echo "       Se a mensagem fala em 'Permission denied (publickey,password)',"
        echo "       a chave SSH ainda não foi instalada no servidor — ver ./tunel.sh ajuda"
        return $codigo
    fi

    # ssh -f volta na hora; o encaminhamento leva um instante para valer.
    for _ in $(seq 1 15); do
        if no_ar; then
            echo "[OK] Túnel no ar. Ollama: $(curl -s -m 4 "http://localhost:${PORTA_LOCAL}/api/version")"
            return 0
        fi
        sleep 1
    done

    echo "[ERRO] Túnel abriu mas o Ollama não respondeu em 15s."
    echo "       A invaders pode estar fora, ou o Ollama parado nela."
    return 1
}

cmd_down() {
    if ! master_vivo; then
        [ -S "$SOCKET" ] && rm -f "$SOCKET"
        echo "[FECHADO] Não havia túnel ativo."
        return 0
    fi

    ssh -o ControlPath="$SOCKET" -O exit "${DCC_USUARIO}@${JUMP_HOST}" >/dev/null 2>&1

    # Verificar, e não presumir: a versão anterior imprimia "derrubado" mesmo
    # quando o taskkill falhava, e foi assim que órfãos se acumularam sem que
    # ninguém percebesse.
    if master_vivo || no_ar; then
        echo "[ERRO] O túnel não encerrou. Master ainda responde em ${SOCKET}."
        return 1
    fi
    [ -S "$SOCKET" ] && rm -f "$SOCKET"
    echo "[FECHADO] Túnel encerrado."
}

cmd_manter() {
    # Supervisor: mantém o túnel vivo indefinidamente.
    #
    # POR QUE: o ssh -f com ServerAlive derruba o túnel quando a rede cai, mas
    # não o levanta de volta — medido na prática, uma oscilação de internet
    # deixou o túnel morto e silencioso até alguém rodar `up` à mão. Para
    # operação pelo telefone isso é inaceitável: o comando remoto falha e não
    # há ninguém no PC para perceber.
    local intervalo="${INTERVALO_CHECAGEM:-30}"
    echo "[MANTER] Supervisionando a cada ${intervalo}s. Ctrl+C para parar."
    local quedas=0
    while true; do
        if ! no_ar; then
            quedas=$((quedas + 1))
            echo "[MANTER] $(date '+%H:%M:%S') túnel caído (queda #${quedas}) — reabrindo..."
            cmd_up || echo "[MANTER] falha ao reabrir; tento de novo em ${intervalo}s"
        fi
        sleep "$intervalo"
    done
}

cmd_ajuda() {
    cat <<'AJUDA'
=== Túnel SSH → Ollama da invaders ===

Uso: ./tunel.sh [up|manter|status|down|ajuda]

  up      Abre o túnel se ainda não estiver no ar (idempotente)
  manter  Supervisiona e reabre sozinho quando a rede cai (fica em execução)
  status  Diz se o Ollama responde através do túnel
  down    Derruba o túnel

Configuração, no .env do projeto:
  DCC_USUARIO=seu_usuario_do_dcc
  PORTA_TUNEL=9999          (opcional; 9999 é o padrão)

--- SETUP DE UMA VEZ SÓ (precisa do PC, ~2 min) ---

Este script exige chave SSH: com senha, o ssh abriria um prompt que trava
qualquer execução automatizada. Para instalar a chave, no PC, uma única vez:

  ssh-keygen -t ed25519 -C "rag-ufrrj"        # Enter em tudo, SEM passphrase
  ssh-copy-id SEU_USUARIO@www.dcc.ufrrj.br    # pede sua senha ESTA vez

Se o ssh-copy-id não existir no Windows, o equivalente é:

  cat ~/.ssh/id_ed25519.pub | ssh SEU_USUARIO@www.dcc.ufrrj.br \
      "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys"

Depois disso, ./tunel.sh up funciona sem digitar nada — e pode ser
disparado remotamente, do telefone.

Sobre a passphrase: sem passphrase, a chave sozinha abre o túnel, que é o
que torna o despacho remoto possível. Em troca, quem tiver acesso de leitura
ao seu perfil do Windows consegue usá-la. Para uma conta de laboratório cujo
acesso é só um túnel, é uma troca comum; se preferir passphrase, ela
precisaria ficar destravada num ssh-agent persistente — mais peças, mesmo
resultado prático.
AJUDA
}

case "${1:-ajuda}" in
    up)     cmd_up ;;
    manter) cmd_manter ;;
    status) cmd_status ;;
    down)   cmd_down ;;
    *)      cmd_ajuda ;;
esac
