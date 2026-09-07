@echo off
rem ===================================================================
rem  tunel.cmd - chama o tunel.sh a partir do PowerShell / cmd.exe
rem
rem  POR QUE EXISTE
rem  No PowerShell, "./tunel.sh manter" nao executa o script: o Windows
rem  entrega o arquivo ao programa associado a extensao .sh e abre a
rem  janela "escolha um aplicativo". Aconteceu em 7 set 2026.
rem
rem  POR QUE NAO BASTA CHAMAR "bash"
rem  Nesta maquina o bash do PATH e C:\Windows\system32\bash.exe, do WSL -
rem  outro sistema de arquivos, outro ~/.ssh, outro /tmp. O tunel subiria
rem  num ambiente diferente do resto do projeto, e o socket de controle
rem  /tmp/rag_tunel_dcc nao seria o mesmo que "status" e "down" procuram.
rem  Por isso o caminho do Git Bash e explicito aqui.
rem
rem  DUAS ARMADILHAS DE .CMD, as duas cometidas na primeira versao:
rem  1. arquivo .cmd PRECISA de quebra de linha CRLF; com LF o cmd.exe
rem     interpreta lixo e reclama de comandos que nao existem.
rem  2. %ProgramFiles(x86)% nao pode aparecer DENTRO de um bloco entre
rem     parenteses: o ")" de "(x86)" fecha o bloco antes da hora. Por
rem     isso o caminho vai para uma variavel ANTES do if.
rem
rem  Uso:  .\tunel.cmd up ^| manter ^| status ^| down ^| ajuda
rem ===================================================================
setlocal

set "P86=%ProgramFiles(x86)%"

set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%P86%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"

if not exist "%BASH%" (
    echo [ERRO] Git Bash nao encontrado.
    echo        Instale o Git para Windows, ou rode pelo Git Bash:
    echo            ./tunel.sh %*
    exit /b 1
)

"%BASH%" "%~dp0tunel.sh" %*
