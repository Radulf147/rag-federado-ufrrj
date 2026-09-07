@echo off
rem ===================================================================
rem  rag.cmd - chama o rag.sh a partir do PowerShell / cmd.exe
rem
rem  Mesmo motivo do tunel.cmd: "./rag.sh" no PowerShell abre a janela
rem  "escolha um aplicativo" em vez de executar, e o bash do PATH nesta
rem  maquina e o do WSL. Ver o cabecalho de tunel.cmd, que registra
rem  tambem as duas armadilhas de arquivo .cmd (CRLF obrigatorio e o
rem  parentese de "(x86)" fechando bloco if antes da hora).
rem
rem  Uso:  .\rag.cmd rede ^| testes ^| semear ^| agente ^| etl ^| ...
rem ===================================================================
setlocal

set "P86=%ProgramFiles(x86)%"

set "BASH=%ProgramFiles%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%P86%\Git\bin\bash.exe"
if not exist "%BASH%" set "BASH=%LOCALAPPDATA%\Programs\Git\bin\bash.exe"

if not exist "%BASH%" (
    echo [ERRO] Git Bash nao encontrado.
    echo        Instale o Git para Windows, ou rode pelo Git Bash:
    echo            ./rag.sh %*
    exit /b 1
)

"%BASH%" "%~dp0rag.sh" %*
