"""
Interface CLI (REPL local) — para testar o agente pelo terminal.

Único papel: interação via terminal. Monta os componentes (llm_setup),
inicializa o histórico (agent) e faz o loop de leitura/impressão. Toda a
lógica de decisão fica em agent.py — este arquivo não sabe nada sobre tools
ou sobre como o LLM decide qual usar.

Quando o Módulo 3 (Mastodon) existir, ele vira um arquivo irmão deste,
interfaces/mastodon_listener.py, chamando os mesmos
llm_setup.montar_componentes() e agent.processar_pergunta() — sem duplicar
nada do que está aqui.
"""

from modulo2_inferencia.llm_setup import montar_componentes
from modulo2_inferencia.agent import montar_historico_inicial, processar_pergunta


def executar_repl() -> None:
    print("=" * 60)
    print("Inicializando Motor Agentic RAG (Haystack + Ollama)...")
    print("=" * 60)

    componentes = montar_componentes()
    chat_history = montar_historico_inicial()

    print(
        "Agente pronto! Pode fazer perguntas estruturadas "
        "(ex: 'quantos professores tem a física?') ou interpretativas."
    )

    while True:
        pergunta_usuario = input("\nVocê: ").strip()
        if pergunta_usuario.lower() in ("sair", "exit", "q"):
            break
        if not pergunta_usuario:
            continue

        resposta_texto, chat_history = processar_pergunta(
            chat_generator=componentes.chat_generator,
            embedder=componentes.embedder,
            retriever=componentes.retriever,
            chat_history=chat_history,
            pergunta_usuario=pergunta_usuario,
        )
        print(f"\nGrok UFRRJ: {resposta_texto}")


if __name__ == "__main__":
    executar_repl()
