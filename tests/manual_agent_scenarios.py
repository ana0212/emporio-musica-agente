"""Cenários manuais de integração com o Qwen via Ollama.

As conversas são salvas automaticamente em test_results/ como Markdown.
Estes cenários chamam o modelo local real; não são testes unitários de pytest.
"""

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import ask_agent  # noqa: E402


SCENARIOS = [
    ("01 - Saudação inicial", ["Olá, tudo bem?"]),
    ("02 - Busca de produtos", [
        "Quais violões vocês têm disponíveis por até R$ 1.000?"
    ]),
    ("03 - Política de arrependimento", [
        "Comprei um instrumento pela internet e me arrependi. Posso devolver?"
    ]),
    ("04 - Consulta de pedido", [
        "Quero consultar o status do meu pedido."
    ]),
    ("05 - Continuidade e mudança de assunto", [
        "Posso trocar um instrumento porque não gostei da cor?",
        "E se o produto estiver com defeito?",
    ]),
]


def run_scenario(name, user_messages):
    lines = [f"## {name}", ""]
    history = []
    success = True

    for user_message in user_messages:
        lines.extend([f"**Cliente:** {user_message}", ""])

        try:
            answer = ask_agent(user_message, history=history)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            lines.extend([f"**Erro:** `{error}`", ""])
            print(f"ERRO: {error}")
            success = False
            break

        answer = answer or ""
        lines.extend([f"**Kiko:** {answer}", ""])

        if not answer.strip():
            lines.extend(["**Resultado:** resposta vazia.", ""])
            success = False
            break

        history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer},
        ])

    lines.extend([
        f"**Execução:** {'concluída' if success else 'com erro'}",
        "---",
        "",
    ])

    print("\n".join(lines))
    return success, "\n".join(lines)


def main():
    print("TESTES MANUAIS DE INTEGRAÇÃO - EMPÓRIO DA MÚSICA")
    print("Modelo configurado no agent.py: qwen2.5:3b")
    print("Confirme que Ollama, SQLite e ChromaDB estão disponíveis.")
    input("Pressione ENTER para iniciar os cenários...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"conversas_qwen_{timestamp}.md"

    document = [
        "# Registro de testes manuais do agente",
        "",
        f"- Data/hora: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "- Modelo configurado: `qwen2.5:3b`",
        "- Execução: Ollama local, com ferramentas e bases locais do projeto",
        "",
        "> Registro bruto das respostas."
        "",
        "---",
        "",
    ]

    results = []
    for name, messages in SCENARIOS:
        success, transcript = run_scenario(name, messages)
        results.append((name, success))
        document.append(transcript)
        # Save after every scenario so partial results are retained if interrupted.
        output_file.write_text("\n".join(document), encoding="utf-8")

    print("\nRESUMO")
    for name, success in results:
        print(f'{"EXECUTADO" if success else "ERRO"}: {name}')

    print(f"\nRegistro salvo em: {output_file}")


if __name__ == "__main__":
    main()
