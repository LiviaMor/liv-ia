#!/usr/bin/env python3
"""
LIV IA - Assistente de Arquitetura de Software Local
CLI principal da aplicação
"""

import json
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from brain import LIVIAEngine, detect_slides_request
from ingestor import DocumentProcessor

console = Console()


def show_welcome():
    """Exibe boas-vindas e menu principal"""
    console.clear()
    welcome_text = """
[bold cyan]╔═══════════════════════════════════════════════════════╗
║                    LIV IA v1.0                        ║
║        Arquiteta de Soluções Sênior Local             ║
╚═══════════════════════════════════════════════════════╝[/bold cyan]

[dim]Especialista em HealthTech, Padrões de Projeto e Automação[/dim]
    """
    console.print(welcome_text)


def show_menu():
    """Exibe o menu de opções"""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Opção", style="cyan bold", width=8)
    table.add_column("Descrição", style="white")

    table.add_row("1", "Iniciar Chat Interativo")
    table.add_row("2", "Fazer Pergunta Rápida")
    table.add_row("3", "Indexar Documentos (PDF, MD, código...)")
    table.add_row("4", "Ver Conversas Salvas")
    table.add_row("5", "Status da Base de Conhecimento")
    table.add_row("6", "Configurações")
    table.add_row("0", "Sair")

    console.print(table)
    console.print()


def save_conversation(messages, folder="docs/conversas"):
    """Salva a conversa em arquivo JSON"""
    Path(folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/conversa_{timestamp}.json"

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(
            {"timestamp": datetime.now().isoformat(), "messages": messages},
            f,
            ensure_ascii=False,
            indent=2,
        )

    return filename


def list_conversations(folder="docs/conversas"):
    """Lista conversas salvas"""
    if not os.path.exists(folder):
        console.print("[yellow]Nenhuma conversa salva ainda.[/yellow]")
        return

    files = sorted(Path(folder).glob("conversa_*.json"), reverse=True)

    if not files:
        console.print("[yellow]Nenhuma conversa salva ainda.[/yellow]")
        return

    table = Table(title="Conversas Salvas", show_header=True)
    table.add_column("Data/Hora", style="cyan")
    table.add_column("Mensagens", style="green")
    table.add_column("Arquivo", style="dim")

    for file in files[:10]:  # Mostra últimas 10
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            timestamp = datetime.fromisoformat(data["timestamp"])
            msg_count = len(data["messages"])
            table.add_row(timestamp.strftime("%d/%m/%Y %H:%M"), str(msg_count), file.name)

    console.print(table)


def show_quick_options():
    """Mostra opções rápidas durante o chat"""
    options = [
        "Sugerir arquitetura",
        "Analisar padrões",
        "Boas práticas HealthTech",
        "Segurança e LGPD",
        "Otimização de performance",
        "Escrever minha pergunta",
    ]

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Opção", style="dim", width=4)
    table.add_column("Descrição", style="cyan")

    for i, opt in enumerate(options, 1):
        table.add_row(f"[{i}]", opt)

    console.print("\n[dim]Opções rápidas:[/dim]")
    console.print(table)


def interactive_chat():
    """Chat interativo com opções e salvamento"""
    console.print(
        Panel.fit(
            "[bold cyan]LIV IA[/bold cyan] - Chat Interativo\n"
            "[dim]Pressione Ctrl+C para sair e salvar a conversa[/dim]",
            border_style="cyan",
        )
    )

    engine = LIVIAEngine()
    messages = []

    try:
        while True:
            show_quick_options()

            choice = Prompt.ask(
                "\n[bold green]Escolha uma opção ou digite sua pergunta[/bold green]", default="6"
            )

            # Opções rápidas
            quick_prompts = {
                "1": "Sugira uma arquitetura de software adequada para o meu projeto",
                "2": "Analise os padrões de projeto mais adequados para esta situação",
                "3": "Quais são as melhores práticas de HealthTech que devo seguir?",
                "4": "Como garantir segurança e conformidade com LGPD?",
                "5": "Como posso otimizar a performance da aplicação?",
            }

            if choice in quick_prompts:
                user_input = quick_prompts[choice]
                console.print(f"\n[bold green]Você:[/bold green] {user_input}")
            elif choice == "6" or not choice.isdigit():
                user_input = Prompt.ask("\n[bold green]Você[/bold green]")
            else:
                console.print("[yellow]Opção inválida. Digite sua pergunta:[/yellow]")
                user_input = Prompt.ask("\n[bold green]Você[/bold green]")

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            # Pedido de slides Marp? Gera e salva o arquivo localmente.
            topic = detect_slides_request(user_input)
            if topic:
                console.print("\n[bold cyan]LIV IA:[/bold cyan] gerando apresentação Marp...")
                try:
                    with console.status("[cyan]Montando os slides...[/cyan]"):
                        path = engine.generate_slides(topic)
                    msg = f"Apresentação Marp salva em: {path}"
                    console.print(f"[green]OK[/green] {msg}")
                    console.print(
                        "[dim]Converter para PDF:[/dim] " f"npx -y @marp-team/marp-cli {path} --pdf"
                    )
                    messages.append({"role": "assistant", "content": msg})
                except Exception as e:
                    console.print(f"[red]ERRO[/red] ao gerar slides: {e}")
                console.print("\n" + "─" * 60 + "\n")
                continue

            console.print("\n[bold cyan]LIV IA:[/bold cyan]")
            with console.status("[cyan]Pensando...[/cyan]"):
                response = engine.chat(user_input)

            console.print(Markdown(response))
            messages.append({"role": "assistant", "content": response})

            console.print("\n" + "─" * 60 + "\n")

    except KeyboardInterrupt:
        console.print("\n\n[yellow]Encerrando chat...[/yellow]")

        if messages:
            if Confirm.ask("\n[cyan]Deseja salvar esta conversa?[/cyan]", default=True):
                filename = save_conversation(messages)
                console.print(f"[green]OK[/green] Conversa salva em: [cyan]{filename}[/cyan]")

        console.print("[yellow]Até logo![/yellow]\n")


def quick_ask():
    """Pergunta rápida"""
    question = Prompt.ask("\n[bold green]Sua pergunta[/bold green]")

    if not question.strip():
        console.print("[yellow]Pergunta vazia.[/yellow]")
        return

    console.print(
        Panel.fit(
            "[bold cyan]LIV IA[/bold cyan] - Consultando base de conhecimento...",
            border_style="cyan",
        )
    )

    engine = LIVIAEngine()
    try:
        with console.status("[cyan]Pensando...[/cyan]"):
            response = engine.ask(question)

        console.print("\n[bold cyan]Resposta:[/bold cyan]\n")
        console.print(Markdown(response))
        console.print()

        if Confirm.ask("\n[cyan]Deseja salvar esta resposta?[/cyan]", default=False):
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": response},
            ]
            filename = save_conversation(messages)
            console.print(f"[green]OK[/green] Resposta salva em: [cyan]{filename}[/cyan]")

    except Exception as e:
        console.print(f"[red]ERRO[/red] {str(e)}")


def ingest_documents():
    """Indexa documentos (PDF, Markdown, texto e código-fonte)"""
    path = Prompt.ask("\n[cyan]Caminho da pasta com documentos[/cyan]", default="./docs")

    collection = Prompt.ask(
        "\n[cyan]Nome da knowledge base[/cyan] [dim](ex: react, aws, healthtech)[/dim]",
        default="livia_default",
    )

    console.print(
        Panel.fit("[bold cyan]LIV IA[/bold cyan] - Processando documentos...", border_style="cyan")
    )

    processor = DocumentProcessor()
    try:
        with console.status("[cyan]Processando documentos...[/cyan]"):
            count = processor.ingest_directory(path, collection_name=collection)
        console.print(
            f"[green]OK[/green] {count} documentos indexados na base " f"[cyan]{collection}[/cyan]!"
        )
    except Exception as e:
        console.print(f"[red]ERRO[/red] {str(e)}")


def show_status():
    """Mostra status da base de conhecimento"""
    storage_path = ".livia_storage"

    if os.path.exists(storage_path):
        size = sum(f.stat().st_size for f in Path(storage_path).rglob("*") if f.is_file())
        size_mb = size / (1024 * 1024)

        console.print(
            Panel(
                f"[green]OK[/green] Base de conhecimento ativa\n"
                f"[dim]Tamanho: {size_mb:.2f} MB\n"
                f"Local: {storage_path}[/dim]",
                title="Status",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel(
                "[yellow]AVISO[/yellow] Base de conhecimento vazia\n"
                "[dim]Execute a opção 3 para indexar documentos[/dim]",
                title="Status",
                border_style="yellow",
            )
        )


def show_settings():
    """Mostra e permite alterar configurações"""
    console.print(Panel.fit("[bold cyan]Configurações[/bold cyan]", border_style="cyan"))

    table = Table(show_header=True)
    table.add_column("Configuração", style="cyan")
    table.add_column("Valor Atual", style="white")

    table.add_row("Modelo de Chat", "deepseek-coder-v2")
    table.add_row("Modelo de Embeddings", "nomic-embed-text")
    table.add_row("URL do Ollama", "http://localhost:11434")
    table.add_row("Pasta de Storage", ".livia_storage")
    table.add_row("Pasta de Conversas", "docs/conversas")

    console.print(table)
    console.print("\n[dim]Para alterar, edite o arquivo brain.py[/dim]")


def main():
    """Função principal"""
    while True:
        show_welcome()
        show_menu()

        choice = Prompt.ask(
            "[bold cyan]Escolha uma opção[/bold cyan]",
            choices=["0", "1", "2", "3", "4", "5", "6"],
            default="1",
        )

        console.print()

        if choice == "0":
            console.print("[yellow]Até logo![/yellow]\n")
            break
        elif choice == "1":
            interactive_chat()
        elif choice == "2":
            quick_ask()
        elif choice == "3":
            ingest_documents()
        elif choice == "4":
            list_conversations()
        elif choice == "5":
            show_status()
        elif choice == "6":
            show_settings()

        if choice != "1":  # Chat já tem seu próprio fluxo
            console.print()
            Prompt.ask("\n[dim]Pressione Enter para continuar[/dim]", default="")


if __name__ == "__main__":
    main()
