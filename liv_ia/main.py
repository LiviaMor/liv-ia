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
    """Exibe boas-vindas do chat direto."""
    console.clear()
    welcome_text = """
[bold cyan]╔═══════════════════════════════════════════════════════╗
║                    LIV IA v2.0                        ║
║        Arquiteta de Soluções Sênior Local             ║
╚═══════════════════════════════════════════════════════╝[/bold cyan]

[dim]Converse direto. Na 1ª mensagem, diga a especialidade/foco do chat
(ex.: "especialista em Node.js" ou "foco em microsserviços").[/dim]

[dim]Comandos: /ajuda  /indexar  /status  /feedback bom|ruim  /sair[/dim]
    """
    console.print(welcome_text)


def show_help():
    """Mostra os comandos disponíveis no chat."""
    table = Table(show_header=True, title="Comandos")
    table.add_column("Comando", style="cyan")
    table.add_column("O que faz", style="white")
    table.add_row("/ajuda", "Mostra esta lista")
    table.add_row("/indexar", "Indexa documentos na base de conhecimento")
    table.add_row("/status", "Mostra o status da base")
    table.add_row("/feedback bom|ruim", "Avalia a conversa atual (ajuda a treinar)")
    table.add_row("/sair", "Encerra e salva a conversa na memória")
    table.add_row("(frase)", "'cria um marp sobre X' gera uma apresentação")
    console.print(table)


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


def _handle_slides(engine, user_input, messages):
    """Trata pedido de slides Marp dentro do chat. Retorna True se tratou."""
    topic = detect_slides_request(user_input)
    if not topic:
        return False
    console.print("\n[bold cyan]LIV IA:[/bold cyan] gerando apresentação Marp...")
    try:
        with console.status("[cyan]Montando os slides...[/cyan]"):
            path = engine.generate_slides(topic)
        msg = f"Apresentação Marp salva em: {path}"
        console.print(f"[green]OK[/green] {msg}")
        console.print(f"[dim]Converter para PDF:[/dim] npx -y @marp-team/marp-cli {path} --pdf")
        messages.append({"role": "assistant", "content": msg})
    except Exception as e:
        console.print(f"[red]ERRO[/red] ao gerar slides: {e}")
    console.print("\n" + "─" * 60 + "\n")
    return True


def chat_loop():
    """Chat direto: entra sem menu; a 1ª mensagem define a especialidade.

    A conversa é corrida. Comandos começam com '/'. Ao sair, a conversa é
    salva em disco e indexada na memória de longo prazo (RAG), com o feedback
    informado — assim a LIV IA fica um pouco melhor a cada dia.
    """
    show_welcome()

    engine = LIVIAEngine()
    messages = []
    feedback = ""
    first_message = True

    try:
        while True:
            prompt_label = (
                "[bold green]Qual a especialidade/foco deste chat?[/bold green]"
                if first_message
                else "[bold green]Você[/bold green]"
            )
            user_input = Prompt.ask(f"\n{prompt_label}").strip()
            if not user_input:
                continue

            # --- Comandos especiais ---
            if user_input.startswith("/"):
                cmd, _, arg = user_input.partition(" ")
                cmd = cmd.lower()
                if cmd == "/sair":
                    break
                if cmd == "/ajuda":
                    show_help()
                    continue
                if cmd == "/indexar":
                    ingest_documents()
                    continue
                if cmd == "/status":
                    show_status()
                    continue
                if cmd == "/feedback":
                    feedback = arg.strip().lower() or "sem_avaliacao"
                    console.print(f"[green]OK[/green] Feedback registrado: [cyan]{feedback}[/cyan]")
                    continue
                console.print("[yellow]Comando desconhecido. Use /ajuda.[/yellow]")
                continue

            # --- 1ª mensagem define a especialidade/foco (persona) ---
            if first_message:
                engine.set_persona(user_input)
                first_message = False
                messages.append({"role": "system", "content": f"especialidade: {user_input}"})
                console.print(
                    f"\n[dim]Foco definido: [cyan]{user_input}[/cyan]. "
                    "Pode conversar normalmente.[/dim]\n"
                )
                continue

            messages.append({"role": "user", "content": user_input})

            # Pedido de slides Marp?
            if _handle_slides(engine, user_input, messages):
                continue

            console.print("\n[bold cyan]LIV IA:[/bold cyan]")
            with console.status("[cyan]Pensando...[/cyan]"):
                response = engine.chat(user_input)
            console.print(Markdown(response))
            messages.append({"role": "assistant", "content": response})
            console.print("\n" + "─" * 60 + "\n")

    except KeyboardInterrupt:
        console.print("\n")

    _finish_session(engine, messages, feedback)


def _finish_session(engine, messages, feedback):
    """Salva a conversa em disco e na memória de longo prazo ao encerrar."""
    console.print("\n[yellow]Encerrando...[/yellow]")

    # Considera apenas mensagens reais (ignora o marcador de especialidade).
    reais = [m for m in messages if m.get("role") in ("user", "assistant")]
    if not reais:
        console.print("[yellow]Até logo![/yellow]\n")
        return

    if Confirm.ask("\n[cyan]Salvar esta conversa?[/cyan]", default=True):
        filename = save_conversation(messages)
        console.print(f"[green]OK[/green] Conversa salva em: [cyan]{filename}[/cyan]")

        # Memória de longo prazo: indexa a conversa na base (seguro, via RAG).
        try:
            n = engine.remember_conversation(reais, feedback=feedback)
            if n:
                console.print(
                    "[green]OK[/green] Conversa adicionada à memória de longo prazo "
                    "(a LIV IA vai lembrar dela nas próximas sessões)."
                )
        except Exception as e:
            console.print(f"[dim]Memória não atualizada ({e}).[/dim]")

    console.print("[yellow]Até logo![/yellow]\n")


def ingest_documents():
    """Indexa documentos (PDF, Markdown, texto e código-fonte) na base única."""
    path = Prompt.ask("\n[cyan]Caminho da pasta com documentos[/cyan]", default="./docs")

    console.print(
        Panel.fit("[bold cyan]LIV IA[/bold cyan] - Processando documentos...", border_style="cyan")
    )

    processor = DocumentProcessor()
    try:
        with console.status("[cyan]Processando documentos...[/cyan]"):
            count = processor.ingest_directory(path)
        console.print(f"[green]OK[/green] {count} documento(s) indexado(s) na base!")
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


def main():
    """Função principal: entra direto no chat."""
    chat_loop()


if __name__ == "__main__":
    main()
