"""
LIV IA Slides - Geração de apresentações no formato Marp.

Marp (https://marp.app) transforma Markdown em slides. Este módulo monta um
Markdown Marp válido a partir de um título e de um conteúdo, salva localmente
e, opcionalmente, converte para PDF/HTML/PPTX usando o marp-cli via npx.

O conteúdo pode vir da LIV IA (gerado pelo LLM com base no RAG) ou de qualquer
texto. A separação de slides segue a convenção do Marp: `---` entre slides.
"""

import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

# Diretório padrão onde os slides são salvos (relativo ao projeto).
DEFAULT_SLIDES_DIR = "slides"

# Frontmatter padrão do Marp. 'marp: true' ativa o processamento.
DEFAULT_FRONTMATTER = {
    "marp": "true",
    "theme": "default",
    "paginate": "true",
}


def slugify(text: str) -> str:
    """Converte um título em um nome de arquivo seguro (sem acentos/espaços)."""
    text = text.strip().lower()
    # Troca acentos comuns do português por equivalentes ASCII.
    acentos = str.maketrans("áàâãäéèêëíìîïóòôõöúùûüç", "aaaaaeeeeiiiiooooouuuuc")
    text = text.translate(acentos)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "apresentacao"


def _build_frontmatter(frontmatter: dict) -> str:
    linhas = "\n".join(f"{k}: {v}" for k, v in frontmatter.items())
    return f"---\n{linhas}\n---\n"


def build_marp_markdown(title: str, content: str, frontmatter: dict | None = None) -> str:
    """Monta o Markdown Marp completo.

    Args:
        title: título da apresentação (vira o 1º slide).
        content: corpo. Se já contiver separadores '---' entre blocos, eles são
            respeitados; caso contrário, o conteúdo entra como um bloco único.
        frontmatter: sobrescreve/adiciona chaves ao frontmatter padrão.

    Returns:
        String com o Markdown Marp pronto para salvar.
    """
    fm = {**DEFAULT_FRONTMATTER, **(frontmatter or {})}
    header = _build_frontmatter(fm)

    # Slide de capa com o título.
    capa = f"# {title}\n"

    body = content.strip()
    # Garante que haja um separador entre a capa e o conteúdo.
    return f"{header}\n{capa}\n---\n\n{body}\n"


def save_slides(
    title: str,
    content: str,
    out_dir: str = DEFAULT_SLIDES_DIR,
    frontmatter: dict | None = None,
) -> Path:
    """Gera e salva um arquivo Marp .md localmente.

    Returns:
        Caminho (Path) do arquivo .md salvo.
    """
    markdown = build_marp_markdown(title, content, frontmatter)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{slugify(title)}_{timestamp}.md"
    file_path = out_path / filename
    file_path.write_text(markdown, encoding="utf-8")

    return file_path


def marp_cli_available() -> bool:
    """Diz se é possível converter slides (precisa de npx ou marp no PATH)."""
    return shutil.which("marp") is not None or shutil.which("npx") is not None


def convert(md_path: str, fmt: str = "pdf", timeout: int = 180) -> Path:
    """Converte um .md Marp para PDF/HTML/PPTX usando o marp-cli.

    Usa `marp` se estiver no PATH; senão cai para `npx @marp-team/marp-cli`.

    Args:
        md_path: caminho do arquivo .md Marp.
        fmt: 'pdf', 'html' ou 'pptx'.
        timeout: tempo máximo em segundos.

    Returns:
        Caminho do arquivo convertido.

    Raises:
        FileNotFoundError: se o .md não existir.
        RuntimeError: se não houver marp-cli disponível ou a conversão falhar.
    """
    md = Path(md_path)
    if not md.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {md_path}")

    fmt = fmt.lower()
    if fmt not in {"pdf", "html", "pptx"}:
        raise ValueError(f"Formato não suportado: {fmt} (use pdf, html ou pptx)")

    out = md.with_suffix(f".{fmt}")

    if shutil.which("marp"):
        cmd = ["marp", str(md), f"--{fmt}", "-o", str(out)]
    elif shutil.which("npx"):
        cmd = ["npx", "-y", "@marp-team/marp-cli", str(md), f"--{fmt}", "-o", str(out)]
    else:
        raise RuntimeError(
            "marp-cli não disponível. Instale com: npm i -g @marp-team/marp-cli "
            "ou garanta que o npx esteja no PATH."
        )

    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=timeout)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Falha na conversão Marp: {exc.stderr.decode(errors='ignore')}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Conversão Marp excedeu o tempo limite.")

    return out
