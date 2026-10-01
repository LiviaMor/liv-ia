"""
Testes unitários para geração de slides Marp (slides.py) e detecção de intenção.
"""

from pathlib import Path

import pytest

import slides
from brain import detect_slides_request


class TestSlugify:
    def test_remove_acentos_e_espacos(self):
        assert slides.slugify("Arquitetura Event-Driven") == "arquitetura-event-driven"
        assert slides.slugify("Introdução ao GPS") == "introducao-ao-gps"

    def test_fallback_vazio(self):
        assert slides.slugify("!!!") == "apresentacao"


class TestBuildMarp:
    def test_frontmatter_e_capa(self):
        md = slides.build_marp_markdown("Meu Tema", "## Slide 1\n- ponto")
        # Frontmatter Marp obrigatório
        assert md.startswith("---\n")
        assert "marp: true" in md
        assert "paginate: true" in md
        # Capa com o título
        assert "# Meu Tema" in md
        # Conteúdo preservado
        assert "## Slide 1" in md

    def test_frontmatter_customizado(self):
        md = slides.build_marp_markdown("T", "corpo", frontmatter={"theme": "gaia"})
        assert "theme: gaia" in md


class TestSaveSlides:
    def test_salva_arquivo_md(self, tmp_path):
        path = slides.save_slides("Node.js Streams", "## Streams\n- pipe", out_dir=str(tmp_path))
        assert path.exists()
        assert path.suffix == ".md"
        conteudo = path.read_text(encoding="utf-8")
        assert "marp: true" in conteudo
        assert "# Node.js Streams" in conteudo
        # Nome do arquivo usa o slug do título
        assert path.name.startswith("node-js-streams_")


class TestConvertGuards:
    def test_arquivo_inexistente(self):
        with pytest.raises(FileNotFoundError):
            slides.convert("/nao/existe.md", "pdf")

    def test_formato_invalido(self, tmp_path):
        md = tmp_path / "a.md"
        md.write_text("---\nmarp: true\n---\n# x\n", encoding="utf-8")
        with pytest.raises(ValueError):
            slides.convert(str(md), "docx")


class TestDetectSlidesRequest:
    def test_detecta_marp_sobre_tema(self):
        assert detect_slides_request("cria um marp sobre microsserviços") == "microsserviços"

    def test_detecta_apresentacao_de_tema(self):
        assert detect_slides_request("faça uma apresentação de GPS") == "GPS"

    def test_detecta_slides_para_tema(self):
        assert detect_slides_request("gere slides para Node.js") == "Node.js"

    def test_nao_detecta_pergunta_normal(self):
        assert detect_slides_request("como funciona o event loop?") is None

    def test_pedido_sem_sobre_usa_mensagem(self):
        # Menciona 'slides' mas sem 'sobre X' explícito -> usa a mensagem toda.
        resultado = detect_slides_request("quero slides")
        assert resultado is not None
