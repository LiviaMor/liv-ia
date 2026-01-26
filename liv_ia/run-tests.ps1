# Script para executar testes no Windows

param(
    [string]$Type = "all"
)

Write-Host "LIV IA - Test Runner" -ForegroundColor Cyan
Write-Host "===================" -ForegroundColor Cyan
Write-Host ""

# Verifica se pytest está instalado
try {
    python -m pytest --version | Out-Null
} catch {
    Write-Host "ERRO: pytest não encontrado. Instalando dependências..." -ForegroundColor Red
    pip install pytest pytest-cov pytest-mock
}

switch ($Type) {
    "unit" {
        Write-Host "Executando testes unitários..." -ForegroundColor Yellow
        python -m pytest tests/unit -v
    }
    "integration" {
        Write-Host "Executando testes de integração..." -ForegroundColor Yellow
        Write-Host "AVISO: Certifique-se que o Ollama está rodando!" -ForegroundColor Yellow
        python -m pytest tests/integration -v
    }
    "coverage" {
        Write-Host "Executando testes com cobertura..." -ForegroundColor Yellow
        python -m pytest tests/ -v --cov=. --cov-report=html --cov-report=term
        Write-Host ""
        Write-Host "Relatório de cobertura gerado em: htmlcov/index.html" -ForegroundColor Green
    }
    "lint" {
        Write-Host "Verificando código..." -ForegroundColor Yellow
        
        Write-Host "  - Black..." -ForegroundColor Cyan
        python -m black --check .
        
        Write-Host "  - isort..." -ForegroundColor Cyan
        python -m isort --check-only .
        
        Write-Host "  - Flake8..." -ForegroundColor Cyan
        python -m flake8 .
    }
    "format" {
        Write-Host "Formatando código..." -ForegroundColor Yellow
        python -m black .
        python -m isort .
        Write-Host "Código formatado!" -ForegroundColor Green
    }
    default {
        Write-Host "Executando todos os testes..." -ForegroundColor Yellow
        python -m pytest tests/ -v --cov=. --cov-report=term
    }
}

Write-Host ""
Write-Host "Concluído!" -ForegroundColor Green
