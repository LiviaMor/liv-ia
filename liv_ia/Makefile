.PHONY: help install test lint format clean docker-up docker-down

help:
	@echo "Comandos disponíveis:"
	@echo "  make install      - Instala dependências"
	@echo "  make test         - Executa todos os testes"
	@echo "  make test-unit    - Executa testes unitários"
	@echo "  make test-int     - Executa testes de integração"
	@echo "  make lint         - Verifica código com linters"
	@echo "  make format       - Formata código"
	@echo "  make clean        - Remove arquivos temporários"
	@echo "  make docker-up    - Inicia Ollama via Docker"
	@echo "  make docker-down  - Para Ollama"

install:
	pip install -r requirements.txt
	pip install pytest pytest-cov pytest-mock black isort flake8 safety bandit

test:
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-unit:
	pytest tests/unit -v

test-int:
	pytest tests/integration -v

lint:
	@echo "Running Black..."
	black --check .
	@echo "Running isort..."
	isort --check-only .
	@echo "Running Flake8..."
	flake8 .

format:
	black .
	isort .

security:
	@echo "Running Safety..."
	safety check
	@echo "Running Bandit..."
	bandit -r . -f screen

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .coverage

docker-up:
	docker-compose up -d
	@echo "Aguardando Ollama iniciar..."
	sleep 10
	docker logs livia-ollama-setup

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v
