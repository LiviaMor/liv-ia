#!/bin/bash
# Script para baixar os modelos do Ollama

echo "🚀 Baixando modelos do Ollama..."

echo "📦 Baixando deepseek-coder-v2 (pode demorar ~8GB)..."
docker exec -it livia-ollama ollama pull deepseek-coder-v2

echo "📦 Baixando nomic-embed-text (~274MB)..."
docker exec -it livia-ollama ollama pull nomic-embed-text

echo "✅ Modelos instalados com sucesso!"
echo "📋 Listando modelos disponíveis:"
docker exec -it livia-ollama ollama list
