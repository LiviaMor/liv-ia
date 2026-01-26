# Script PowerShell para baixar os modelos do Ollama

Write-Host "🚀 Baixando modelos do Ollama..." -ForegroundColor Cyan

Write-Host "`n📦 Baixando deepseek-coder-v2 (pode demorar ~8GB)..." -ForegroundColor Yellow
docker exec -it livia-ollama ollama pull deepseek-coder-v2

Write-Host "`n📦 Baixando nomic-embed-text (~274MB)..." -ForegroundColor Yellow
docker exec -it livia-ollama ollama pull nomic-embed-text

Write-Host "`n✅ Modelos instalados com sucesso!" -ForegroundColor Green
Write-Host "`n📋 Listando modelos disponíveis:" -ForegroundColor Cyan
docker exec -it livia-ollama ollama list
