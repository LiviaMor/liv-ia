# LIV IA

Assistente de Arquitetura de Software Local com RAG (Retrieval Augmented Generation).

## Visão Geral

LIV IA é uma arquiteta de soluções sênior especializada em HealthTech, padrões de projeto e automação. Utiliza Ollama local com ChromaDB para consultar documentação técnica e fornecer respostas fundamentadas.

## Arquitetura

```
liv_ia/
├── main.py              # Interface CLI
├── brain.py             # Motor de IA (LangChain + Ollama)
├── ingestor.py          # Processador de PDFs
├── requirements.txt     # Dependências Python
├── docker-compose.yml   # Setup Ollama
└── docs/               # Documentos PDF
    └── conversas/      # Histórico (auto-gerado)
```

## Tecnologias

- **LLM**: Ollama (deepseek-coder-v2 ou qwen2.5-coder:7b)
- **Embeddings**: nomic-embed-text
- **Vector Store**: ChromaDB
- **Framework**: LangChain
- **Interface**: Rich (CLI)

## Instalação

### 1. Clonar e Instalar Dependências

```bash
cd liv_ia
pip install -r requirements.txt
```

### 2. Iniciar Ollama via Docker

```bash
docker-compose up -d
```

Aguarde o download dos modelos (primeira vez):
```bash
docker logs -f livia-ollama-setup
```

### 3. Verificar Modelos

```bash
docker exec -it livia-ollama ollama list
```

Deve mostrar:
- deepseek-coder-v2
- nomic-embed-text

## Uso

### Iniciar Aplicação

```bash
python main.py
```

### Menu Principal

```
1  Iniciar Chat Interativo
2  Fazer Pergunta Rápida
3  Indexar Documentos PDF
4  Ver Conversas Salvas
5  Status da Base de Conhecimento
6  Configurações
0  Sair
```

### Chat Interativo

Opções rápidas disponíveis:
1. Sugerir arquitetura
2. Analisar padrões
3. Boas práticas HealthTech
4. Segurança e LGPD
5. Otimização de performance
6. Escrever pergunta customizada

Sair: `Ctrl+C`

### Indexar Documentos

1. Coloque PDFs em `docs/`
2. Menu > Opção 3
3. Digite caminho: `./docs`

## Configuração

### Modelos

Editar `brain.py` linha 52:

```python
# Modelo padrão (preciso, lento)
LIVIAEngine(model_name="deepseek-coder-v2")

# Modelo alternativo (rápido, menor)
LIVIAEngine(model_name="qwen2.5-coder:7b")
```

### URL do Ollama

Editar `brain.py` linha 52:

```python
LIVIAEngine(ollama_base_url="http://localhost:11434")
```

## Funcionalidades

### RAG (Retrieval Augmented Generation)

- Indexa PDFs em chunks de 1000 caracteres
- Busca semântica com ChromaDB
- Retorna 4 documentos mais relevantes
- LLM sintetiza resposta fundamentada

### Histórico de Conversas

- Mantém últimas 6 mensagens (3 trocas)
- Salva conversas em JSON
- Localização: `docs/conversas/`

### Conhecimento Base

Funciona sem PDFs usando conhecimento do modelo.
PDFs complementam com informações específicas.

## Limitações Conhecidas

### Contexto

- Histórico: 6 mensagens (3 trocas)
- Contexto do modelo: ~32k tokens (~50 páginas)

### PDFs Grandes

- Recomendado: < 100MB por arquivo
- PDFs maiores: processar em lotes

### Performance

- Primeira resposta: lenta (carregamento do modelo)
- Respostas seguintes: mais rápidas
- Modelo menor (qwen): 3x mais rápido

## Troubleshooting

### Ollama não conecta

```bash
docker ps                    # Verificar se está rodando
docker-compose restart       # Reiniciar
docker logs livia-ollama     # Ver logs
```

### Modelos não baixaram

```bash
docker exec -it livia-ollama ollama pull deepseek-coder-v2
docker exec -it livia-ollama ollama pull nomic-embed-text
```

### Erro de memória

Use modelo menor:
```bash
docker exec -it livia-ollama ollama pull qwen2.5-coder:7b
```

Edite `brain.py` para usar `qwen2.5-coder:7b`.

### Respostas lentas

- Normal na primeira vez
- Considere modelo menor
- Aumente RAM do Docker Desktop

## Requisitos do Sistema

- Docker Desktop
- Python 3.12+
- 8GB RAM (16GB recomendado)
- 10GB espaço em disco

## Melhorias Futuras

### Alta Prioridade

- Streaming de PDFs grandes
- Sistema de resumos para contexto longo
- Modelo híbrido (rápido + preciso)
- Metadados e filtros de busca

### Média Prioridade

- Cache de embeddings
- Análise de código-fonte
- Geração de diagramas

### Baixa Prioridade

- API REST
- Interface Web

## Licença

MIT

## Versão

1.0.2
