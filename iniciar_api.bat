@echo off
echo =====================================
echo Iniciando API da IA (RAG Atas)
echo =====================================

cd /d %~dp0

if not exist ".env" (
    echo ❌ Arquivo .env nao encontrado
    echo Crie um .env com OPENAI_API_KEY
    pause
    exit /b
)

python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

pause
