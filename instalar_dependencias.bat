@echo off
echo =====================================
echo Instalando dependencias da IA
echo =====================================

cd /d %~dp0

where python >nul 2>&1
if errorlevel 1 (
    echo ❌ Python nao encontrado no PATH
    pause
    exit /b
)

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo ✅ Dependencias instaladas com sucesso
pause
