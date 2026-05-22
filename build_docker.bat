@echo off
REM Build and run Docker image for LSTM + LLM Ablation Study (Windows)

setlocal enabledelayedexpansion

cls
echo.
echo ==================================================
echo LSTM + LLM Ablation Study - Docker Build Script
echo ==================================================
echo.

REM Check if Docker is installed
docker --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Step 1: Build image
echo [1/4] Building Docker image...
docker build -t lstm-llm:latest .

if errorlevel 1 (
    echo [ERROR] Build failed
    pause
    exit /b 1
)

echo [OK] Image built successfully
echo.

REM Step 2: Show image size
echo [2/4] Image information:
docker images lstm-llm:latest --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
echo.

REM Step 3: Run container
echo [3/4] Running container...
echo.
echo This will:
echo   1. Start Ollama service
echo   2. Download gemma:4b model (~2GB^)
echo   3. Run ablation_pipeline.py --variants all
echo.
echo Note: First run will take 10-15 minutes due to model download
echo.

set /p response="Continue? (y/n): "
if /i "!response!"=="y" (
    docker run -it --rm ^
        -v "%cd%\results:/app/results" ^
        -p 11434:11434 ^
        lstm-llm:latest
    
    echo.
    echo [OK] Pipeline completed
    echo.
    
    REM Step 4: Show results
    echo [4/4] Results saved to:
    echo   - results\ablation_study\ABLATION_COMPARISON.md
    echo   - results\ablation_study\plots\
) else (
    echo Cancelled
    exit /b 1
)

echo.
echo [OK] Done!
pause
