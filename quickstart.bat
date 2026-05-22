@echo off
REM Quick start script for LSTM + LLM Ablation Study on Windows

echo.
echo ========================================
echo LSTM + LLM Ablation Study - Quick Start
echo ========================================
echo.

REM Check if conda is available
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Conda not found. Please install Anaconda/Miniconda first.
    pause
    exit /b 1
)

REM Check if environment exists
conda info --envs | findstr /R "lstm_llm" >nul
if %errorlevel% equ 0 (
    echo Found existing lstm_llm environment. Activating...
) else (
    echo Creating new conda environment: lstm_llm
    call conda create -n lstm_llm python=3.10 -y
    if %errorlevel% neq 0 (
        echo Failed to create environment
        pause
        exit /b 1
    )
)

REM Activate environment
call conda activate lstm_llm
if %errorlevel% neq 0 (
    echo Failed to activate environment
    pause
    exit /b 1
)

echo Environment activated: lstm_llm
echo.

REM Install/upgrade pip
echo Installing/upgrading pip...
python -m pip install --upgrade pip -q

REM Install PyTorch
echo Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 -q

REM Install requirements
echo Installing other dependencies...
pip install -r requirements.txt -q

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.

REM Ask what to run
echo Choose an option:
echo 1. Run setup check (recommended first)
echo 2. Run quick demo (2-3 minutes)
echo 3. Run full ablation study (30+ minutes)
echo 4. Run only V1 and V2 (skip LLM)
echo 5. Exit
echo.

set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" (
    echo.
    python setup_check.py
    pause
) else if "%choice%"=="2" (
    echo.
    python demo.py
    pause
) else if "%choice%"=="3" (
    echo.
    python ablation_pipeline.py --variants all
    pause
) else if "%choice%"=="4" (
    echo.
    python ablation_pipeline.py --variants V1 V2
    pause
) else (
    echo Exiting...
)

echo.
pause
